#!/usr/bin/env python3
"""
벤치마크 학습용 데이터셋 구성 스크립트.

ControlNet 생성 결과를 CASDA 포맷으로 패키징하고,
기존 baseline_raw YOLO 데이터셋에 CASDA 합성 이미지를 merge하여
벤치마크 학습에 사용할 4개 데이터셋 그룹을 구성한다.

동작:
  Step 1 (Packaging):
    ControlNet 생성 이미지 + hint → augmented_dir/{casda_full, casda_pruning}
    (images/, masks/, metadata.json, packaging_report.json)

  Step 2 (YOLO Merge):
    baseline_raw/ 복사 → yolo_dir/{casda_full, casda_pruning}
    + CASDA 이미지/라벨을 train 에만 추가
    (val/test 는 baseline_raw 와 동일하게 유지 — 공정한 비교)

사전 조건:
  - baseline_raw/, baseline_trad/ YOLO 데이터셋이 이미 --baseline-dir 에 존재해야 함
  - ControlNet 생성 결과 (generated/*.png, generation_summary.json, hints/) 가 있어야 함

사용법 (Colab):
    PROJECT=/content/severstal-steel-defect-detection
    DRIVE=/content/drive/MyDrive/data/Severstal

    python ${PROJECT}/scripts/prepare_benchmark_data.py \\
        --generated-dir ${DRIVE}/test_results_v5.1/phase1_basic/generated \\
        --summary-json ${DRIVE}/test_results_v5.1/phase1_basic/generation_summary.json \\
        --hint-dir ${DRIVE}/controlnet_dataset_v5.1/hints \\
        --augmented-dir ${DRIVE}/augmented_v5.1 \\
        --baseline-dir ${DRIVE}/yolo_datasets/baseline_raw \\
        --yolo-dir ${DRIVE}/yolo_datasets \\
        --suitability-threshold 0.60 \\
        --pruning-top-k 2000

    이후 벤치마크 실행:
    python ${PROJECT}/scripts/run_benchmark.py \\
        --config ${PROJECT}/configs/benchmark_experiment.yaml \\
        --yolo-dir ${DRIVE}/yolo_datasets \\
        --casda-dir ${DRIVE}/augmented_v5.1 \\
        --data-dir ${DRIVE}/train_images \\
        --csv ${DRIVE}/train.csv \\
        --split-csv ${DRIVE}/casda/splits/split_70_15_15_seed42.csv \\
        --output-dir ${DRIVE}/benchmark_results_v5.1 \\
        --groups all
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# YOLO Label Generation (mask → bbox)
# ============================================================================

def generate_yolo_label_from_mask(
    mask_path: str,
    class_id: int,
    min_area: int = 16,
) -> str:
    """
    마스크 이미지에서 contour 기반 YOLO bbox 라벨을 생성한다.

    Args:
        mask_path: 그레이스케일 마스크 이미지 경로 (0 or 255)
        class_id: 0-indexed 클래스 ID
        min_area: 최소 bbox 면적 (이하 필터링, 기본 16px)

    Returns:
        YOLO 포맷 라벨 문자열 (빈 문자열 = 유효한 contour 없음)
    """
    import cv2

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw * bh >= min_area:
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return "\n".join(lines)


def generate_yolo_label_fullimage(class_id: int) -> str:
    """마스크 없을 때 전체 이미지를 bbox로 하는 fallback 라벨."""
    return f"{class_id} 0.500000 0.500000 1.000000 1.000000"


# ============================================================================
# Step 2: YOLO Dataset Merge
# ============================================================================

def copy_baseline_to_group(
    baseline_dir: Path,
    group_dir: Path,
) -> None:
    """
    baseline_raw YOLO 디렉토리를 group_dir 로 전체 복사한다.
    (images/{train,val,test}/, labels/{train,val,test}/, dataset.yaml)

    이미 존재하면 덮어쓰지 않고 에러를 발생시킨다.
    --force 옵션으로 사전 삭제 후 호출해야 한다.
    """
    if group_dir.exists():
        raise FileExistsError(
            f"Target directory already exists: {group_dir}\n"
            f"  Use --force to remove and recreate."
        )

    logging.info(f"  Copying baseline: {baseline_dir} → {group_dir}")
    shutil.copytree(str(baseline_dir), str(group_dir))
    logging.info(f"  Copied baseline dataset ({_count_files(group_dir)} files)")


def _count_files(directory: Path) -> int:
    """디렉토리 내 전체 파일 수."""
    return sum(1 for _ in directory.rglob("*") if _.is_file())


def add_casda_to_yolo_train(
    casda_data_dir: Path,
    yolo_group_dir: Path,
    group_name: str,
) -> int:
    """
    CASDA 패키징 디렉토리의 이미지/마스크를 YOLO train 세트에 추가한다.

    Args:
        casda_data_dir: augmented_dir/casda_full 또는 casda_pruning
                        (images/, masks/, metadata.json 포함)
        yolo_group_dir: yolo_dir/casda_full 또는 casda_pruning
                        (baseline_raw 복사본, images/train/ 에 추가)
        group_name: 로깅용 그룹 이름 (e.g. "casda_full")

    Returns:
        추가된 이미지 수
    """
    # 대상 디렉토리
    images_train = yolo_group_dir / "images" / "train"
    labels_train = yolo_group_dir / "labels" / "train"

    if not images_train.exists() or not labels_train.exists():
        raise FileNotFoundError(
            f"YOLO train directories not found in {yolo_group_dir}.\n"
            f"  Expected: images/train/ and labels/train/"
        )

    # metadata.json 로드
    meta_path = casda_data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {casda_data_dir}.\n"
            f"  Run Step 1 (packaging) first."
        )

    with open(meta_path) as f:
        samples = json.load(f)

    logging.info(f"  [{group_name}] Adding {len(samples)} CASDA images to train/")

    added = 0
    skipped = 0

    for idx, sample in enumerate(samples):
        # 이미지 경로 해석 (relative → absolute)
        img_rel = sample.get("image_path", "")
        img_src = Path(img_rel) if os.path.isabs(img_rel) else (casda_data_dir / img_rel)

        if not img_src.exists():
            logging.warning(f"    Image not found: {img_src}")
            skipped += 1
            continue

        # 고유 파일명 생성 (충돌 방지)
        dst_name = f"casda_{idx:05d}_{img_src.name}"
        dst_img = images_train / dst_name

        # 이미지 복사 (symlink 우선, 실패 시 copy)
        if not dst_img.exists():
            try:
                os.symlink(img_src.resolve(), dst_img)
            except (OSError, NotImplementedError):
                shutil.copy2(str(img_src), str(dst_img))

        # 라벨 생성
        label_name = Path(dst_name).stem + ".txt"
        dst_lbl = labels_train / label_name

        class_id = sample.get("class_id", 0)

        if "mask_path" in sample:
            mask_rel = sample["mask_path"]
            mask_path = Path(mask_rel) if os.path.isabs(mask_rel) else (casda_data_dir / mask_rel)

            if mask_path.exists():
                label_text = generate_yolo_label_from_mask(
                    str(mask_path), class_id
                )
            else:
                label_text = generate_yolo_label_fullimage(class_id)
        else:
            label_text = generate_yolo_label_fullimage(class_id)

        with open(dst_lbl, "w") as f:
            f.write(label_text + "\n" if label_text else "")

        added += 1

    logging.info(f"  [{group_name}] Added {added} images, skipped {skipped}")
    return added


def update_dataset_yaml(yolo_group_dir: Path) -> None:
    """
    dataset.yaml 의 path 를 현재 group_dir 절대경로로 갱신한다.
    baseline_raw 에서 복사했으므로 path 가 원래 baseline_raw 를 가리키고 있다.
    """
    yaml_path = yolo_group_dir / "dataset.yaml"
    if not yaml_path.exists():
        logging.warning(f"  dataset.yaml not found in {yolo_group_dir}")
        return

    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    data["path"] = str(yolo_group_dir.resolve())

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logging.info(f"  Updated dataset.yaml path → {yolo_group_dir.resolve()}")


def merge_yolo_dataset(
    baseline_dir: Path,
    casda_data_dir: Path,
    yolo_group_dir: Path,
    group_name: str,
    force: bool = False,
) -> dict:
    """
    baseline_raw 를 복사하고 CASDA 데이터를 train 에 merge 한다.

    Returns:
        통계 dict {baseline_images, casda_added, total_train}
    """
    # 강제 재생성
    if force and yolo_group_dir.exists():
        logging.info(f"  [--force] Removing existing: {yolo_group_dir}")
        shutil.rmtree(str(yolo_group_dir))

    # Step 2a: baseline 복사
    copy_baseline_to_group(baseline_dir, yolo_group_dir)

    # baseline train 이미지 수
    train_img_dir = yolo_group_dir / "images" / "train"
    baseline_count = len(list(train_img_dir.glob("*")))

    # Step 2b: CASDA 이미지 추가
    casda_added = add_casda_to_yolo_train(
        casda_data_dir=casda_data_dir,
        yolo_group_dir=yolo_group_dir,
        group_name=group_name,
    )

    # Step 2c: dataset.yaml path 갱신
    update_dataset_yaml(yolo_group_dir)

    total_train = len(list(train_img_dir.glob("*")))

    return {
        "baseline_train_images": baseline_count,
        "casda_added": casda_added,
        "total_train_images": total_train,
    }


# ============================================================================
# Validation
# ============================================================================

def validate_baseline_dir(baseline_dir: Path) -> None:
    """baseline_raw 디렉토리가 유효한 YOLO 데이터셋인지 검증."""
    required = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "dataset.yaml",
    ]
    missing = []
    for rel in required:
        p = baseline_dir / rel
        if not p.exists():
            missing.append(rel)

    if missing:
        raise FileNotFoundError(
            f"baseline_raw directory is incomplete: {baseline_dir}\n"
            f"  Missing: {', '.join(missing)}\n"
            f"  Expected structure: images/{{train,val,test}}/, labels/{{train,val,test}}/, dataset.yaml"
        )


def validate_packaging_output(augmented_dir: Path) -> None:
    """패키징 결과 디렉토리 검증."""
    for group in ["casda_full", "casda_pruning"]:
        group_dir = augmented_dir / group
        meta = group_dir / "metadata.json"
        imgs = group_dir / "images"
        if not meta.exists():
            raise FileNotFoundError(f"Packaging output missing: {meta}")
        if not imgs.exists() or not any(imgs.iterdir()):
            raise FileNotFoundError(f"No images in packaging output: {imgs}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="벤치마크 학습용 데이터셋 구성 (CASDA packaging + YOLO merge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
동작:
  Step 1: ControlNet 생성 결과를 CASDA 포맷으로 패키징
          → augmented_dir/{casda_full, casda_pruning}/
  Step 2: baseline_raw 복사 + CASDA train merge
          → yolo_dir/{casda_full, casda_pruning}/

사용 예시 (Colab):
  PROJECT=/content/severstal-steel-defect-detection
  DRIVE=/content/drive/MyDrive/data/Severstal

  python ${PROJECT}/scripts/prepare_benchmark_data.py \\
      --generated-dir ${DRIVE}/test_results_v5.1/phase1_basic/generated \\
      --summary-json ${DRIVE}/test_results_v5.1/phase1_basic/generation_summary.json \\
      --hint-dir ${DRIVE}/controlnet_dataset_v5.1/hints \\
      --augmented-dir ${DRIVE}/augmented_v5.1 \\
      --baseline-dir ${DRIVE}/yolo_datasets/baseline_raw \\
      --yolo-dir ${DRIVE}/yolo_datasets \\
      --suitability-threshold 0.60 \\
      --pruning-top-k 2000
        """,
    )

    # Step 1: Packaging 관련 인자
    step1 = parser.add_argument_group("Step 1: CASDA Packaging")
    step1.add_argument(
        "--generated-dir", type=str, required=True,
        help="ControlNet 생성 이미지 디렉토리 (*.png)",
    )
    step1.add_argument(
        "--summary-json", type=str, required=True,
        help="generation_summary.json 경로",
    )
    step1.add_argument(
        "--hint-dir", type=str, required=True,
        help="hint 이미지 디렉토리 (*_hint.png)",
    )
    step1.add_argument(
        "--augmented-dir", type=str, required=True,
        help="패키징 출력 디렉토리 (casda_full/, casda_pruning/ 생성)",
    )
    step1.add_argument(
        "--quality-json", type=str, default=None,
        help="별도 quality score JSON 파일 (옵션)",
    )
    step1.add_argument(
        "--default-score", type=float, default=1.0,
        help="quality score 없을 때 기본값 (기본: 1.0)",
    )
    step1.add_argument(
        "--suitability-threshold", type=float, default=0.60,
        help="pruning 최소 quality score 임계값 (기본: 0.60)",
    )
    step1.add_argument(
        "--pruning-top-k", type=int, default=2000,
        help="pruning 최대 이미지 수 (기본: 2000)",
    )
    step1.add_argument(
        "--mask-threshold", type=int, default=127,
        help="hint Red 채널 이진화 임계값 (기본: 127)",
    )

    # Step 2: YOLO Merge 관련 인자
    step2 = parser.add_argument_group("Step 2: YOLO Dataset Merge")
    step2.add_argument(
        "--baseline-dir", type=str, required=True,
        help="baseline_raw YOLO 데이터셋 경로 (images/, labels/, dataset.yaml)",
    )
    step2.add_argument(
        "--yolo-dir", type=str, required=True,
        help="YOLO 데이터셋 출력 상위 디렉토리 (casda_full/, casda_pruning/ 생성)",
    )

    # 공통 옵션
    common = parser.add_argument_group("공통 옵션")
    common.add_argument(
        "--force", action="store_true",
        help="기존 출력 디렉토리 삭제 후 재생성",
    )
    common.add_argument(
        "--skip-packaging", action="store_true",
        help="Step 1 (패키징) 건너뛰기 (이미 완료된 경우)",
    )
    common.add_argument(
        "--skip-yolo", action="store_true",
        help="Step 2 (YOLO merge) 건너뛰기 (패키징만 하고 싶을 때)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    total_start = time.time()

    logging.info("=" * 70)
    logging.info("Benchmark Data Preparation")
    logging.info("=" * 70)

    augmented_dir = Path(args.augmented_dir)
    baseline_dir = Path(args.baseline_dir)
    yolo_dir = Path(args.yolo_dir)

    # ================================================================
    # Step 1: CASDA Packaging
    # ================================================================
    if not args.skip_packaging:
        logging.info("")
        logging.info("=" * 70)
        logging.info("Step 1: CASDA Packaging")
        logging.info("=" * 70)
        logging.info(f"  Generated dir : {args.generated_dir}")
        logging.info(f"  Summary JSON  : {args.summary_json}")
        logging.info(f"  Hint dir      : {args.hint_dir}")
        logging.info(f"  Output dir    : {augmented_dir}")
        logging.info(f"  Threshold     : {args.suitability_threshold}")
        logging.info(f"  Top-K         : {args.pruning_top_k}")

        # 강제 재생성
        if args.force:
            for sub in ["casda_full", "casda_pruning"]:
                sub_dir = augmented_dir / sub
                if sub_dir.exists():
                    logging.info(f"  [--force] Removing: {sub_dir}")
                    shutil.rmtree(str(sub_dir))

        from scripts.package_casda_data import package_data

        step1_start = time.time()
        package_data(
            generated_dir=Path(args.generated_dir),
            summary_json=Path(args.summary_json),
            hint_dir=Path(args.hint_dir),
            output_dir=augmented_dir,
            quality_json=Path(args.quality_json) if args.quality_json else None,
            suitability_threshold=args.suitability_threshold,
            pruning_top_k=args.pruning_top_k,
            mask_threshold=args.mask_threshold,
            default_score=args.default_score,
        )
        step1_time = time.time() - step1_start
        logging.info(f"Step 1 completed in {step1_time:.1f}s")
    else:
        logging.info("")
        logging.info("[SKIP] Step 1: --skip-packaging specified")

    # 패키징 결과 검증
    logging.info("")
    logging.info("Validating packaging output...")
    validate_packaging_output(augmented_dir)
    logging.info("  Packaging output validated OK")

    # ================================================================
    # Step 2: YOLO Dataset Merge
    # ================================================================
    if not args.skip_yolo:
        logging.info("")
        logging.info("=" * 70)
        logging.info("Step 2: YOLO Dataset Merge")
        logging.info("=" * 70)
        logging.info(f"  Baseline dir  : {baseline_dir}")
        logging.info(f"  YOLO dir      : {yolo_dir}")
        logging.info(f"  Force         : {args.force}")

        # baseline 검증
        validate_baseline_dir(baseline_dir)
        logging.info("  Baseline directory validated OK")

        yolo_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        groups = [
            ("casda_full", augmented_dir / "casda_full"),
            ("casda_pruning", augmented_dir / "casda_pruning"),
        ]

        for group_name, casda_data_dir in groups:
            logging.info("")
            logging.info(f"--- {group_name} ---")

            yolo_group_dir = yolo_dir / group_name

            step2_start = time.time()
            try:
                stats = merge_yolo_dataset(
                    baseline_dir=baseline_dir,
                    casda_data_dir=casda_data_dir,
                    yolo_group_dir=yolo_group_dir,
                    group_name=group_name,
                    force=args.force,
                )
                results[group_name] = stats
                step2_time = time.time() - step2_start
                logging.info(
                    f"  [OK] {group_name}: "
                    f"baseline={stats['baseline_train_images']}, "
                    f"casda=+{stats['casda_added']}, "
                    f"total_train={stats['total_train_images']} "
                    f"({step2_time:.1f}s)"
                )
            except Exception as e:
                logging.error(f"  [FAIL] {group_name}: {e}")
                results[group_name] = None

        # 결과 요약 리포트 저장
        report_path = yolo_dir / "prepare_benchmark_data_report.json"
        report = {
            "baseline_dir": str(baseline_dir),
            "augmented_dir": str(augmented_dir),
            "yolo_dir": str(yolo_dir),
            "suitability_threshold": args.suitability_threshold,
            "pruning_top_k": args.pruning_top_k,
            "groups": {},
        }
        for gname, stats in results.items():
            if stats:
                report["groups"][gname] = stats
            else:
                report["groups"][gname] = {"status": "FAILED"}

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"\nMerge report saved to: {report_path}")
    else:
        logging.info("")
        logging.info("[SKIP] Step 2: --skip-yolo specified")

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - total_start

    logging.info("")
    logging.info("=" * 70)
    logging.info(f"Benchmark data preparation complete ({total_time:.1f}s)")
    logging.info("=" * 70)

    # YOLO 데이터셋 현황
    logging.info("")
    logging.info("YOLO datasets in %s:", yolo_dir)
    for group in ["baseline_raw", "baseline_trad", "casda_full", "casda_pruning"]:
        group_path = yolo_dir / group
        if group_path.exists():
            yaml_file = group_path / "dataset.yaml"
            status = "OK" if yaml_file.exists() else "NO dataset.yaml"
            train_dir = group_path / "images" / "train"
            train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
            logging.info(f"  {group:<20s} [{status}] train={train_count} images")
        else:
            logging.info(f"  {group:<20s} [NOT FOUND]")

    logging.info("")
    logging.info("다음 단계: 벤치마크 실행")
    logging.info(
        "  python scripts/run_benchmark.py \\\n"
        "      --config configs/benchmark_experiment.yaml \\\n"
        f"      --yolo-dir {yolo_dir} \\\n"
        f"      --casda-dir {augmented_dir} \\\n"
        "      --data-dir <train_images_path> \\\n"
        "      --csv <train.csv_path> \\\n"
        "      --split-csv <split.csv_path> \\\n"
        "      --output-dir <benchmark_results_path> \\\n"
        "      --groups all"
    )


if __name__ == "__main__":
    main()
