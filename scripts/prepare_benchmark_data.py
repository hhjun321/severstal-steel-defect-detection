#!/usr/bin/env python3
"""
[DEPRECATED] 벤치마크 학습용 YOLO 데이터셋 구성 스크립트.

** 이 스크립트는 더 이상 사용되지 않습니다. **
run_benchmark.py가 inject/clean 방식으로 CASDA 이미지를 baseline_raw에
직접 주입/삭제하므로, 별도의 YOLO 데이터셋 생성이 불필요합니다.

대신 사용:
  python scripts/run_benchmark.py \\
      --config configs/benchmark_experiment.yaml \\
      --yolo-dir ${DRIVE}/yolo_datasets \\
      --casda-dir ${DRIVE}/augmented_dataset_v5.1 \\
      ...

이전 동작 (참고용):
  이미 패키징 완료된 CASDA 합성 이미지(augmented_dataset_v5.1/)를
baseline_raw YOLO 데이터셋에 merge하여 벤치마크 학습용 데이터셋을 생성한다.

동작:
  baseline_raw/ 구조를 심볼릭 링크로 복제 → yolo_dir/{casda_full, casda_pruning}/
  + CASDA 이미지/라벨을 train/에만 추가
  (val/test는 baseline_raw와 동일하게 유지 — 공정한 비교)

사전 조건:
  - baseline_raw/ YOLO 데이터셋이 --baseline-dir에 이미 존재해야 함
  - CASDA 패키징 결과(casda_full/, casda_pruning/)가 --augmented-dir에 존재해야 함
    각 하위 디렉토리에 images/, masks/, metadata.json이 포함되어 있어야 함

사용법 (Colab):
    PROJECT=/content/severstal-steel-defect-detection
    DRIVE=/content/drive/MyDrive/data/Severstal

    python ${PROJECT}/scripts/prepare_benchmark_data.py \\
        --augmented-dir ${DRIVE}/augmented_dataset_v5.1 \\
        --baseline-dir ${DRIVE}/yolo_datasets/baseline_raw \\
        --yolo-dir ${DRIVE}/yolo_datasets \\
        --force

    이후 벤치마크 실행:
    python ${PROJECT}/scripts/run_benchmark.py \\
        --config ${PROJECT}/configs/benchmark_experiment.yaml \\
        --yolo-dir ${DRIVE}/yolo_datasets \\
        --casda-dir ${DRIVE}/augmented_dataset_v5.1 \\
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

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# 생성할 YOLO 데이터셋 그룹 목록
GROUPS = ["casda_full", "casda_pruning"]
SPLITS = ["train", "val", "test"]


# ============================================================================
# Baseline 심볼릭 링크 복제
# ============================================================================

def symlink_baseline_to_group(baseline_dir: Path, group_dir: Path) -> dict:
    """
    baseline_raw YOLO 디렉토리를 group_dir에 심볼릭 링크로 복제한다.

    디렉토리 구조(images/{train,val,test}/, labels/{train,val,test}/)를 생성하고,
    각 파일을 os.symlink()으로 연결한다. dataset.yaml은 수정이 필요하므로 실제 복사.

    Args:
        baseline_dir: baseline_raw YOLO 데이터셋 경로
        group_dir: 생성할 그룹 디렉토리 경로

    Returns:
        split별 이미지/라벨 수 통계 dict
    """
    if group_dir.exists():
        raise FileExistsError(
            f"Target directory already exists: {group_dir}\n"
            f"  Use --force to remove and recreate."
        )

    logger.info(f"  Symlink baseline: {baseline_dir} -> {group_dir}")

    stats = {}

    for category in ["images", "labels"]:
        for split in SPLITS:
            src_dir = baseline_dir / category / split
            dst_dir = group_dir / category / split
            dst_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for src_file in sorted(src_dir.iterdir()):
                if src_file.is_file():
                    dst_file = dst_dir / src_file.name
                    try:
                        os.symlink(src_file.resolve(), dst_file)
                    except (OSError, NotImplementedError):
                        shutil.copy2(str(src_file), str(dst_file))
                    count += 1

            key = f"{category}_{split}"
            stats[key] = count

    # dataset.yaml은 실제 복사 (path 필드 수정 필요)
    yaml_src = baseline_dir / "dataset.yaml"
    yaml_dst = group_dir / "dataset.yaml"
    if yaml_src.exists():
        shutil.copy2(str(yaml_src), str(yaml_dst))

    logger.info(
        f"  Symlinked: "
        f"train={stats.get('images_train', 0)} imgs / "
        f"{stats.get('labels_train', 0)} lbls, "
        f"val={stats.get('images_val', 0)}, "
        f"test={stats.get('images_test', 0)}"
    )

    return stats


# ============================================================================
# dataset.yaml 갱신
# ============================================================================

def update_dataset_yaml(group_dir: Path) -> None:
    """
    dataset.yaml의 path 필드를 현재 group_dir 절대경로로 갱신한다.
    baseline_raw에서 복사했으므로 path가 원래 baseline_raw를 가리키고 있다.
    """
    import yaml

    yaml_path = group_dir / "dataset.yaml"
    if not yaml_path.exists():
        logger.warning(f"  dataset.yaml not found in {group_dir}")
        return

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    data["path"] = str(group_dir.resolve())

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"  Updated dataset.yaml path -> {group_dir.resolve()}")


# ============================================================================
# 단일 그룹 Merge
# ============================================================================

def merge_single_group(
    baseline_dir: Path,
    casda_data_dir: Path,
    group_dir: Path,
    group_name: str,
    num_classes: int = 4,
    force: bool = False,
) -> dict:
    """
    하나의 데이터셋 그룹을 생성한다.

    1. baseline_raw를 심볼릭 링크로 복제
    2. CASDA 합성 이미지를 train/에 추가 (_add_casda_to_training)
    3. dataset.yaml path 갱신
    4. validate_yolo_dataset으로 검증

    Args:
        baseline_dir: baseline_raw YOLO 데이터셋
        casda_data_dir: augmented_dir/casda_full 또는 casda_pruning
        group_dir: 출력 그룹 디렉토리 (yolo_dir/casda_full 등)
        group_name: 로깅용 그룹 이름
        num_classes: 클래스 수 (기본 4)
        force: 기존 출력 삭제 후 재생성

    Returns:
        통계 dict
    """
    from src.training.dataset_yolo import (
        _add_casda_to_training,
        validate_yolo_dataset,
    )

    # 강제 재생성
    if force and group_dir.exists():
        logger.info(f"  [--force] Removing existing: {group_dir}")
        shutil.rmtree(str(group_dir))

    # (1) baseline 심볼릭 링크 복제
    baseline_stats = symlink_baseline_to_group(baseline_dir, group_dir)
    baseline_train_count = baseline_stats.get("images_train", 0)

    # (2) CASDA 이미지 추가
    # casda_mode="full": 추가 필터링 없음
    # (casda_pruning은 패키징 단계에서 이미 threshold=0.60 필터링 완료)
    images_train_dir = group_dir / "images" / "train"
    labels_train_dir = group_dir / "labels" / "train"

    casda_config = {}  # 필터링 불필요 — mode="full"
    casda_added = _add_casda_to_training(
        casda_dir=str(casda_data_dir),
        casda_mode="full",
        casda_config=casda_config,
        images_train_dir=images_train_dir,
        labels_train_dir=labels_train_dir,
        num_classes=num_classes,
    )

    # (3) dataset.yaml path 갱신
    update_dataset_yaml(group_dir)

    # (4) 검증
    yaml_path = validate_yolo_dataset(
        yolo_dir=str(group_dir),
    )
    if yaml_path:
        logger.info(f"  Validation: OK ({yaml_path})")
    else:
        logger.warning(f"  Validation: FAILED — dataset may be incomplete")

    # 통계
    total_train = len(list(images_train_dir.glob("*")))

    return {
        "baseline_train_images": baseline_train_count,
        "casda_added": casda_added,
        "total_train_images": total_train,
        "val_images": baseline_stats.get("images_val", 0),
        "test_images": baseline_stats.get("images_test", 0),
        "validated": yaml_path is not None,
    }


# ============================================================================
# 입력 검증
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
    missing = [rel for rel in required if not (baseline_dir / rel).exists()]

    if missing:
        raise FileNotFoundError(
            f"baseline_raw directory is incomplete: {baseline_dir}\n"
            f"  Missing: {', '.join(missing)}\n"
            f"  Expected: images/{{train,val,test}}/, labels/{{train,val,test}}/, dataset.yaml"
        )


def validate_augmented_dir(augmented_dir: Path) -> None:
    """패키징 결과 디렉토리 검증 (casda_full, casda_pruning 존재 확인)."""
    for group in GROUPS:
        group_dir = augmented_dir / group
        meta = group_dir / "metadata.json"
        imgs = group_dir / "images"
        if not meta.exists():
            raise FileNotFoundError(
                f"Packaging output missing: {meta}\n"
                f"  package_casda_data.py를 먼저 실행하세요."
            )
        if not imgs.exists() or not any(imgs.iterdir()):
            raise FileNotFoundError(
                f"No images in packaging output: {imgs}"
            )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="벤치마크 학습용 YOLO 데이터셋 구성 (baseline_raw + CASDA merge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
동작:
  baseline_raw YOLO 디렉토리를 심볼릭 링크로 복제한 뒤,
  CASDA 합성 이미지/라벨을 train/에만 추가하여 2개 데이터셋 그룹을 생성한다.
  val/test는 baseline_raw와 동일 (심볼릭 링크) — 공정한 비교 보장.

생성되는 그룹:
  casda_full     — baseline_raw + CASDA 전체 이미지
  casda_pruning  — baseline_raw + CASDA 품질 선별 이미지 (threshold >= 0.60)

사용 예시 (Colab):
  PROJECT=/content/severstal-steel-defect-detection
  DRIVE=/content/drive/MyDrive/data/Severstal

  python ${PROJECT}/scripts/prepare_benchmark_data.py \\
      --augmented-dir ${DRIVE}/augmented_dataset_v5.1 \\
      --baseline-dir ${DRIVE}/yolo_datasets/baseline_raw \\
      --yolo-dir ${DRIVE}/yolo_datasets \\
      --force
        """,
    )

    parser.add_argument(
        "--augmented-dir", type=str, required=True,
        help="CASDA 패키징 출력 디렉토리 (casda_full/, casda_pruning/ 포함)",
    )
    parser.add_argument(
        "--baseline-dir", type=str, required=True,
        help="baseline_raw YOLO 데이터셋 경로 (images/, labels/, dataset.yaml)",
    )
    parser.add_argument(
        "--yolo-dir", type=str, required=True,
        help="YOLO 데이터셋 출력 상위 디렉토리 (casda_full/, casda_pruning/ 생성)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=4,
        help="결함 클래스 수 (기본: 4)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="기존 출력 디렉토리 삭제 후 재생성",
    )

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    total_start = time.time()

    augmented_dir = Path(args.augmented_dir)
    baseline_dir = Path(args.baseline_dir)
    yolo_dir = Path(args.yolo_dir)

    logger.info("=" * 70)
    logger.info("Benchmark YOLO Dataset Preparation")
    logger.info("=" * 70)
    logger.info(f"  Augmented dir : {augmented_dir}")
    logger.info(f"  Baseline dir  : {baseline_dir}")
    logger.info(f"  YOLO dir      : {yolo_dir}")
    logger.info(f"  Num classes   : {args.num_classes}")
    logger.info(f"  Force         : {args.force}")

    # ================================================================
    # 입력 검증
    # ================================================================
    logger.info("")
    logger.info("Validating inputs...")

    validate_baseline_dir(baseline_dir)
    logger.info("  baseline_raw: OK")

    validate_augmented_dir(augmented_dir)
    logger.info("  augmented_dir: OK")

    # 패키징 통계 출력
    for group in GROUPS:
        meta_path = augmented_dir / group / "metadata.json"
        with open(meta_path) as f:
            samples = json.load(f)
        logger.info(f"  {group}: {len(samples)} images in metadata.json")

    # ================================================================
    # YOLO 데이터셋 Merge
    # ================================================================
    yolo_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for group_name in GROUPS:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Group: {group_name}")
        logger.info("=" * 70)

        casda_data_dir = augmented_dir / group_name
        group_dir = yolo_dir / group_name

        group_start = time.time()
        try:
            stats = merge_single_group(
                baseline_dir=baseline_dir,
                casda_data_dir=casda_data_dir,
                group_dir=group_dir,
                group_name=group_name,
                num_classes=args.num_classes,
                force=args.force,
            )
            results[group_name] = stats
            elapsed = time.time() - group_start
            logger.info(
                f"  [OK] {group_name}: "
                f"baseline={stats['baseline_train_images']}, "
                f"casda=+{stats['casda_added']}, "
                f"total_train={stats['total_train_images']} "
                f"({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(f"  [FAIL] {group_name}: {e}", exc_info=True)
            results[group_name] = {"status": "FAILED", "error": str(e)}

    # ================================================================
    # 리포트 저장
    # ================================================================
    report_path = yolo_dir / "prepare_benchmark_report.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "augmented_dir": str(augmented_dir),
        "baseline_dir": str(baseline_dir),
        "yolo_dir": str(yolo_dir),
        "num_classes": args.num_classes,
        "groups": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\nReport saved: {report_path}")

    # ================================================================
    # 최종 요약
    # ================================================================
    total_time = time.time() - total_start

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Benchmark data preparation complete ({total_time:.1f}s)")
    logger.info("=" * 70)

    # YOLO 데이터셋 현황 테이블
    logger.info("")
    logger.info("YOLO datasets in %s:", yolo_dir)
    for group in ["baseline_raw", "baseline_trad", "casda_full", "casda_pruning"]:
        group_path = yolo_dir / group
        if group_path.exists():
            yaml_file = group_path / "dataset.yaml"
            status = "OK" if yaml_file.exists() else "NO yaml"
            train_dir = group_path / "images" / "train"
            count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
            logger.info(f"  {group:<20s} [{status}] train={count} images")
        else:
            logger.info(f"  {group:<20s} [NOT FOUND]")

    # 다음 단계 안내
    logger.info("")
    logger.info("Next: run benchmark")
    logger.info(
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
