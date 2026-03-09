#!/usr/bin/env python3
"""
CopyPaste Baseline Generator

원본 결함 ROI 패치를 결함 없는 배경 이미지에 Poisson Blending 없이 직접 붙여넣어
CopyPaste baseline 데이터를 생성합니다.

ControlNet 없이 단순 mask-based paste만 수행. CASDA와의 차이:
- ControlNet 생성 이미지 미사용 (원본 ROI 패치 사용)
- Poisson Blending 미적용 (직접 paste)
- 목적: "단순 copy-paste 대비 ControlNet이 왜 필요한가?" 입증

출력 형식은 casda_composed/와 동일:
  copypaste_baseline/
  ├── images/          # 1600x256 합성 이미지 (.png)
  ├── masks/           # 1600x256 전체 크기 마스크 (.png)
  └── metadata.json    # 메타데이터 (YOLO bbox 포함)

Usage:
  python scripts/create_copypaste_baseline.py \\
    --roi-dir /content/drive/MyDrive/data/Severstal/roi_patches \\
    --metadata-csv data/processed/roi_patches/roi_metadata.csv \\
    --clean-images-dir /content/drive/MyDrive/data/Severstal/train_images \\
    --train-csv /content/drive/MyDrive/data/Severstal/train.csv \\
    --output-dir /content/drive/MyDrive/data/Severstal/copypaste_baseline

  # 고속 (멀티프로세싱 + 배경 캐시):
  python scripts/create_copypaste_baseline.py \\
    --roi-dir /content/drive/MyDrive/data/Severstal/roi_patches \\
    --metadata-csv data/processed/roi_patches/roi_metadata.csv \\
    --clean-images-dir /content/drive/MyDrive/data/Severstal/train_images \\
    --train-csv /content/drive/MyDrive/data/Severstal/train.csv \\
    --output-dir /content/drive/MyDrive/data/Severstal/copypaste_baseline \\
    --workers -1 \\
    --bg-cache /content/drive/MyDrive/data/Severstal/cache/bg_types.json
"""

import argparse
import ast
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트 및 scripts/ 를 path에 추가
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

# compose_casda_images.py에서 공유 유틸리티 임포트
from compose_casda_images import (
    BackgroundPool,
    find_clean_images,
    parse_bbox_string,
)

from src.preprocessing.poisson_blender import PoissonBlender

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# ROI 직접 합성 (Poisson Blending 없음)
# ============================================================================

def paste_roi_direct(
    roi_img: np.ndarray,
    roi_mask: np.ndarray,
    clean_bg: np.ndarray,
    roi_bbox: Tuple[int, int, int, int],
    jitter_x: int = 0,
) -> Tuple[np.ndarray, bool]:
    """
    ROI를 결함 없는 배경에 직접 붙여넣기 (Poisson Blending 없음).

    mask > 127 픽셀만 배경에 덮어씌운다.

    Args:
        roi_img:   ROI 패치 이미지 (BGR, 임의 크기)
        roi_mask:  ROI 패치 마스크 (단채널, 0/255)
        clean_bg:  1600x256 배경 이미지 (BGR)
        roi_bbox:  (x1, y1, x2, y2) 원본 이미지 좌표계
        jitter_x:  x축 오프셋 (px)

    Returns:
        (composited_image, success)
    """
    result = clean_bg.copy()
    target_h, target_w = clean_bg.shape[:2]
    x1, y1, x2, y2 = roi_bbox

    # jitter 적용
    x1_j = x1 + jitter_x
    x2_j = x2 + jitter_x

    roi_w = x2 - x1
    roi_h = y2 - y1

    if roi_w <= 0 or roi_h <= 0:
        return result, False

    # ROI를 roi_bbox 크기로 리사이즈
    roi_resized = cv2.resize(roi_img, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(roi_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    # target 경계 클리핑
    px1 = max(0, x1_j)
    py1 = max(0, y1)
    px2 = min(target_w, x2_j)
    py2 = min(target_h, y2)

    if px2 <= px1 or py2 <= py1:
        return result, False

    # source 인덱스 계산
    sx1 = px1 - x1_j
    sy1 = py1 - y1
    sx2 = sx1 + (px2 - px1)
    sy2 = sy1 + (py2 - py1)

    if sx2 <= sx1 or sy2 <= sy1:
        return result, False

    mask_region = mask_resized[sy1:sy2, sx1:sx2]
    mask_bool = mask_region > 127

    if not mask_bool.any():
        return result, False

    # 직접 paste
    roi_region = roi_resized[sy1:sy2, sx1:sx2]
    result[py1:py2, px1:px2][mask_bool] = roi_region[mask_bool]

    return result, True


# ============================================================================
# 메인 파이프라인
# ============================================================================

def create_copypaste_dataset(
    roi_dir: Path,
    metadata_csv: Path,
    clean_images_dir: Path,
    train_csv: Path,
    output_dir: Path,
    seed: int = 42,
    jitter_range: int = 100,
    max_backgrounds: int = 5000,
    num_workers: int = 0,
    bg_cache_path: Optional[Path] = None,
    png_compression: int = 1,
    brightness_tolerance: float = 30.0,
    min_defect_area: int = 16,
):
    """
    CopyPaste baseline 데이터셋 생성.

    Args:
        roi_dir:            ROI 패치 디렉토리 (images/ + masks/ 하위 디렉토리 포함)
        metadata_csv:       roi_metadata.csv 경로
        clean_images_dir:   원본 이미지 디렉토리
        train_csv:          train.csv 경로 (결함 이미지 식별용)
        output_dir:         출력 디렉토리
        seed:               랜덤 시드
        jitter_range:       x축 위치 랜덤 오프셋 최대값 ±N px
        max_backgrounds:    배경 유형 분석 최대 이미지 수
        num_workers:        병렬 워커 수 (0=순차)
        bg_cache_path:      배경 유형 캐시 경로
        png_compression:    PNG 압축 레벨
        brightness_tolerance: 배경 밝기 매칭 허용 오차
        min_defect_area:    최소 결함 면적 (YOLO bbox용)
    """
    pipeline_start = time.time()
    rng = random.Random(seed)

    roi_images_dir = roi_dir / "images"
    roi_masks_dir = roi_dir / "masks"

    # ── Step 1: ROI 메타데이터 로딩 ──
    logger.info("=" * 60)
    logger.info("Step 1: ROI 메타데이터 로딩")
    logger.info("=" * 60)

    df = pd.read_csv(metadata_csv)
    logger.info(f"ROI 메타데이터: {len(df)}행")

    # ── Step 2: 배경 이미지 풀 구축 ──
    logger.info("=" * 60)
    logger.info("Step 2: 배경 이미지 풀 구축")
    logger.info("=" * 60)

    clean_names = find_clean_images(train_csv, clean_images_dir)
    logger.info(f"결함 없는 이미지: {len(clean_names)}장")

    if not clean_names:
        logger.error("결함 없는 이미지를 찾을 수 없음")
        sys.exit(1)

    bg_pool = BackgroundPool(
        clean_image_names=clean_names,
        images_dir=clean_images_dir,
        cache_bg_types=True,
        max_analyze=max_backgrounds,
        bg_cache_path=bg_cache_path,
        num_workers=num_workers,
    )

    # ── Step 3: 출력 디렉토리 생성 ──
    out_img_dir = output_dir / "images"
    out_mask_dir = output_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 4: 합성 실행 ──
    logger.info("=" * 60)
    logger.info("Step 4: CopyPaste 합성 실행")
    logger.info("=" * 60)

    blender = PoissonBlender(min_defect_area=min_defect_area)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    all_metadata = []
    stats = {
        'total': len(df),
        'success': 0,
        'fail_roi_missing': 0,
        'fail_bg_missing': 0,
        'fail_paste': 0,
        'class_counts': {},
    }

    records = df.to_dict('records')

    for row in tqdm(records, desc="CopyPaste 합성"):
        image_id = row['image_id']
        class_id = int(row['class_id'])
        region_id = int(row['region_id'])
        sample_name = f"{image_id}_class{class_id}_region{region_id}"

        roi_bbox = parse_bbox_string(row['roi_bbox'])
        defect_subtype = str(row.get('defect_subtype', 'unknown'))
        background_type = str(row.get('background_type', 'unknown'))
        suitability_score = float(row.get('suitability_score', 0.5))

        # ROI 이미지/마스크 경로 구성 (메타데이터 경로의 basename 사용)
        roi_img_basename = Path(str(row['roi_image_path'])).name
        roi_mask_basename = Path(str(row['roi_mask_path'])).name

        roi_img_path = roi_images_dir / roi_img_basename
        roi_mask_path = roi_masks_dir / roi_mask_basename

        roi_img = cv2.imread(str(roi_img_path), cv2.IMREAD_COLOR)
        roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)

        if roi_img is None or roi_mask is None:
            logger.debug(f"ROI 파일 없음: {roi_img_path}")
            stats['fail_roi_missing'] += 1
            continue

        # 배경 밝기 매칭
        x1, y1, x2, y2 = roi_bbox
        roi_mean_brightness = float(np.mean(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)))

        target_brightness = roi_mean_brightness if brightness_tolerance > 0 else None

        bg_name = bg_pool.get_compatible_background(
            defect_subtype=defect_subtype,
            target_brightness=target_brightness,
            brightness_tolerance=brightness_tolerance,
            rng=rng,
        )
        if bg_name is None:
            stats['fail_bg_missing'] += 1
            continue

        bg_path = clean_images_dir / bg_name
        clean_bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if clean_bg is None:
            stats['fail_bg_missing'] += 1
            continue

        # jitter 적용
        jitter_x = rng.randint(-jitter_range, jitter_range) if jitter_range > 0 else 0

        # 직접 paste
        composited, ok = paste_roi_direct(
            roi_img, roi_mask, clean_bg, roi_bbox, jitter_x=jitter_x
        )
        if not ok:
            stats['fail_paste'] += 1
            continue

        # 전체 크기 마스크 생성 (jitter 반영)
        x1_j = x1 + jitter_x
        x2_j = x2 + jitter_x
        roi_bbox_jittered = (x1_j, y1, x2_j, y2)

        roi_mask_resized = cv2.resize(
            roi_mask,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        )
        target_shape = (clean_bg.shape[0], clean_bg.shape[1])
        full_mask = blender.generate_full_mask(roi_mask_resized, roi_bbox_jittered, target_shape)

        bboxes, labels = blender.compute_yolo_bboxes(full_mask, class_id - 1)  # 0-indexed

        if not bboxes:
            stats['fail_paste'] += 1
            continue

        # 출력 저장
        out_name = f"copypaste_{sample_name}.png"
        out_img_path = out_img_dir / out_name
        out_mask_path = out_mask_dir / out_name

        cv2.imwrite(str(out_img_path), composited, png_params)
        cv2.imwrite(str(out_mask_path), full_mask, png_params)

        entry = {
            "image_path": f"images/{out_name}",
            "class_id": class_id - 1,  # 0-indexed
            "suitability_score": round(suitability_score, 6),
            "mask_path": f"masks/{out_name}",
            "bboxes": [[round(v, 6) for v in bbox] for bbox in bboxes],
            "labels": labels,
            "bbox_format": "yolo",
            "image_width": composited.shape[1],
            "image_height": composited.shape[0],
            "source_roi": roi_img_basename,
            "source_background": bg_name,
            "blend_method": "direct_paste",
            "roi_bbox": list(roi_bbox),
            "jitter_x": jitter_x,
            "defect_subtype": defect_subtype,
            "background_type": background_type,
        }

        all_metadata.append(entry)
        stats['success'] += 1
        cid = class_id - 1
        stats['class_counts'][cid] = stats['class_counts'].get(cid, 0) + 1

    # ── 메타데이터 저장 ──
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    pipeline_elapsed = time.time() - pipeline_start

    # ── 리포트 저장 ──
    report = {
        "pipeline": "copypaste_baseline",
        "config": {
            "roi_dir": str(roi_dir),
            "metadata_csv": str(metadata_csv),
            "seed": seed,
            "jitter_range": jitter_range,
            "brightness_tolerance": brightness_tolerance,
        },
        "statistics": {
            "total_rois": stats['total'],
            "success": stats['success'],
            "fail_roi_missing": stats['fail_roi_missing'],
            "fail_bg_missing": stats['fail_bg_missing'],
            "fail_paste": stats['fail_paste'],
            "success_rate": round(stats['success'] / max(stats['total'], 1) * 100, 1),
            "class_distribution": {str(k): v for k, v in sorted(stats['class_counts'].items())},
        },
        "performance": {
            "total_pipeline_sec": round(pipeline_elapsed, 1),
        },
        "output": {
            "output_dir": str(output_dir),
            "total_images": len(all_metadata),
            "total_bboxes": sum(len(m.get("bboxes", [])) for m in all_metadata),
        },
    }

    report_path = output_dir / "composition_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CopyPaste Baseline 생성 완료")
    logger.info("=" * 60)
    logger.info(f"  입력 ROI: {stats['total']}개")
    logger.info(f"  성공: {stats['success']}장 ({stats['success']/max(stats['total'],1)*100:.1f}%)")
    logger.info(f"  실패 — ROI 파일 없음: {stats['fail_roi_missing']}")
    logger.info(f"  실패 — 배경 없음: {stats['fail_bg_missing']}")
    logger.info(f"  실패 — paste 실패: {stats['fail_paste']}")
    logger.info(f"  클래스 분포 (0-indexed): {dict(sorted(stats['class_counts'].items()))}")
    logger.info(f"  총 bbox 수: {report['output']['total_bboxes']}")
    logger.info(f"  출력 디렉토리: {output_dir}")
    logger.info(f"  소요 시간: {pipeline_elapsed:.1f}초")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CopyPaste Baseline Generator — ROI 직접 붙여넣기 (Poisson Blending 없음)"
    )
    parser.add_argument(
        "--roi-dir", type=str, required=True,
        help="ROI 패치 디렉토리 경로 (images/ + masks/ 하위 디렉토리 포함)",
    )
    parser.add_argument(
        "--metadata-csv", type=str, required=True,
        help="roi_metadata.csv 경로",
    )
    parser.add_argument(
        "--clean-images-dir", type=str, required=True,
        help="원본 이미지 디렉토리 (train_images/, 1600x256)",
    )
    parser.add_argument(
        "--train-csv", type=str, required=True,
        help="train.csv 경로 (결함 없는 이미지 식별용)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    parser.add_argument(
        "--jitter-range", type=int, default=100,
        help="x축 위치 랜덤 오프셋 최대값 ±N px (0=비활성, 기본: 100)",
    )
    parser.add_argument(
        "--max-backgrounds", type=int, default=5000,
        help="배경 유형 분석할 최대 이미지 수 (기본: 5000)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="배경 분석 병렬 워커 수 (0=순차, -1=CPU 자동, 기본: 0)",
    )
    parser.add_argument(
        "--bg-cache", type=str, default=None,
        help="배경 유형 캐시 JSON 경로 (지정 시 재실행에서 분석 건너뜀)",
    )
    parser.add_argument(
        "--png-compression", type=int, default=1,
        help="PNG 압축 레벨 0-9 (낮을수록 빠름, 기본: 1)",
    )
    parser.add_argument(
        "--brightness-tolerance", type=float, default=30.0,
        help="배경 밝기 매칭 허용 오차 ±N (0=비활성, 기본: 30.0)",
    )
    parser.add_argument(
        "--min-defect-area", type=int, default=16,
        help="YOLO bbox 추출 최소 결함 면적 (기본: 16)",
    )

    args = parser.parse_args()

    num_workers = args.workers
    if num_workers < 0:
        cpu_count = os.cpu_count() or 4
        num_workers = max(1, cpu_count - 1)
        logger.info(f"워커 수 자동 설정: {num_workers} (CPU: {cpu_count})")

    create_copypaste_dataset(
        roi_dir=Path(args.roi_dir),
        metadata_csv=Path(args.metadata_csv),
        clean_images_dir=Path(args.clean_images_dir),
        train_csv=Path(args.train_csv),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        jitter_range=args.jitter_range,
        max_backgrounds=args.max_backgrounds,
        num_workers=num_workers,
        bg_cache_path=Path(args.bg_cache) if args.bg_cache else None,
        png_compression=args.png_compression,
        brightness_tolerance=args.brightness_tolerance,
        min_defect_area=args.min_defect_area,
    )


if __name__ == "__main__":
    main()
