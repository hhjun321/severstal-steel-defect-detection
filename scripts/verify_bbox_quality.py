#!/usr/bin/env python3
"""
CASDA Bbox Label Quality Verification

v5.3 분석에서 Detection 모델 성능이 CASDA 증강 시 급락한 원인 중 하나로
mask→contour→boundingRect 변환 과정의 bbox 품질 문제를 의심.

이 스크립트는 다음을 검증:
  1. 합성 이미지의 bbox 크기/비율 분포 vs 원본 bbox 분포
  2. 합성 bbox의 이상치 비율 (너무 작거나/크거나/비정상 비율)
  3. 전체 이미지를 bbox로 잡는 fallback 비율
  4. 클래스별 bbox 분포 차이

사용법:
  python scripts/verify_bbox_quality.py \
      --csv train.csv \
      --image-dir train_images \
      --casda-dir data/augmented/casda_full \
      --output-dir outputs/bbox_analysis
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_real_bboxes(csv_path: str, image_dir: str) -> Dict[str, List[Dict]]:
    """원본 데이터셋에서 RLE → bbox 변환."""
    import pandas as pd
    from src.utils.rle_utils import rle_decode

    df = pd.read_csv(csv_path)
    if 'ImageId_ClassId' in df.columns:
        df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.rsplit('_', n=1, expand=True)
        df['ClassId'] = df['ClassId'].astype(int)

    defect_df = df[df['EncodedPixels'].notna()]

    bboxes_by_class = defaultdict(list)
    for _, row in defect_df.iterrows():
        cls_id = int(row['ClassId']) - 1
        rle = row['EncodedPixels']
        mask = rle_decode(rle, (256, 1600))
        if mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 16:
                continue
            bboxes_by_class[cls_id].append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': w * h,
                'aspect_ratio': w / max(h, 1),
                'norm_w': w / 1600,
                'norm_h': h / 256,
            })

    return bboxes_by_class


def load_casda_bboxes(casda_dir: str) -> Tuple[Dict[str, List[Dict]], dict]:
    """CASDA 합성 데이터에서 bbox 로드."""
    casda_path = Path(casda_dir)
    meta_path = casda_path / "metadata.json"

    if not meta_path.exists():
        logging.error(f"metadata.json not found in {casda_dir}")
        return {}, {}

    with open(meta_path) as f:
        all_samples = json.load(f)

    bboxes_by_class = defaultdict(list)
    stats = {
        'total_samples': len(all_samples),
        'has_bboxes': 0,
        'has_mask_only': 0,
        'fallback_full_image': 0,
        'no_bbox_no_mask': 0,
        'bbox_format_yolo': 0,
        'bbox_format_xyxy': 0,
    }

    for sample in all_samples:
        cls_id = sample.get('class_id', 0)
        bboxes = sample.get('bboxes', [])
        labels = sample.get('labels', [])
        bbox_format = sample.get('bbox_format', 'xyxy')

        if bboxes and labels:
            stats['has_bboxes'] += 1
            if bbox_format == 'yolo':
                stats['bbox_format_yolo'] += 1
                for bbox, lbl in zip(bboxes, labels):
                    cx, cy, bw, bh = bbox
                    # YOLO normalized → pixel (assume 512x512 ControlNet output)
                    # 분포 비교를 위해 정규화 좌표 그대로 사용
                    bboxes_by_class[lbl].append({
                        'cx': cx, 'cy': cy,
                        'norm_w': bw, 'norm_h': bh,
                        'area_norm': bw * bh,
                        'aspect_ratio': bw / max(bh, 0.001),
                        'is_full_image': (bw > 0.95 and bh > 0.95),
                    })
            else:
                stats['bbox_format_xyxy'] += 1
                # xyxy 형식 — 이미지 읽어서 크기 필요
                img_path = sample.get('image_path', '')
                if not os.path.isabs(img_path):
                    img_path = str(casda_path / img_path)
                img = cv2.imread(img_path)
                if img is not None:
                    ih, iw = img.shape[:2]
                    for bbox, lbl in zip(bboxes, labels):
                        x1, y1, x2, y2 = bbox
                        bw = (x2 - x1) / iw
                        bh = (y2 - y1) / ih
                        bboxes_by_class[lbl].append({
                            'norm_w': bw, 'norm_h': bh,
                            'area_norm': bw * bh,
                            'aspect_ratio': bw / max(bh, 0.001),
                            'is_full_image': (bw > 0.95 and bh > 0.95),
                        })
        elif 'mask_path' in sample:
            stats['has_mask_only'] += 1
            # mask → contour → bbox
            mask_path = sample['mask_path']
            if not os.path.isabs(mask_path):
                mask_path = str(casda_path / mask_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mh, mw = mask.shape[:2]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        bboxes_by_class[cls_id].append({
                            'norm_w': w / mw, 'norm_h': h / mh,
                            'area_norm': (w * h) / (mw * mh),
                            'aspect_ratio': w / max(h, 1),
                            'is_full_image': (w / mw > 0.95 and h / mh > 0.95),
                        })
                else:
                    stats['fallback_full_image'] += 1
            else:
                stats['fallback_full_image'] += 1
        else:
            stats['no_bbox_no_mask'] += 1
            stats['fallback_full_image'] += 1

    return bboxes_by_class, stats


def analyze_distribution(real_bboxes, casda_bboxes, output_dir):
    """bbox 분포 비교 분석."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 70)
    report.append("CASDA Bbox Label Quality Report")
    report.append("=" * 70)

    class_names = {0: 'Class1', 1: 'Class2', 2: 'Class3', 3: 'Class4'}

    for cls_id in range(4):
        cls_name = class_names[cls_id]
        real = real_bboxes.get(cls_id, [])
        casda = casda_bboxes.get(cls_id, [])

        report.append(f"\n--- {cls_name} ---")
        report.append(f"  Real bboxes: {len(real)}")
        report.append(f"  CASDA bboxes: {len(casda)}")

        if real:
            real_widths = [b['norm_w'] for b in real]
            real_heights = [b['norm_h'] for b in real]
            real_aspects = [b['aspect_ratio'] for b in real]
            report.append(f"  Real width  (norm): mean={np.mean(real_widths):.4f}, "
                          f"std={np.std(real_widths):.4f}, "
                          f"median={np.median(real_widths):.4f}")
            report.append(f"  Real height (norm): mean={np.mean(real_heights):.4f}, "
                          f"std={np.std(real_heights):.4f}, "
                          f"median={np.median(real_heights):.4f}")
            report.append(f"  Real aspect ratio:  mean={np.mean(real_aspects):.4f}, "
                          f"std={np.std(real_aspects):.4f}")

        if casda:
            casda_widths = [b['norm_w'] for b in casda]
            casda_heights = [b['norm_h'] for b in casda]
            casda_aspects = [b['aspect_ratio'] for b in casda]
            full_image_count = sum(1 for b in casda if b.get('is_full_image', False))

            report.append(f"  CASDA width  (norm): mean={np.mean(casda_widths):.4f}, "
                          f"std={np.std(casda_widths):.4f}, "
                          f"median={np.median(casda_widths):.4f}")
            report.append(f"  CASDA height (norm): mean={np.mean(casda_heights):.4f}, "
                          f"std={np.std(casda_heights):.4f}, "
                          f"median={np.median(casda_heights):.4f}")
            report.append(f"  CASDA aspect ratio:  mean={np.mean(casda_aspects):.4f}, "
                          f"std={np.std(casda_aspects):.4f}")
            report.append(f"  Full-image fallback: {full_image_count} "
                          f"({full_image_count / len(casda) * 100:.1f}%)")

            # Outlier detection: bboxes covering >80% of image
            large_count = sum(1 for b in casda if b.get('area_norm', 0) > 0.8)
            tiny_count = sum(1 for b in casda if b.get('area_norm', 0) < 0.001)
            report.append(f"  CASDA >80% area: {large_count} ({large_count / len(casda) * 100:.1f}%)")
            report.append(f"  CASDA <0.1% area: {tiny_count} ({tiny_count / len(casda) * 100:.1f}%)")

        # Distribution difference
        if real and casda:
            real_w_mean = np.mean([b['norm_w'] for b in real])
            casda_w_mean = np.mean([b['norm_w'] for b in casda])
            real_h_mean = np.mean([b['norm_h'] for b in real])
            casda_h_mean = np.mean([b['norm_h'] for b in casda])

            report.append(f"  Width diff:  {casda_w_mean - real_w_mean:+.4f} "
                          f"({(casda_w_mean - real_w_mean) / real_w_mean * 100:+.1f}%)")
            report.append(f"  Height diff: {casda_h_mean - real_h_mean:+.4f} "
                          f"({(casda_h_mean - real_h_mean) / real_h_mean * 100:+.1f}%)")

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_path = output_path / "bbox_quality_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Save raw data as JSON
    summary = {
        'real': {str(k): len(v) for k, v in real_bboxes.items()},
        'casda': {str(k): len(v) for k, v in casda_bboxes.items()},
    }

    for cls_id in range(4):
        real = real_bboxes.get(cls_id, [])
        casda = casda_bboxes.get(cls_id, [])
        cls_name = class_names[cls_id]

        summary[cls_name] = {
            'real_count': len(real),
            'casda_count': len(casda),
        }
        if real:
            summary[cls_name]['real_width_mean'] = float(np.mean([b['norm_w'] for b in real]))
            summary[cls_name]['real_height_mean'] = float(np.mean([b['norm_h'] for b in real]))
        if casda:
            summary[cls_name]['casda_width_mean'] = float(np.mean([b['norm_w'] for b in casda]))
            summary[cls_name]['casda_height_mean'] = float(np.mean([b['norm_h'] for b in casda]))
            summary[cls_name]['full_image_pct'] = float(
                sum(1 for b in casda if b.get('is_full_image', False)) / len(casda) * 100
            )

    with open(output_path / "bbox_quality_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Report saved to: {report_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="CASDA Bbox Quality Verification")
    parser.add_argument('--csv', type=str, required=True,
                        help='Original annotation CSV (train.csv)')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Original image directory')
    parser.add_argument('--casda-dir', type=str, required=True,
                        help='CASDA data directory (with metadata.json)')
    parser.add_argument('--output-dir', type=str, default='outputs/bbox_analysis',
                        help='Output directory for analysis results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    logging.info("Loading real bboxes from original dataset...")
    real_bboxes = load_real_bboxes(args.csv, args.image_dir)
    logging.info(f"  Total real bboxes: {sum(len(v) for v in real_bboxes.values())}")

    logging.info("Loading CASDA bboxes...")
    casda_bboxes, casda_stats = load_casda_bboxes(args.casda_dir)
    logging.info(f"  Total CASDA samples: {casda_stats.get('total_samples', 0)}")
    logging.info(f"  Has bboxes: {casda_stats.get('has_bboxes', 0)}")
    logging.info(f"  Has mask only: {casda_stats.get('has_mask_only', 0)}")
    logging.info(f"  Fallback (full image): {casda_stats.get('fallback_full_image', 0)}")
    logging.info(f"  YOLO format: {casda_stats.get('bbox_format_yolo', 0)}")
    logging.info(f"  XYXY format: {casda_stats.get('bbox_format_xyxy', 0)}")

    logging.info("\nAnalyzing distribution differences...")
    summary = analyze_distribution(real_bboxes, casda_bboxes, args.output_dir)

    # Save stats
    stats_path = Path(args.output_dir) / "casda_metadata_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(casda_stats, f, indent=2)

    logging.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
