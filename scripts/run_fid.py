#!/usr/bin/env python3
"""
CASDA FID Score Evaluation — 독립 실행 스크립트

run_benchmark.py에서 분리된 FID 전용 스크립트.
FID-ROI (ROI vs ROI) + FID-Composed (1600x256 vs 1600x256) 평가를 수행한다.

FID-ROI:
  - real ROI crops (roi_metadata.csv 기반) vs ControlNet 생성 ROI (generated/)
  - 세분화: class_id / defect_subtype / background_type / class×subtype 교차

FID-Composed:
  - real 전체 이미지 (train_images/) vs pruning된 합성 이미지 (casda_composed/ top-k)
  - 기본 동작: metadata.json의 suitability_score 기반 pruning 적용
  - --no-fid-pruning 옵션으로 composed 전체 대상 계산 가능

Config:
  benchmark_experiment.yaml을 run_benchmark.py와 동일하게 공유.
  evaluation.fid 섹션의 설정을 사용.

Usage:
  # 기본 사용 (YAML config에서 경로 읽기)
  python scripts/run_fid.py --config configs/benchmark_experiment.yaml

  # Colab: 명시적 경로 지정
  python scripts/run_fid.py \\
      --config /content/severstal-steel-defect-detection/configs/benchmark_experiment.yaml \\
      --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --casda-dir /content/drive/MyDrive/data/Severstal/augmented_dataset_v5.5 \\
      --casda-roi-dir /content/drive/MyDrive/data/Severstal/augmented_images_v5.5/generated \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/benchmark_results/fid_v5.5

  # FID-ROI만 실행
  python scripts/run_fid.py --config configs/benchmark_experiment.yaml --fid-mode roi

  # FID-Composed만 실행
  python scripts/run_fid.py --config configs/benchmark_experiment.yaml --fid-mode composed
"""

import os
import sys
import argparse
import logging
import random
import re
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── 프로젝트 루트 설정 ──
# NOTE: scripts/__init__.py가 없으므로 `from scripts.utils import ...` 전에
# 프로젝트 루트를 sys.path에 먼저 추가해야 한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config, remove_empty_classes, sample_images  # noqa: E402
from src.training.metrics import FIDCalculator  # noqa: E402


# ============================================================================
# FID 헬퍼 함수들 — run_benchmark.py L773-1132에서 추출
# ============================================================================

def _collect_images_from_dir(directory: Path) -> list:
    """디렉토리에서 이미지 파일 경로를 수집 (중복 제거)."""
    imgs: set = set()
    for subdir in [directory, directory / "images"]:
        if subdir.exists():
            for ext in ("*.png", "*.jpg"):
                for p in subdir.glob(ext):
                    imgs.add(str(p.resolve()))
    return sorted(imgs)


def _group_gen_images_by_class(
    metadata_path: Path,
    base_dir: Path,
) -> Optional[dict]:
    """metadata.json에서 합성 이미지를 class_id(0-based)별로 그룹핑.

    Returns:
        {cls_id: [abs_path, ...]} 또는 metadata가 없으면 None
    """
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    by_class: dict = {}
    for entry in metadata:
        cls_id = entry.get('class_id')
        if cls_id is None:
            continue
        img_rel = entry.get('image_path', '')
        img_abs = str(base_dir / img_rel)
        if os.path.exists(img_abs):
            by_class.setdefault(cls_id, []).append(img_abs)
    return by_class if by_class else None


def _load_pruned_image_paths(
    metadata_path: Path,
    base_dir: Path,
    top_k: int = 2000,
    suitability_threshold: float = 0.0,
    stratified: bool = True,
) -> list:
    """metadata.json에서 pruning된 이미지 경로 리스트를 반환한다.

    run_benchmark.py의 pruning 로직과 동일:
      1. suitability_threshold 이상 필터링
      2. suitability_score 내림차순 정렬
      3. stratified top-k 또는 global top-k 선택

    Args:
        metadata_path: composed 디렉토리의 metadata.json 경로
        base_dir: 이미지 상대 경로의 기준 디렉토리
        top_k: 선택할 최대 이미지 수
        suitability_threshold: 최소 suitability 점수
        stratified: True이면 클래스 비율 유지 top-k

    Returns:
        pruning된 이미지 절대 경로 리스트
    """
    if not metadata_path.exists():
        logging.warning(f"  metadata.json not found: {metadata_path} — pruning 불가")
        return []

    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # threshold 필터
    filtered = [
        s for s in all_samples
        if s.get('suitability_score', 0.0) >= suitability_threshold
    ]
    logging.info(f"  Pruning: {len(all_samples)} total → {len(filtered)} above threshold {suitability_threshold}")

    # 정렬
    filtered.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)

    # top-k 선택
    if stratified and top_k < len(filtered):
        # 클래스별 비율 유지 top-k
        by_class: dict = {}
        for s in filtered:
            cls_id = s.get('class_id', 0)
            by_class.setdefault(cls_id, []).append(s)

        total = len(filtered)
        selected = []
        remainders: list = []
        for cls_id in sorted(by_class.keys()):
            cls_samples = by_class[cls_id]
            quota = (len(cls_samples) / total) * top_k
            int_quota = int(quota)
            remainder = quota - int_quota
            selected.extend(cls_samples[:int_quota])
            remainders.append((remainder, cls_id, cls_samples, int_quota))

        # 나머지 할당 (소수점 큰 순서로)
        remainders.sort(key=lambda x: x[0], reverse=True)
        deficit = top_k - len(selected)
        for i in range(min(deficit, len(remainders))):
            _, cls_id, cls_samples, int_quota = remainders[i]
            if int_quota < len(cls_samples):
                selected.append(cls_samples[int_quota])

        filtered = selected[:top_k]
        logging.info(f"  Stratified top-k: {len(filtered)} selected")
    else:
        filtered = filtered[:top_k]
        logging.info(f"  Global top-k: {len(filtered)} selected")

    # 경로 해석
    paths = []
    for s in filtered:
        img_rel = s.get('image_path', '')
        img_abs = str(base_dir / img_rel) if not os.path.isabs(img_rel) else img_rel
        if os.path.exists(img_abs):
            paths.append(img_abs)

    logging.info(f"  Pruned images (existing): {len(paths)}")
    return paths


def _group_pruned_by_class(
    metadata_path: Path,
    base_dir: Path,
    top_k: int = 2000,
    suitability_threshold: float = 0.0,
    stratified: bool = True,
) -> Optional[dict]:
    """pruning된 이미지를 class_id(0-based)별로 그룹핑하여 반환.

    Returns:
        {cls_id: [abs_path, ...]} 또는 None
    """
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # threshold + sort + top-k (동일 로직)
    filtered = [
        s for s in all_samples
        if s.get('suitability_score', 0.0) >= suitability_threshold
    ]
    filtered.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)

    if stratified and top_k < len(filtered):
        by_cls: dict = {}
        for s in filtered:
            by_cls.setdefault(s.get('class_id', 0), []).append(s)
        total = len(filtered)
        selected = []
        remainders: list = []
        for cls_id in sorted(by_cls.keys()):
            cls_samples = by_cls[cls_id]
            quota = (len(cls_samples) / total) * top_k
            int_quota = int(quota)
            remainder = quota - int_quota
            selected.extend(cls_samples[:int_quota])
            remainders.append((remainder, cls_id, cls_samples, int_quota))
        remainders.sort(key=lambda x: x[0], reverse=True)
        deficit = top_k - len(selected)
        for i in range(min(deficit, len(remainders))):
            _, cls_id, cls_samples, int_quota = remainders[i]
            if int_quota < len(cls_samples):
                selected.append(cls_samples[int_quota])
        filtered = selected[:top_k]
    else:
        filtered = filtered[:top_k]

    by_class: dict = {}
    for s in filtered:
        cls_id = s.get('class_id', 0)
        img_rel = s.get('image_path', '')
        img_abs = str(base_dir / img_rel) if not os.path.isabs(img_rel) else img_rel
        if os.path.exists(img_abs):
            by_class.setdefault(cls_id, []).append(img_abs)
    return by_class if by_class else None


# ── FID-ROI 세분화 헬퍼 함수들 (v5.5) ──────────────────────────────────

_CLASS_PATTERN = re.compile(r"_class(\d+)_")
_SAMPLE_NAME_PATTERN = re.compile(
    r"^(.+?)_class(\d+)_region(\d+)(?:_gen\d+)?\.png$"
)


def _group_gen_images_by_filename(image_paths: list) -> Optional[dict]:
    """파일명에서 class_id를 파싱하여 클래스별 그룹핑 (metadata.json 없을 때 fallback).

    파일명 형식: {image_id}_class{N}_region{M}_gen{J}.png (N은 1-based)
    Returns: {cls_id_0based: [abs_path, ...]} 또는 파싱 불가 시 None
    """
    by_class: dict = {}
    for path in image_paths:
        fname = os.path.basename(path)
        match = _CLASS_PATTERN.search(fname)
        if match:
            cls_id = int(match.group(1)) - 1  # 1-based → 0-based
            by_class.setdefault(cls_id, []).append(path)
    return by_class if by_class else None


def _parse_sample_name(filename: str) -> Optional[str]:
    """생성 파일명 → sample_name 변환.

    예: 0c6720401.jpg_class2_region0_gen0.png → 0c6720401.jpg_class2_region0
    Returns: sample_name 또는 파싱 불가 시 None
    """
    match = _SAMPLE_NAME_PATTERN.match(filename)
    if match:
        image_id, cls_num, region_num = match.group(1), match.group(2), match.group(3)
        return f"{image_id}_class{cls_num}_region{region_num}"
    return None


def _build_roi_lookup_from_metadata(roi_metadata_path: Path) -> dict:
    """roi_metadata.csv에서 sample_name → {class_id, defect_subtype, background_type} 룩업 구축.

    sample_name 형식: {image_id}_class{N}_region{M}  (N은 1-based, csv에서 조합)

    Returns:
        {sample_name: {'class_id': int(0-based), 'defect_subtype': str, 'background_type': str}}
    """
    roi_df = pd.read_csv(roi_metadata_path)
    lookup: dict = {}
    for _, row in roi_df.iterrows():
        image_id = str(row['image_id'])
        class_id_1based = int(row['class_id'])
        region_id = int(row['region_id'])
        sample_name = f"{image_id}_class{class_id_1based}_region{region_id}"
        lookup[sample_name] = {
            'class_id': class_id_1based - 1,  # 0-based
            'defect_subtype': str(row.get('defect_subtype', 'unknown')),
            'background_type': str(row.get('background_type', 'unknown')),
        }
    return lookup


def _group_images_by_dimension(
    image_paths: list,
    lookup: dict,
    dimension: str,
    source: str = 'synthetic',
) -> Optional[dict]:
    """이미지를 지정 차원(defect_subtype, background_type 등)으로 그룹핑.

    Args:
        image_paths: 이미지 절대 경로 리스트
        lookup: _build_roi_lookup_from_metadata()의 반환값
        dimension: 그룹핑 차원 키 ('defect_subtype', 'background_type', 'class_id')
        source: 'synthetic' (파일명에서 sample_name 파싱) 또는 'real' (lookup key가 경로에 포함)

    Returns:
        {dimension_value: [abs_path, ...]} 또는 매칭 불가 시 None
    """
    by_dim: dict = {}
    unmatched = 0
    for path in image_paths:
        fname = os.path.basename(path)
        if source == 'synthetic':
            sample_name = _parse_sample_name(fname)
        else:
            # real ROI: 경로에서 sample_name 유추는 어려움 → lookup에서 직접 매칭
            sample_name = None
            # real ROI 파일명 형식: {image_id}_class{N}_region{M}_roi.png 등
            # 그러나 실제로 real ROI는 roi_metadata.csv의 roi_image_path 그대로이므로
            # lookup 매칭 대신 별도 처리 필요 → _group_real_roi_by_dimension() 사용
            pass

        if sample_name and sample_name in lookup:
            dim_val = lookup[sample_name].get(dimension, 'unknown')
            by_dim.setdefault(dim_val, []).append(path)
        else:
            unmatched += 1

    if unmatched > 0:
        logging.debug(f"  _group_images_by_dimension({dimension}, {source}): "
                      f"{unmatched}/{len(image_paths)} unmatched")
    return by_dim if by_dim else None


def _group_real_roi_by_dimension(
    roi_metadata_path: Path,
    dimension: str,
    max_images: int,
    rng: random.Random,
) -> Optional[dict]:
    """roi_metadata.csv에서 real ROI를 지정 차원으로 그룹핑.

    Args:
        roi_metadata_path: roi_metadata.csv 경로
        dimension: 'defect_subtype', 'background_type', 또는 'class_subtype' (교차)
        max_images: 그룹당 최대 이미지 수
        rng: 샘플링용 Random 인스턴스

    Returns:
        {dimension_value: [abs_path, ...]}
    """
    roi_df = pd.read_csv(roi_metadata_path)
    by_dim: dict = {}

    for _, row in roi_df.iterrows():
        roi_path = str(row['roi_image_path'])
        if not os.path.exists(roi_path):
            continue

        if dimension == 'class_subtype':
            # 교차: "Class{N}_{subtype}"
            cls_1based = int(row['class_id'])
            subtype = str(row.get('defect_subtype', 'unknown'))
            dim_val = f"Class{cls_1based}_{subtype}"
        else:
            dim_val = str(row.get(dimension, 'unknown'))
        by_dim.setdefault(dim_val, []).append(roi_path)

    # 그룹당 max_images 샘플링
    for dim_val in by_dim:
        if len(by_dim[dim_val]) > max_images:
            by_dim[dim_val] = rng.sample(by_dim[dim_val], max_images)

    return by_dim if by_dim else None


def _group_synthetic_by_cross_dimension(
    image_paths: list,
    lookup: dict,
) -> Optional[dict]:
    """합성 이미지를 class_id × defect_subtype 교차 차원으로 그룹핑.

    Returns:
        {"Class{N}_{subtype}": [abs_path, ...]} 또는 매칭 불가 시 None
    """
    by_cross: dict = {}
    for path in image_paths:
        fname = os.path.basename(path)
        sample_name = _parse_sample_name(fname)
        if sample_name and sample_name in lookup:
            info = lookup[sample_name]
            cls_1based = info['class_id'] + 1
            subtype = info['defect_subtype']
            key = f"Class{cls_1based}_{subtype}"
            by_cross.setdefault(key, []).append(path)
    return by_cross if by_cross else None


def _compute_granular_fid(
    fid_calc,
    real_by_dim: dict,
    gen_by_dim: dict,
    dimension_name: str,
    min_samples: int,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[Path],
) -> dict:
    """차원별 FID 계산 (min_samples 필터링 포함).

    Args:
        fid_calc: FIDCalculator 인스턴스 (feature 캐시 유지)
        real_by_dim: {dim_val: [real_paths]}
        gen_by_dim: {dim_val: [gen_paths]}
        dimension_name: 결과 키 접두사 (예: 'defect_subtype', 'background_type')
        min_samples: 최소 샘플 수 — real/gen 중 하나라도 미달 시 스킵
        batch_size, num_workers, cache_dir: FID 계산 파라미터

    Returns:
        {"{dim_val}_FID_roi": float, ...}
    """
    results: dict = {}
    all_dim_vals = sorted(set(real_by_dim.keys()) | set(gen_by_dim.keys()))

    for dim_val in all_dim_vals:
        real_paths = real_by_dim.get(dim_val, [])
        gen_paths = gen_by_dim.get(dim_val, [])

        if len(real_paths) < min_samples or len(gen_paths) < min_samples:
            logging.info(
                f"    {dimension_name}/{dim_val}: real={len(real_paths)}, "
                f"syn={len(gen_paths)} — 최소 {min_samples}장 미달, 스킵"
            )
            results[f"{dim_val}_FID_roi"] = None  # N/A 표시
            continue

        logging.info(
            f"    {dimension_name}/{dim_val}: real={len(real_paths)}, "
            f"syn={len(gen_paths)}"
        )

        # feature 추출 (이미 캐시된 것은 재사용)
        all_paths = sorted(set(real_paths) | set(gen_paths))
        fid_calc._extract_features(all_paths, batch_size, num_workers, cache_dir)

        real_feats = fid_calc._gather_cached_features(real_paths)
        gen_feats = fid_calc._gather_cached_features(gen_paths)

        if len(real_feats) < 2 or len(gen_feats) < 2:
            results[f"{dim_val}_FID_roi"] = float('inf')
        else:
            mu1, s1 = fid_calc._compute_statistics(real_feats)
            mu2, s2 = fid_calc._compute_statistics(gen_feats)
            fid_val = fid_calc._calculate_fid(mu1, s1, mu2, s2)
            results[f"{dim_val}_FID_roi"] = fid_val
            logging.info(f"      → FID = {fid_val:.2f}")

    return results


def _build_real_by_class_from_csv(
    annotation_path: Path,
    image_dir: Path,
    class_ids: list,
    max_images: int,
    rng: random.Random,
) -> dict:
    """train.csv에서 클래스별 real 이미지 경로를 구성한다.

    Args:
        annotation_path: train.csv 경로
        image_dir: train_images/ 디렉토리
        class_ids: 대상 cls_id 리스트 (0-based)
        max_images: 클래스당 최대 이미지 수
        rng: 샘플링용 Random 인스턴스

    Returns:
        {cls_id: [abs_path, ...]}
    """
    ann_df = pd.read_csv(annotation_path)
    real_by_class: dict = {}
    for cls_id in class_ids:
        cls_label = cls_id + 1  # 0-based → 1-based
        cls_rows = ann_df[
            (ann_df['ClassId'] == cls_label) &
            (ann_df['EncodedPixels'].notna())
        ]
        cls_image_ids = cls_rows['ImageId'].unique().tolist()
        cls_real_paths = [
            str(image_dir / img_id) for img_id in cls_image_ids
            if (image_dir / img_id).exists()
        ]
        if len(cls_real_paths) > max_images:
            cls_real_paths = rng.sample(cls_real_paths, max_images)
        real_by_class[cls_id] = cls_real_paths
    return real_by_class


def _build_real_roi_by_class_from_csv(
    roi_metadata_path: Path,
    class_ids: list,
    max_images: int,
    rng: random.Random,
) -> dict:
    """roi_metadata.csv에서 클래스별 real ROI 이미지 경로를 구성한다.

    roi_metadata.csv 컬럼: class_id(1-based), roi_image_path(절대경로)

    Args:
        roi_metadata_path: roi_metadata.csv 경로
        class_ids: 대상 cls_id 리스트 (0-based)
        max_images: 클래스당 최대 이미지 수
        rng: 샘플링용 Random 인스턴스

    Returns:
        {cls_id: [abs_path, ...]}
    """
    roi_df = pd.read_csv(roi_metadata_path)
    real_by_class: dict = {}
    for cls_id in class_ids:
        cls_label = cls_id + 1  # 0-based → 1-based
        cls_rows = roi_df[roi_df['class_id'] == cls_label]
        cls_paths = [
            str(p) for p in cls_rows['roi_image_path'].tolist()
            if os.path.exists(str(p))
        ]
        if len(cls_paths) > max_images:
            cls_paths = rng.sample(cls_paths, max_images)
        real_by_class[cls_id] = cls_paths
    return real_by_class


# ============================================================================
# FID 평가 메인 함수 — run_benchmark.py L1135-1513에서 추출
# ============================================================================

def run_fid_evaluation(config: dict, experiment_dir: Path, device: str = 'cuda') -> dict:
    """FID 평가: fid_roi (ROI vs ROI) + fid_composed (전체 이미지 vs 전체 이미지).

    v5.5+ 변경:
      - fid_roi: real ROI 크롭(roi_patches) vs ControlNet 생성 ROI
        → roi_dir (generated/) 직접 참조 지원 (casda_full fallback)
        → ControlNet 생성 품질 직접 비교 (동일 스케일)
      - fid_composed: real 전체 이미지(train_images) vs Poisson Blending 합성(casda_composed)
        → 최종 합성 이미지 품질 평가 (동일 해상도 1600x256)
      - fid_mode: "both"(기본), "roi", "composed" 선택 가능
      - 하위 호환: fid_overall = fid_composed_overall

    FID-ROI 세분화 (roi_granularity):
      - by_defect_subtype: defect_subtype별 FID-ROI (5그룹)
      - by_background_type: background_type별 FID-ROI (5그룹)
      - cross_class_subtype: class_id × defect_subtype 교차 FID (유효 조합만)
      - min_samples: 그룹당 최소 샘플 수 (미달 시 스킵)
      - roi_metadata.csv의 메타데이터를 활용하여 generated/ 파일명과 조인

    성능 최적화:
      - compute_fid_with_preextract()로 overall + per-class를 한 번의 추출로 수행
      - DataLoader num_workers로 이미지 I/O 병렬화
      - 디스크 캐시로 재실행 시 즉시 로드
      - 세분화 FID는 기존 feature 캐시 재활용 (중복 추출 없음)

    Args:
        config: benchmark_experiment.yaml에서 로드한 dict
        experiment_dir: 결과 저장 디렉토리 (fid_results.json, fid_cache/)
        device: 'cuda' 또는 'cpu'

    Returns:
        FID 결과 dict (fid_roi_overall, fid_composed_overall, 클래스별, 세분화별 등)
    """
    logging.info(f"\n{'#'*70}")
    logging.info(f"# FID Score Evaluation")
    logging.info(f"{'#'*70}")

    ds_config = config['dataset']
    casda_config = ds_config.get('casda', {})
    fid_config = config.get('evaluation', {}).get('fid', {})

    batch_size = fid_config.get('batch_size', 64)
    max_images = fid_config.get('max_images', 1000)
    per_class = fid_config.get('per_class', False)
    num_workers = fid_config.get('num_workers', 4)
    fid_mode = fid_config.get('fid_mode', 'both')  # "roi", "composed", "both"
    fid_seed = 42  # 재현성을 위한 고정 시드

    # 디스크 캐시 디렉토리 (experiment_dir 하위)
    cache_dir = experiment_dir / "fid_cache"

    # 공통 경로 해석
    raw_image_dir = ds_config['image_dir']
    image_dir = Path(raw_image_dir) if os.path.isabs(raw_image_dir) else PROJECT_ROOT / raw_image_dir

    annotation_csv = ds_config.get('annotation_csv', 'train.csv')
    annotation_path = (
        Path(annotation_csv) if os.path.isabs(annotation_csv)
        else PROJECT_ROOT / annotation_csv
    )

    results: dict = {}
    fid_calc = FIDCalculator(device=device)

    # ==================================================================
    # fid_roi: ROI vs ROI (real ROI crops vs ControlNet 512x512 outputs)
    #   v5.5+: roi_dir (generated/) 직접 참조 지원
    #   세분화: class_id / defect_subtype / background_type / 교차
    # ==================================================================
    if fid_mode in ('roi', 'both'):
        logging.info(f"\n{'─'*50}")
        logging.info(f"  FID-ROI: real ROI patches vs CASDA ROI outputs")
        logging.info(f"{'─'*50}")

        # ── roi_granularity 설정 로드 ──
        granularity_cfg = fid_config.get('roi_granularity', {})
        by_defect_subtype = granularity_cfg.get('by_defect_subtype', False)
        by_background_type = granularity_cfg.get('by_background_type', False)
        cross_class_subtype = granularity_cfg.get('cross_class_subtype', False)
        min_samples = granularity_cfg.get('min_samples', 50)

        # ── Real ROI 수집 (roi_metadata.csv 기반) ──
        raw_roi_csv = fid_config.get('roi_metadata_csv', '')
        if not raw_roi_csv:
            # CLI --casda-dir 사용 시 자동 탐색:
            #   casda_dir/../roi_patches_*/roi_metadata.csv
            #   또는 data/processed/roi_patches/roi_metadata.csv
            for candidate in [
                Path(casda_config.get('full_dir', '')) / '..' / 'roi_patches' / 'roi_metadata.csv',
                PROJECT_ROOT / 'data' / 'processed' / 'roi_patches' / 'roi_metadata.csv',
            ]:
                candidate = candidate.resolve()
                if candidate.exists():
                    raw_roi_csv = str(candidate)
                    break

        roi_metadata_path = (
            Path(raw_roi_csv) if raw_roi_csv and os.path.isabs(raw_roi_csv)
            else (PROJECT_ROOT / raw_roi_csv if raw_roi_csv else None)
        )

        if roi_metadata_path and roi_metadata_path.exists():
            roi_df = pd.read_csv(roi_metadata_path)
            roi_real_paths = [
                str(p) for p in roi_df['roi_image_path'].tolist()
                if os.path.exists(str(p))
            ]
            logging.info(f"  Real ROI patches: {len(roi_real_paths)} "
                         f"(from {roi_metadata_path.name})")
        else:
            roi_real_paths = []
            logging.warning(
                f"  roi_metadata_csv not found: {roi_metadata_path} "
                f"— fid_roi 건너뜀"
            )

        # ── Synthetic ROI 수집 (roi_dir 우선 → full_dir fallback) ──
        raw_roi_dir = casda_config.get('roi_dir', '') or casda_config.get('full_dir', 'data/augmented/casda_full')
        casda_roi_dir = (
            Path(raw_roi_dir) if os.path.isabs(raw_roi_dir)
            else PROJECT_ROOT / raw_roi_dir
        )
        casda_full_images = _collect_images_from_dir(casda_roi_dir)
        logging.info(f"  Synthetic ROI ({casda_roi_dir}): {len(casda_full_images)}")

        if roi_real_paths and casda_full_images:
            rng_roi = random.Random(fid_seed)
            roi_real_sampled = sample_images(roi_real_paths, max_images, rng_roi)
            roi_gen_sampled = sample_images(casda_full_images, max_images, rng_roi)

            logging.info(
                f"  Computing FID-ROI: {len(roi_real_sampled)} real vs "
                f"{len(roi_gen_sampled)} synthetic"
            )

            # Per-class 그룹핑 (기존 호환)
            roi_real_by_class: Optional[dict] = None
            roi_gen_by_class: Optional[dict] = None
            if per_class:
                # metadata.json 우선 → 없으면 파일명 파싱 fallback
                roi_gen_by_class = _group_gen_images_by_class(
                    casda_roi_dir / "metadata.json", casda_roi_dir)
                if roi_gen_by_class is None:
                    logging.info("    metadata.json 없음 — 파일명에서 class_id 파싱")
                    roi_gen_by_class = _group_gen_images_by_filename(casda_full_images)

                if roi_gen_by_class and roi_metadata_path and roi_metadata_path.exists():
                    pc_rng = random.Random(fid_seed + 1)
                    roi_real_by_class = _build_real_roi_by_class_from_csv(
                        roi_metadata_path,
                        list(roi_gen_by_class.keys()),
                        max_images, pc_rng,
                    )
                    for cid in sorted(roi_gen_by_class.keys()):
                        logging.info(
                            f"    Class {cid + 1}: "
                            f"{len(roi_real_by_class.get(cid, []))} real ROI, "
                            f"{len(roi_gen_by_class.get(cid, []))} synthetic ROI"
                        )
                    remove_empty_classes(roi_real_by_class, roi_gen_by_class)
                    if not roi_gen_by_class:
                        roi_gen_by_class = None
                        roi_real_by_class = None

            # FID 계산: overall + per-class (feature 일괄 추출)
            fid_calc.clear_cache()  # ROI와 composed의 feature 공간이 다르므로 초기화
            roi_results = fid_calc.compute_fid_with_preextract(
                all_real_paths=roi_real_sampled,
                all_gen_paths=roi_gen_sampled,
                real_by_class=roi_real_by_class,
                gen_by_class=roi_gen_by_class,
                batch_size=batch_size,
                num_workers=num_workers,
                cache_dir=cache_dir,
            )

            # 키 이름에 _roi 접미사 추가
            for k, v in roi_results.items():
                if k == 'fid_overall':
                    results['fid_roi_overall'] = v
                else:
                    # "Class1_FID" → "Class1_FID_roi"
                    results[f"{k}_roi"] = v

            # ==============================================================
            # FID-ROI 세분화 분석 (defect_subtype, background_type, 교차)
            # roi_metadata.csv의 메타데이터를 활용하여 세분화 FID 계산
            # ==============================================================
            needs_granular = (by_defect_subtype or by_background_type or cross_class_subtype)
            if needs_granular and roi_metadata_path and roi_metadata_path.exists():
                logging.info(f"\n  {'─'*46}")
                logging.info(f"  FID-ROI 세분화 분석 (min_samples={min_samples})")
                logging.info(f"  {'─'*46}")

                # 룩업 테이블 구축: sample_name → {class_id, defect_subtype, background_type}
                roi_lookup = _build_roi_lookup_from_metadata(roi_metadata_path)
                logging.info(f"  ROI lookup 구축 완료: {len(roi_lookup)} entries")

                # ── defect_subtype별 FID-ROI ──
                if by_defect_subtype:
                    logging.info(f"\n  [defect_subtype별 FID-ROI]")
                    gen_by_subtype = _group_images_by_dimension(
                        casda_full_images, roi_lookup, 'defect_subtype', source='synthetic')
                    real_by_subtype = _group_real_roi_by_dimension(
                        roi_metadata_path, 'defect_subtype',
                        max_images, random.Random(fid_seed + 20))

                    if gen_by_subtype and real_by_subtype:
                        subtype_results = _compute_granular_fid(
                            fid_calc, real_by_subtype, gen_by_subtype,
                            'defect_subtype', min_samples,
                            batch_size, num_workers, cache_dir)
                        results['fid_roi_by_defect_subtype'] = subtype_results
                    else:
                        logging.warning("    defect_subtype 그룹핑 실패 — 스킵")

                # ── background_type별 FID-ROI ──
                if by_background_type:
                    logging.info(f"\n  [background_type별 FID-ROI]")
                    gen_by_bg = _group_images_by_dimension(
                        casda_full_images, roi_lookup, 'background_type', source='synthetic')
                    real_by_bg = _group_real_roi_by_dimension(
                        roi_metadata_path, 'background_type',
                        max_images, random.Random(fid_seed + 30))

                    if gen_by_bg and real_by_bg:
                        bg_results = _compute_granular_fid(
                            fid_calc, real_by_bg, gen_by_bg,
                            'background_type', min_samples,
                            batch_size, num_workers, cache_dir)
                        results['fid_roi_by_background_type'] = bg_results
                    else:
                        logging.warning("    background_type 그룹핑 실패 — 스킵")

                # ── class_id × defect_subtype 교차 FID-ROI ──
                if cross_class_subtype:
                    logging.info(f"\n  [class_id × defect_subtype 교차 FID-ROI]")
                    gen_by_cross = _group_synthetic_by_cross_dimension(
                        casda_full_images, roi_lookup)
                    real_by_cross = _group_real_roi_by_dimension(
                        roi_metadata_path, 'class_subtype',
                        max_images, random.Random(fid_seed + 40))

                    if gen_by_cross and real_by_cross:
                        cross_results = _compute_granular_fid(
                            fid_calc, real_by_cross, gen_by_cross,
                            'class_subtype', min_samples,
                            batch_size, num_workers, cache_dir)
                        results['fid_roi_by_class_subtype'] = cross_results
                    else:
                        logging.warning("    class × defect_subtype 교차 그룹핑 실패 — 스킵")

            elif needs_granular:
                logging.warning(
                    "  FID-ROI 세분화 분석 필요하지만 roi_metadata_csv 없음 — 스킵")

        else:
            logging.warning("  fid_roi 계산 불가: real ROI 또는 synthetic ROI 부족")
            results['fid_roi_overall'] = float('inf')

    # ==================================================================
    # fid_composed: 전체 이미지 vs pruning된 합성 이미지
    #   real: train_images/ (1600x256)
    #   synthetic: casda_composed/에서 pruning된 top-k (1600x256 Poisson Blending)
    #   --no-fid-pruning 시 composed 전체 대상
    # ==================================================================
    if fid_mode in ('composed', 'both'):
        logging.info(f"\n{'─'*50}")
        logging.info(f"  FID-Composed: real full images vs CASDA composed (pruned)")
        logging.info(f"{'─'*50}")

        # ── Real 전체 이미지 수집 ──
        real_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        real_images = [str(p) for p in real_images]

        if not real_images:
            logging.warning("  No real images found for FID-composed computation")
            results['fid_composed_overall'] = float('inf')
            results['fid_overall'] = float('inf')
        else:
            # ── Synthetic composed 이미지 수집 (pruning 적용) ──
            raw_composed_dir = casda_config.get('composed_dir', 'data/augmented/casda_composed')
            composed_dir = (
                Path(raw_composed_dir) if os.path.isabs(raw_composed_dir)
                else PROJECT_ROOT / raw_composed_dir
            )

            # Pruning 설정 읽기
            use_pruning = fid_config.get('use_pruning', True)  # 기본값: pruning 사용
            pruning_top_k = casda_config.get('pruning_top_k', 2000)
            pruning_threshold = casda_config.get('suitability_threshold', 0.0)
            pruning_stratified = True  # 기본 stratified

            # dataset_groups.casda_composed_pruning 설정도 참조
            pruning_grp = config.get('dataset_groups', {}).get('casda_composed_pruning', {})
            pruning_cfg = pruning_grp.get('casda_pruning', {})
            if pruning_cfg.get('enabled', False):
                pruning_top_k = pruning_cfg.get('top_k', pruning_top_k)
                pruning_threshold = pruning_cfg.get('suitability_threshold', pruning_threshold)
                pruning_stratified = pruning_cfg.get('stratified', pruning_stratified)

            metadata_path = composed_dir / "metadata.json"

            if use_pruning and metadata_path.exists():
                logging.info(f"  Pruning mode: top_k={pruning_top_k}, "
                             f"threshold={pruning_threshold}, "
                             f"stratified={pruning_stratified}")
                composed_images = _load_pruned_image_paths(
                    metadata_path, composed_dir,
                    top_k=pruning_top_k,
                    suitability_threshold=pruning_threshold,
                    stratified=pruning_stratified,
                )
            else:
                if use_pruning and not metadata_path.exists():
                    logging.warning(
                        f"  metadata.json not found ({metadata_path}) "
                        f"— pruning 불가, composed 전체 사용"
                    )
                else:
                    logging.info("  Pruning disabled (--no-fid-pruning) — composed 전체 사용")
                composed_images = _collect_images_from_dir(composed_dir)

            logging.info(f"  Real images: {len(real_images)}, "
                         f"Composed synthetic (pruned): {len(composed_images)}")

            if composed_images:
                rng_comp = random.Random(fid_seed + 10)  # ROI와 다른 시드
                real_sampled = sample_images(real_images, max_images, rng_comp)
                comp_sampled = sample_images(composed_images, max_images, rng_comp)

                logging.info(
                    f"  Computing FID-Composed: {len(real_sampled)} real vs "
                    f"{len(comp_sampled)} synthetic"
                )

                # Per-class 그룹핑 (pruning 적용)
                comp_real_by_class: Optional[dict] = None
                comp_gen_by_class: Optional[dict] = None
                if per_class:
                    if use_pruning and metadata_path.exists():
                        comp_gen_by_class = _group_pruned_by_class(
                            metadata_path, composed_dir,
                            top_k=pruning_top_k,
                            suitability_threshold=pruning_threshold,
                            stratified=pruning_stratified,
                        )
                    else:
                        comp_gen_by_class = _group_gen_images_by_class(
                            metadata_path, composed_dir)

                    if comp_gen_by_class and annotation_path.exists():
                        pc_rng = random.Random(fid_seed + 11)
                        comp_real_by_class = _build_real_by_class_from_csv(
                            annotation_path, image_dir,
                            list(comp_gen_by_class.keys()),
                            max_images, pc_rng,
                        )
                        for cid in sorted(comp_gen_by_class.keys()):
                            logging.info(
                                f"    Class {cid + 1}: "
                                f"{len(comp_real_by_class.get(cid, []))} real, "
                                f"{len(comp_gen_by_class.get(cid, []))} composed"
                            )
                        remove_empty_classes(comp_real_by_class, comp_gen_by_class)
                        if not comp_gen_by_class:
                            comp_gen_by_class = None
                            comp_real_by_class = None
                    elif comp_gen_by_class and not annotation_path.exists():
                        logging.warning(
                            f"  annotation_csv not found: {annotation_path} "
                            f"— per-class FID-composed will use mixed real images"
                        )
                        for cid in comp_gen_by_class.keys():
                            comp_real_by_class = comp_real_by_class or {}
                            comp_real_by_class[cid] = real_sampled

                # FID 계산 (feature 일괄 추출)
                fid_calc.clear_cache()
                comp_results = fid_calc.compute_fid_with_preextract(
                    all_real_paths=real_sampled,
                    all_gen_paths=comp_sampled,
                    real_by_class=comp_real_by_class,
                    gen_by_class=comp_gen_by_class,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    cache_dir=cache_dir,
                )

                # 키 이름에 _composed 접미사 추가
                for k, v in comp_results.items():
                    if k == 'fid_overall':
                        results['fid_composed_overall'] = v
                    else:
                        results[f"{k}_composed"] = v

                # 하위 호환: fid_overall = fid_composed_overall
                results['fid_overall'] = results.get('fid_composed_overall', float('inf'))
            else:
                logging.warning("  No composed images found — fid_composed 건너뜀")
                results['fid_composed_overall'] = float('inf')
                results['fid_overall'] = float('inf')

    # ==================================================================
    # 결과 저장
    # ==================================================================
    fid_path = experiment_dir / "fid_results.json"
    with open(fid_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"\nFID results saved to: {fid_path}")

    # 요약 로그
    for key in ['fid_roi_overall', 'fid_composed_overall']:
        if key in results:
            val = results[key]
            if val is not None and val != float('inf'):
                logging.info(f"  {key}: {val:.2f}")
            else:
                logging.info(f"  {key}: {val}")

    # 세분화 FID-ROI 요약
    for granular_key in ['fid_roi_by_defect_subtype', 'fid_roi_by_background_type',
                         'fid_roi_by_class_subtype']:
        if granular_key in results:
            logging.info(f"\n  {granular_key}:")
            for dim_val, fid_val in sorted(results[granular_key].items()):
                if fid_val is not None:
                    logging.info(f"    {dim_val}: {fid_val:.2f}")
                else:
                    logging.info(f"    {dim_val}: N/A (insufficient samples)")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CASDA FID Score Evaluation (독립 실행 스크립트)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FID Modes:
  both       FID-ROI + FID-Composed 양쪽 모두 (기본값)
  roi        FID-ROI만 계산 (ROI vs ROI)
  composed   FID-Composed만 계산 (전체 이미지 vs 전체 이미지)

Examples:
  # 기본 사용
  python scripts/run_fid.py --config configs/benchmark_experiment.yaml

  # Colab: 명시적 경로 지정
  python scripts/run_fid.py \\
      --config configs/benchmark_experiment.yaml \\
      --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --casda-dir /content/drive/MyDrive/data/Severstal/augmented_dataset_v5.6 \\
      --casda-roi-dir /content/drive/MyDrive/data/Severstal/augmented_images_v5.5/generated \\
      --output-dir /content/drive/MyDrive/data/Severstal/fid_results_v5.6 \\
      --workers 12

  # ROI metadata 명시 + FID-ROI만 실행
  python scripts/run_fid.py \\
      --config configs/benchmark_experiment.yaml \\
      --roi-metadata-csv /content/drive/MyDrive/data/Severstal/roi_patches_v5.1/roi_metadata.csv \\
      --fid-mode roi

  # FID-ROI만 실행
  python scripts/run_fid.py --config configs/benchmark_experiment.yaml --fid-mode roi
        """,
    )
    parser.add_argument('--config', type=str, default='configs/benchmark_experiment.yaml',
                        help='Path to experiment config YAML (benchmark_experiment.yaml)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='이미지 디렉토리 경로 (overrides config dataset.image_dir)')
    parser.add_argument('--csv', type=str, default=None,
                        help='annotation CSV 경로 (overrides config dataset.annotation_csv)')
    parser.add_argument('--casda-dir', type=str, default=None,
                        help='CASDA 데이터 상위 디렉토리 (casda_full/, casda_composed/ 포함). '
                             'Overrides config dataset.casda paths')
    parser.add_argument('--casda-roi-dir', type=str, default=None,
                        help='ControlNet 생성 ROI 이미지 디렉토리 (generated/). '
                             'FID-ROI 계산 시 casda_full 대신 이 경로를 사용. '
                             '(overrides config dataset.casda.roi_dir)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='결과 저장 디렉토리 (fid_results.json, fid_cache/). '
                             'Overrides config experiment.output_dir')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--fid-mode', type=str, default=None,
                        choices=['roi', 'composed', 'both'],
                        help='FID 계산 모드: roi, composed, both (기본: YAML 설정 또는 both)')
    parser.add_argument('--workers', type=int, default=None,
                        help='DataLoader num_workers (이미지 I/O 병렬화). '
                             'Overrides config evaluation.fid.num_workers. '
                             '기본: YAML 설정 또는 4')
    parser.add_argument('--roi-metadata-csv', type=str, default=None,
                        help='roi_metadata.csv 경로 (real ROI 목록). '
                             'Overrides config evaluation.fid.roi_metadata_csv. '
                             '미지정 시 --casda-dir 기반 자동 탐색')
    parser.add_argument('--no-fid-pruning', action='store_true', default=False,
                        help='FID-Composed 계산 시 pruning 없이 composed 전체 사용. '
                             '기본: pruning 적용 (metadata.json의 suitability_score 기반 top-k)')

    args = parser.parse_args()

    # ── Config 로드 ──
    config = load_config(args.config)

    # ── CLI → config override ──
    if args.data_dir:
        config['dataset']['image_dir'] = args.data_dir
        print(f"[INFO] image_dir overridden to: {args.data_dir}")
    if args.csv:
        config['dataset']['annotation_csv'] = args.csv
        print(f"[INFO] annotation_csv overridden to: {args.csv}")
    if args.casda_dir:
        casda_base = args.casda_dir
        if 'casda' not in config['dataset']:
            config['dataset']['casda'] = {}
        config['dataset']['casda']['full_dir'] = os.path.join(casda_base, 'casda_full')
        config['dataset']['casda']['pruning_dir'] = os.path.join(casda_base, 'casda_pruning')
        config['dataset']['casda']['composed_dir'] = os.path.join(casda_base, 'casda_composed')
        print(f"[INFO] casda paths overridden: {casda_base}/casda_{{full,pruning,composed}}")

        # FID-ROI용 roi_metadata.csv 자동 탐색
        fid_cfg = config.setdefault('evaluation', {}).setdefault('fid', {})
        if not fid_cfg.get('roi_metadata_csv'):
            casda_parent = Path(casda_base).parent
            for candidate_name in ['roi_patches', 'roi_patches_v5.1', 'roi_patches_v5']:
                candidate = casda_parent / candidate_name / 'roi_metadata.csv'
                if candidate.exists():
                    fid_cfg['roi_metadata_csv'] = str(candidate)
                    print(f"[INFO] roi_metadata_csv auto-detected: {candidate}")
                    break

    if args.casda_roi_dir:
        if 'casda' not in config['dataset']:
            config['dataset']['casda'] = {}
        config['dataset']['casda']['roi_dir'] = args.casda_roi_dir
        print(f"[INFO] casda roi_dir overridden to: {args.casda_roi_dir}")

    if args.fid_mode:
        fid_cfg = config.setdefault('evaluation', {}).setdefault('fid', {})
        fid_cfg['fid_mode'] = args.fid_mode
        print(f"[INFO] fid_mode overridden to: {args.fid_mode}")

    if args.workers is not None:
        fid_cfg = config.setdefault('evaluation', {}).setdefault('fid', {})
        fid_cfg['num_workers'] = args.workers
        print(f"[INFO] fid num_workers overridden to: {args.workers}")

    if args.roi_metadata_csv:
        fid_cfg = config.setdefault('evaluation', {}).setdefault('fid', {})
        fid_cfg['roi_metadata_csv'] = args.roi_metadata_csv
        print(f"[INFO] roi_metadata_csv overridden to: {args.roi_metadata_csv}")

    if args.no_fid_pruning:
        fid_cfg = config.setdefault('evaluation', {}).setdefault('fid', {})
        fid_cfg['use_pruning'] = False
        print(f"[INFO] FID-Composed pruning disabled (--no-fid-pruning)")

    # ── Device 설정 ──
    import torch
    device = args.device or config.get('experiment', {}).get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # ── Output 디렉토리 결정 ──
    output_dir = Path(
        args.output_dir
        or config.get('experiment', {}).get('output_dir', 'outputs/fid_results')
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging 설정 ──
    log_path = experiment_dir / "fid_evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"CASDA FID Evaluation (standalone)")
    logging.info(f"Config: {args.config}")
    logging.info(f"Output: {experiment_dir}")
    logging.info(f"Device: {device}")

    # ── Config 사본 저장 ──
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── FID 평가 실행 ──
    results = run_fid_evaluation(config, experiment_dir, device)

    logging.info(f"\nFID evaluation complete. Results saved to: {experiment_dir}")
    return results


if __name__ == "__main__":
    main()
