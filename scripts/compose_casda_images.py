#!/usr/bin/env python3
"""
Compose CASDA images using Poisson Blending.
ControlNet 생성 512x512 결함 ROI를 Poisson Blending으로 결함 없는 원본(1600x256)에 합성.

파이프라인:
  1. 메타데이터 로딩 (generation_summary.json + packaged_roi_metadata.csv)
  2. 결함 없는 배경 이미지 풀 구축 (train.csv에 없는 이미지)
  3. 배경 유형 분석 + 호환성 매칭 (캐시 지원)
  4. PoissonBlender로 각 생성 이미지를 1600x256 배경에 합성 (병렬 지원)
  5. 합성 이미지 + 전체 크기 마스크 + metadata.json 저장

성능 최적화:
  - --workers N : 멀티프로세싱 병렬 합성 (배경 분석 + Poisson 블렌딩)
  - --bg-cache path : 배경 유형 분석 결과 캐싱 (재실행 시 분석 건너뜀)
  - --png-compression 1 : PNG 압축 레벨 조절 (기본 1, 빠른 저장)
  - --workers -1 : CPU 코어 수 자동 감지

출력:
  casda_composed/
  ├── images/          # 1600x256 합성 이미지 (.png)
  ├── masks/           # 1600x256 전체 크기 마스크 (.png)
  └── metadata.json    # 메타데이터 (YOLO bbox, suitability_score 포함)

Usage:
  # 기본 (순차 처리):
  python scripts/compose_casda_images.py \\
    --generated-dir outputs/v5.1/test_results_v5.1/generated \\
    --hint-dir data/processed/controlnet_dataset/hints \\
    --metadata-csv data/processed/controlnet_dataset/packaged_roi_metadata.csv \\
    --summary-json outputs/v5.1/test_results_v5.1/generation_summary.json \\
    --clean-images-dir data/raw/train_images \\
    --train-csv data/raw/train.csv \\
    --output-dir data/augmented/casda_composed

  # 고속 (8 워커 + 배경 캐시 + 빠른 PNG):
  python scripts/compose_casda_images.py \\
    --generated-dir outputs/v5.1/test_results_v5.1/generated \\
    --hint-dir data/processed/controlnet_dataset/hints \\
    --metadata-csv data/processed/controlnet_dataset/packaged_roi_metadata.csv \\
    --summary-json outputs/v5.1/test_results_v5.1/generation_summary.json \\
    --clean-images-dir data/raw/train_images \\
    --train-csv data/raw/train.csv \\
    --output-dir data/augmented/casda_composed \\
    --workers 8 \\
    --bg-cache data/cache/bg_types.json \\
    --png-compression 1

  # 자동 워커 수 감지:
  python scripts/compose_casda_images.py \\
    ... \\
    --workers -1 --bg-cache data/cache/bg_types.json
"""

import argparse
import ast
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.poisson_blender import PoissonBlender

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# 호환성 매트릭스 (background_library.py에서 복제)
# ============================================================================
# defect_subtype → {background_type: compatibility_score}
COMPATIBILITY_MATRIX = {
    'compact_blob': {
        'smooth': 1.0, 'vertical_stripe': 0.8, 'horizontal_stripe': 0.8,
        'textured': 0.5, 'complex_pattern': 0.2,
    },
    'linear_scratch': {
        'smooth': 0.8, 'vertical_stripe': 1.0, 'horizontal_stripe': 1.0,
        'textured': 0.5, 'complex_pattern': 0.2,
    },
    'scattered_defects': {
        'smooth': 1.0, 'vertical_stripe': 0.8, 'horizontal_stripe': 0.8,
        'textured': 0.5, 'complex_pattern': 0.2,
    },
    'elongated_region': {
        'smooth': 0.8, 'vertical_stripe': 1.0, 'horizontal_stripe': 1.0,
        'textured': 0.5, 'complex_pattern': 0.2,
    },
}

# 모든 배경 유형 (호환성 없을 때 폴백용)
ALL_BACKGROUND_TYPES = ['smooth', 'vertical_stripe', 'horizontal_stripe',
                        'textured', 'complex_pattern']


# ============================================================================
# 유틸리티 함수
# ============================================================================

def parse_bbox_string(bbox_str) -> Tuple[int, int, int, int]:
    """
    bbox 문자열 "(x1, y1, x2, y2)"을 정수 튜플로 파싱.
    이미 튜플이면 그대로 반환.
    """
    if isinstance(bbox_str, (tuple, list)):
        return tuple(int(v) for v in bbox_str)
    try:
        return tuple(int(v) for v in ast.literal_eval(bbox_str))
    except (ValueError, SyntaxError):
        raise ValueError(f"bbox 문자열 파싱 실패: {bbox_str}")


def parse_class_id_from_filename(filename: str) -> int:
    """파일명에서 0-indexed class_id를 추출."""
    match = re.search(r"_class(\d+)_", filename)
    if match:
        return int(match.group(1)) - 1  # 1-indexed → 0-indexed
    raise ValueError(f"class_id 추출 실패: {filename}")


def filename_to_sample_name(filename: str) -> str:
    """생성 파일명 → sample_name 변환. 예: foo_gen0.png → foo"""
    match = re.match(r"(.+)_gen\d+\.png$", filename)
    if match:
        return match.group(1)
    return Path(filename).stem


def find_clean_images(train_csv_path: Path, train_images_dir: Path) -> List[str]:
    """
    결함 없는 이미지 파일명 목록을 반환.
    train.csv에 없는 이미지 = 결함 없는 이미지.
    """
    all_images = set(f.name for f in train_images_dir.glob("*.jpg"))
    train_df = pd.read_csv(train_csv_path)
    images_with_defects = set(train_df['ImageId'].unique())
    clean = sorted(all_images - images_with_defects)
    return clean


def classify_background_simple(image: np.ndarray) -> str:
    """
    1600x256 이미지의 중앙 영역에서 배경 유형을 간단히 분류.
    
    BackgroundCharacterizer의 전체 파이프라인을 사용하지 않고,
    엣지 방향성과 분산으로 빠르게 분류한다.
    128x128 패치로 축소하여 연산량을 줄임 (256x256 대비 4배 빠름).
    
    Returns:
        'smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern'
    """
    h, w = image.shape[:2]
    
    # 중앙 256x256 패치 추출 후 128x128로 축소
    cx = w // 2
    x1 = max(0, cx - 128)
    x2 = min(w, x1 + 256)
    patch = image[:, x1:x2]
    
    # 128x128로 축소하여 연산량 절감
    patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA)
    
    # 그레이스케일 변환
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    
    variance = np.var(gray.astype(np.float32))
    
    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # 분산이 매우 낮으면 smooth
    if variance < 200 and edge_density < 0.02:
        return 'smooth'
    
    # 소벨 필터로 방향성 분석
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    energy_x = np.mean(np.abs(sobel_x))
    energy_y = np.mean(np.abs(sobel_y))
    
    # 방향성 비율
    total_energy = energy_x + energy_y + 1e-7
    x_ratio = energy_x / total_energy
    y_ratio = energy_y / total_energy
    
    # 수직 줄무늬: x 방향 엣지 우세 (세로선 → 가로 그래디언트)
    if x_ratio > 0.65 and edge_density > 0.02:
        return 'vertical_stripe'
    # 수평 줄무늬: y 방향 엣지 우세
    if y_ratio > 0.65 and edge_density > 0.02:
        return 'horizontal_stripe'
    
    # 엣지 밀도가 높으면 complex
    if edge_density > 0.15:
        return 'complex_pattern'
    
    # 나머지는 textured
    if variance > 500 or edge_density > 0.05:
        return 'textured'
    
    return 'smooth'


def compute_mean_brightness(image: np.ndarray) -> float:
    """
    이미지의 평균 밝기를 계산한다 (0~255 스케일).
    
    그레이스케일 변환 후 전체 평균을 반환.
    Severstal 강재 이미지는 보통 80~200 범위.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


def _classify_single_background(args: Tuple[str, str]) -> Tuple[str, str, float]:
    """
    단일 배경 이미지를 분류하고 밝기를 측정하는 워커 함수 (멀티프로세싱용).
    
    Args:
        args: (image_name, images_dir_str) 튜플
        
    Returns:
        (image_name, bg_type, mean_brightness) 튜플.
        로드 실패 시 (image_name, '', 0.0) 반환.
    """
    name, images_dir_str = args
    img_path = Path(images_dir_str) / name
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return (name, '', 0.0)
    bg_type = classify_background_simple(img)
    brightness = compute_mean_brightness(img)
    return (name, bg_type, brightness)


def _compute_gen_brightness(img_path_str: str) -> Tuple[str, float]:
    """
    생성 이미지 1장의 평균 밝기를 계산하는 워커 함수 (멀티프로세싱용).
    
    Args:
        img_path_str: 이미지 파일 절대경로 문자열
        
    Returns:
        (filename, mean_brightness) 튜플.
        로드 실패 시 밝기 128.0 (중간값) 반환.
    """
    img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (Path(img_path_str).name, 128.0)
    return (Path(img_path_str).name, float(np.mean(img)))


def build_quality_map(
    summary: dict, quality_json_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    filename → quality_score 매핑 구축.
    package_casda_data.py의 build_quality_map()과 동일한 로직.
    """
    sample_scores = []
    
    if quality_json_path and quality_json_path.exists():
        logger.info(f"품질 점수 로딩: {quality_json_path}")
        with open(quality_json_path) as f:
            quality_data = json.load(f)
        if isinstance(quality_data, list):
            sample_scores = quality_data
        elif isinstance(quality_data, dict):
            quality_section = quality_data.get("quality", quality_data)
            sample_scores = quality_section.get("sample_scores", [])
    
    if not sample_scores:
        quality_section = summary.get("quality", {})
        sample_scores = quality_section.get("sample_scores", [])
    
    if not sample_scores:
        logger.warning("품질 점수를 찾을 수 없음. 기본값 0.5 사용.")
        return {}
    
    quality_map = {}
    for entry in sample_scores:
        fname = entry.get("filename", "")
        score = entry.get("quality_score", 0.0)
        quality_map[fname] = score
    
    # sample_name 폴백 매핑
    sample_name_scores = {}
    for fname, score in quality_map.items():
        sname = filename_to_sample_name(fname)
        if sname not in sample_name_scores:
            sample_name_scores[sname] = score
    
    quality_map["__sample_name_fallback__"] = sample_name_scores
    
    real_count = len(quality_map) - 1
    logger.info(f"품질 점수 로딩 완료: {real_count}개 직접 매핑, "
                f"{len(sample_name_scores)}개 sample-name 그룹")
    return quality_map


def get_quality_score(
    quality_map: dict, filename: str, default: float = 0.5
) -> float:
    """quality_map에서 점수 조회. 폴백: sample_name → default."""
    if filename in quality_map:
        return quality_map[filename]
    
    sample_name_scores = quality_map.get("__sample_name_fallback__", {})
    if sample_name_scores:
        sname = filename_to_sample_name(filename)
        if sname in sample_name_scores:
            return sample_name_scores[sname]
    
    return default


# ============================================================================
# 배경 이미지 풀 관리
# ============================================================================

class BackgroundPool:
    """
    결함 없는 배경 이미지 풀.
    배경 유형별로 인덱싱하여 호환성 기반 선택을 지원한다.
    밝기(brightness) 매칭으로 Poisson blending 시 밝기 halo 아티팩트를 최소화한다.
    
    성능 최적화:
    - 배경 유형+밝기 분석 결과를 JSON 캐시 파일로 저장/로드
    - 멀티프로세싱으로 배경 분석 병렬화
    """
    
    def __init__(
        self,
        clean_image_names: List[str],
        images_dir: Path,
        cache_bg_types: bool = True,
        max_analyze: int = 5000,
        bg_cache_path: Optional[Path] = None,
        num_workers: int = 0,
    ):
        """
        Args:
            clean_image_names: 결함 없는 이미지 파일명 목록
            images_dir: 이미지 디렉토리 경로
            cache_bg_types: 배경 유형을 캐싱할지 여부
            max_analyze: 배경 유형 분석할 최대 이미지 수
            bg_cache_path: 배경 유형+밝기 캐시 JSON 경로 (None이면 캐시 미사용)
            num_workers: 병렬 분석 워커 수 (0이면 순차 처리)
        """
        self.images_dir = images_dir
        self.clean_names = clean_image_names
        self.num_workers = num_workers
        
        # 배경 유형별 인덱스: {bg_type: [filename, ...]}
        self.type_index: Dict[str, List[str]] = {t: [] for t in ALL_BACKGROUND_TYPES}
        self.bg_types: Dict[str, str] = {}  # filename → bg_type
        self.bg_brightness: Dict[str, float] = {}  # filename → mean_brightness (0~255)
        
        if cache_bg_types:
            self._analyze_backgrounds(max_analyze, bg_cache_path)
    
    def _load_cache(self, cache_path: Path) -> dict:
        """
        캐시 파일에서 배경 유형+밝기 매핑을 로드한다.
        
        v2 캐시 형식: {"_version": 2, "types": {...}, "brightness": {...}}
        v1 호환: {filename: bg_type, ...} (밝기 없음, 재분석 필요)
        """
        if cache_path and cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                # v2 형식 확인
                if isinstance(data, dict) and data.get("_version") == 2:
                    types = data.get("types", {})
                    brightness = data.get("brightness", {})
                    logger.info(f"배경 캐시 v2 로드 성공: {cache_path} "
                                f"({len(types)}개 유형, {len(brightness)}개 밝기)")
                    return {"types": types, "brightness": brightness}
                # v1 형식 (하위 호환): 유형만 있고 밝기 없음
                if isinstance(data, dict) and "_version" not in data:
                    logger.info(f"배경 캐시 v1 로드 (밝기 없음, 재분석 예정): "
                                f"{cache_path} ({len(data)}개)")
                    return {"types": data, "brightness": {}}
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"배경 유형 캐시 로드 실패: {e}")
        return {"types": {}, "brightness": {}}
    
    def _save_cache(self, cache_path: Path):
        """배경 유형+밝기 매핑을 v2 캐시 파일에 저장한다."""
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    "_version": 2,
                    "types": self.bg_types,
                    "brightness": {k: round(v, 2) for k, v in self.bg_brightness.items()},
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
                logger.info(f"배경 캐시 v2 저장: {cache_path} "
                            f"({len(self.bg_types)}개 유형, "
                            f"{len(self.bg_brightness)}개 밝기)")
            except IOError as e:
                logger.warning(f"배경 유형 캐시 저장 실패: {e}")
    
    def _analyze_backgrounds(
        self, max_analyze: int, cache_path: Optional[Path] = None
    ):
        """배경 이미지들의 유형과 밝기를 분석하여 인덱스 구축."""
        names_to_analyze = self.clean_names[:max_analyze]
        
        # 캐시에서 로드 시도
        cached = self._load_cache(cache_path) if cache_path else {"types": {}, "brightness": {}}
        cached_types = cached.get("types", {})
        cached_brightness = cached.get("brightness", {})
        
        # 유형+밝기 모두 캐시에 있는 이미지만 완전 캐시 히트
        names_full_cached = [
            n for n in names_to_analyze
            if n in cached_types and n in cached_brightness
        ]
        # 유형은 있지만 밝기가 없는 이미지 → 밝기만 재분석
        names_brightness_miss = [
            n for n in names_to_analyze
            if n in cached_types and n not in cached_brightness
        ]
        # 유형도 없는 이미지 → 전체 분석
        names_uncached = [
            n for n in names_to_analyze
            if n not in cached_types
        ]
        
        # 완전 캐시 히트 적용
        for name in names_full_cached:
            bg_type = cached_types[name]
            self.bg_types[name] = bg_type
            self.bg_brightness[name] = float(cached_brightness[name])
            self.type_index[bg_type].append(name)
        
        if names_full_cached:
            logger.info(f"배경 캐시 완전 히트: {len(names_full_cached)}장")
        
        # 밝기만 누락된 이미지 처리 (유형 캐시 활용, 밝기만 측정)
        if names_brightness_miss:
            logger.info(f"밝기 재분석 필요: {len(names_brightness_miss)}장 "
                        f"(유형은 캐시에서 로드)")
            for name in tqdm(names_brightness_miss, desc="밝기 측정"):
                bg_type = cached_types[name]
                self.bg_types[name] = bg_type
                self.type_index[bg_type].append(name)
                # 밝기만 측정
                img_path = self.images_dir / name
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    self.bg_brightness[name] = compute_mean_brightness(img)
                else:
                    self.bg_brightness[name] = 128.0  # 로드 실패 시 중간값
        
        if not names_uncached:
            if not names_brightness_miss:
                logger.info("모든 배경 데이터가 캐시에서 로드됨 — 분석 건너뜀")
            self._log_type_stats()
            # 캐시 갱신 (밝기 추가된 경우)
            if names_brightness_miss and cache_path:
                self._save_cache(cache_path)
            return
        
        logger.info(f"배경 유형+밝기 분석 시작: {len(names_uncached)}장 (캐시 미스)...")
        t_start = time.time()
        
        if self.num_workers > 1 and len(names_uncached) > 10:
            # 멀티프로세싱 병렬 분석
            self._analyze_parallel(names_uncached)
        else:
            # 순차 분석
            self._analyze_sequential(names_uncached)
        
        elapsed = time.time() - t_start
        logger.info(f"배경 분석 완료: {len(names_uncached)}장, {elapsed:.1f}초")
        
        # 캐시 저장
        if cache_path:
            self._save_cache(cache_path)
        
        self._log_type_stats()
    
    def _analyze_sequential(self, names: List[str]):
        """순차적으로 배경 유형과 밝기를 분석한다."""
        for name in tqdm(names, desc="배경 유형+밝기 분석"):
            img_path = self.images_dir / name
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            bg_type = classify_background_simple(img)
            brightness = compute_mean_brightness(img)
            self.bg_types[name] = bg_type
            self.bg_brightness[name] = brightness
            self.type_index[bg_type].append(name)
    
    def _analyze_parallel(self, names: List[str]):
        """멀티프로세싱으로 배경 유형과 밝기를 병렬 분석한다."""
        images_dir_str = str(self.images_dir)
        args_list = [(name, images_dir_str) for name in names]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(_classify_single_background, args_list, chunksize=32),
                total=len(args_list),
                desc=f"배경 유형+밝기 분석 (workers={self.num_workers})",
            ))
        
        for name, bg_type, brightness in results:
            if bg_type:  # 빈 문자열이면 로드 실패
                self.bg_types[name] = bg_type
                self.bg_brightness[name] = brightness
                self.type_index[bg_type].append(name)
    
    def _log_type_stats(self):
        """배경 유형별 통계와 밝기 분포를 로그에 출력한다."""
        total = sum(len(v) for v in self.type_index.values())
        logger.info(f"배경 유형 분석 결과: {total}장")
        for bg_type in ALL_BACKGROUND_TYPES:
            count = len(self.type_index[bg_type])
            pct = 100.0 * count / max(total, 1)
            logger.info(f"  {bg_type:20s}: {count:5d} ({pct:5.1f}%)")
        # 밝기 분포 통계
        if self.bg_brightness:
            brightness_vals = list(self.bg_brightness.values())
            logger.info(f"  밝기 분포: min={min(brightness_vals):.1f}, "
                        f"max={max(brightness_vals):.1f}, "
                        f"mean={sum(brightness_vals)/len(brightness_vals):.1f}")
    
    def get_compatible_background(
        self,
        defect_subtype: str,
        roi_x_center: Optional[int] = None,
        min_compatibility: float = 0.3,
        target_brightness: Optional[float] = None,
        brightness_tolerance: float = 30.0,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """
        호환성 + 밝기 매칭 기반으로 배경 이미지 파일명을 선택한다.
        
        밝기 매칭: target_brightness가 지정되면 해당 밝기와 ±tolerance 이내의
        배경만 후보로 사용한다. 후보가 부족하면(5개 미만) tolerance를 2배로 완화.
        그래도 부족하면 밝기 필터를 해제한다.
        
        Args:
            defect_subtype: 결함 하위 유형 (COMPATIBILITY_MATRIX 키)
            roi_x_center: ROI의 x 중심점 (미사용, 향후 위치 기반 매칭 확장용)
            min_compatibility: 최소 호환 점수
            target_brightness: 목표 밝기 (None이면 밝기 필터링 비활성)
            brightness_tolerance: 밝기 허용 오차 (±, 기본 30.0)
            rng: 난수 생성기 (None이면 모듈 random 사용)
            
        Returns:
            배경 이미지 파일명 또는 None
        """
        if rng is None:
            rng = random
        
        def _brightness_filter(names: List[str], tol: float) -> List[str]:
            """밝기 범위 내 후보 필터링."""
            if target_brightness is None or not self.bg_brightness:
                return names
            lo = target_brightness - tol
            hi = target_brightness + tol
            return [n for n in names
                    if lo <= self.bg_brightness.get(n, 128.0) <= hi]
        
        def _select_with_brightness(candidates: List[Tuple[str, float]]) -> Optional[str]:
            """
            호환성 가중 후보에서 밝기 필터링 후 선택.
            후보 부족 시 tolerance 완화 → 해제 순으로 폴백.
            """
            if not candidates:
                return None
            
            names_only = [name for name, _ in candidates]
            
            if target_brightness is not None and self.bg_brightness:
                # 1차: 기본 tolerance
                filtered = _brightness_filter(names_only, brightness_tolerance)
                if len(filtered) < 5:
                    # 2차: tolerance 2배 완화
                    filtered = _brightness_filter(names_only, brightness_tolerance * 2)
                if len(filtered) < 5:
                    # 3차: 밝기 필터 해제
                    filtered = names_only
                
                # 필터링된 이름 set으로 가중치 재구성
                filtered_set = set(filtered)
                filtered_candidates = [(n, w) for n, w in candidates
                                       if n in filtered_set]
                if filtered_candidates:
                    f_names, f_weights = zip(*filtered_candidates)
                    return rng.choices(f_names, weights=f_weights, k=1)[0]
            
            # 밝기 필터 없이 원래 가중 선택
            c_names, c_weights = zip(*candidates)
            return rng.choices(c_names, weights=c_weights, k=1)[0]
        
        # 호환 배경 유형 수집 (점수순 내림차순)
        compat_scores = COMPATIBILITY_MATRIX.get(defect_subtype, {})
        
        if not compat_scores:
            # 알 수 없는 defect_subtype → 모든 유형에서 밝기 매칭 후 랜덤 선택
            all_analyzed = [n for n in self.clean_names if n in self.bg_types]
            if all_analyzed:
                filtered = _brightness_filter(all_analyzed, brightness_tolerance)
                if len(filtered) < 5:
                    filtered = _brightness_filter(all_analyzed, brightness_tolerance * 2)
                if len(filtered) < 5:
                    filtered = all_analyzed
                return rng.choice(filtered)
            # 분석 안 된 경우 전체에서 랜덤
            return rng.choice(self.clean_names) if self.clean_names else None
        
        # 호환 점수 높은 순으로 정렬
        sorted_types = sorted(
            compat_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # 호환 유형별로 후보 수집
        candidates = []
        for bg_type, score in sorted_types:
            if score < min_compatibility:
                continue
            type_names = self.type_index.get(bg_type, [])
            if type_names:
                # 가중치: 호환 점수를 가중치로 사용
                candidates.extend([(name, score) for name in type_names])
        
        if candidates:
            return _select_with_brightness(candidates)
        
        # 호환 후보가 없으면 전체에서 밝기 매칭 랜덤
        all_analyzed = [n for n in self.clean_names if n in self.bg_types]
        if all_analyzed:
            filtered = _brightness_filter(all_analyzed, brightness_tolerance)
            if len(filtered) < 5:
                filtered = all_analyzed
            return rng.choice(filtered)
        return rng.choice(self.clean_names) if self.clean_names else None
    
    def get_random_background(
        self, rng: Optional[random.Random] = None
    ) -> Optional[str]:
        """배경 유형 무관하게 랜덤 선택."""
        if rng is None:
            rng = random
        return rng.choice(self.clean_names) if self.clean_names else None
    
    def get_brightness(self, name: str) -> float:
        """배경 이미지의 평균 밝기를 반환한다. 미분석 시 128.0."""
        return self.bg_brightness.get(name, 128.0)


# ============================================================================
# 메인 합성 로직
# ============================================================================

def load_roi_metadata(
    metadata_csv: Path,
) -> Dict[str, dict]:
    """
    packaged_roi_metadata.csv를 로드하고 sample_name으로 인덱싱한다.
    
    sample_name = "{image_id}_class{class_id}_region{region_id}"
    
    Returns:
        {sample_name: {roi_bbox, defect_bbox, background_type, defect_subtype, ...}}
    """
    df = pd.read_csv(metadata_csv)
    logger.info(f"ROI 메타데이터 로딩: {len(df)}행, 컬럼: {list(df.columns)}")
    
    lookup = {}
    # to_dict('records')가 iterrows()보다 2~5배 빠름
    has_defect_bbox = 'defect_bbox' in df.columns
    records = df.to_dict('records')
    
    for row in records:
        image_id = row['image_id']
        class_id = int(row['class_id'])
        region_id = int(row['region_id'])
        sample_name = f"{image_id}_class{class_id}_region{region_id}"
        
        # roi_bbox 파싱
        roi_bbox = parse_bbox_string(row['roi_bbox'])
        
        # defect_bbox 파싱 (존재하는 경우)
        defect_bbox = None
        if has_defect_bbox and pd.notna(row.get('defect_bbox')):
            try:
                defect_bbox = parse_bbox_string(row['defect_bbox'])
            except (ValueError, TypeError):
                pass
        
        entry = {
            'image_id': image_id,
            'class_id': class_id,
            'region_id': region_id,
            'roi_bbox': roi_bbox,
            'defect_bbox': defect_bbox,
            'background_type': row.get('background_type', 'unknown'),
            'defect_subtype': row.get('defect_subtype', 'unknown'),
            'suitability_score': float(row.get('suitability_score', 0.5)),
            'stability_score': float(row.get('stability_score', 0.5)),
        }
        lookup[sample_name] = entry
    
    logger.info(f"ROI 메타데이터 인덱싱 완료: {len(lookup)}개 sample_name")
    return lookup


def load_generation_summary(
    summary_json: Path,
) -> Tuple[dict, Dict[str, dict]]:
    """
    generation_summary.json을 로드하고 sample_name → result 매핑을 구축한다.
    
    Returns:
        (summary_dict, {sample_name: result_entry})
    """
    with open(summary_json) as f:
        summary = json.load(f)
    
    sample_map = {}
    for result in summary.get("results", []):
        sample_name = result.get("sample_name", "")
        sample_map[sample_name] = result
    
    logger.info(f"생성 요약 로딩: {summary.get('total_samples', 0)}개 샘플, "
                f"{summary.get('total_images', 0)}개 이미지")
    return summary, sample_map


def _compose_single_task(args: dict) -> Optional[dict]:
    """
    단일 합성 작업을 수행하는 워커 함수 (멀티프로세싱용).
    
    Args:
        args: 합성에 필요한 모든 인자를 담은 dict
        
    Returns:
        성공 시 메타데이터 dict, 실패 시 실패 사유를 담은 dict {'__fail__': reason}
    """
    img_path_str = args['img_path']
    hint_path_str = args['hint_path']
    bg_path_str = args['bg_path']
    roi_bbox = tuple(args['roi_bbox'])
    class_id = args['class_id']
    filename = args['filename']
    bg_name = args['bg_name']
    quality_score = args['quality_score']
    defect_subtype = args['defect_subtype']
    background_type = args['background_type']
    defect_bbox_original = args.get('defect_bbox_original')
    out_img_path_str = args['out_img_path']
    out_mask_path_str = args['out_mask_path']
    dilation_px = args['dilation_px']
    blend_mode = args['blend_mode']
    mask_threshold = args['mask_threshold']
    png_compression = args['png_compression']
    # Tier-1 증강 파라미터
    jitter_x = args.get('jitter_x', 0)
    scale_factor = args.get('scale_factor', 1.0)
    use_smooth_mask = args.get('use_smooth_mask', False)
    smooth_ksize = args.get('smooth_ksize', 21)
    smooth_sigma = args.get('smooth_sigma', 7.0)
    
    # 워커 프로세스에서 PoissonBlender 생성 (stateless이므로 매번 생성해도 무방)
    blender = PoissonBlender(
        dilation_px=dilation_px,
        blend_mode=blend_mode,
        mask_threshold=mask_threshold,
    )

    no_blend = args.get('no_blend', False)

    if no_blend:
        # ── 직접 paste 모드 (w/o Blending ablation) ──
        gen_img = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
        hint_img = cv2.imread(hint_path_str, cv2.IMREAD_COLOR)
        clean_bg = cv2.imread(bg_path_str, cv2.IMREAD_COLOR)

        if gen_img is None or hint_img is None or clean_bg is None:
            return {'__fail__': 'load', 'filename': filename}

        # 힌트 Red 채널에서 마스크 추출
        mask_512 = blender.extract_mask_from_hint(hint_img)
        if np.count_nonzero(mask_512) == 0:
            return {'__fail__': 'empty_mask', 'filename': filename}

        x1, y1, x2, y2 = roi_bbox
        roi_w = x2 - x1
        roi_h = y2 - y1

        # scale_factor 적용
        scaled_h = max(1, round(roi_h * scale_factor))
        scaled_w = max(1, round(roi_w * scale_factor))

        gen_resized = cv2.resize(gen_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_512, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        if np.count_nonzero(mask_resized) == 0:
            return {'__fail__': 'empty_mask_scaled', 'filename': filename}

        target_h, target_w = clean_bg.shape[:2]

        # jitter 적용
        x1_j = x1 + jitter_x
        x2_j = x1_j + scaled_w

        # target 경계 클리핑
        px1 = max(0, x1_j)
        py1 = max(0, y1)
        px2 = min(target_w, x2_j)
        py2 = min(target_h, y1 + scaled_h)

        if px2 <= px1 or py2 <= py1:
            return {'__fail__': 'out_of_bounds', 'filename': filename}

        sx1 = px1 - x1_j
        sy1 = 0
        sx2 = sx1 + (px2 - px1)
        sy2 = sy1 + (py2 - py1)

        mask_bool = mask_resized[sy1:sy2, sx1:sx2] > 127
        if not mask_bool.any():
            return {'__fail__': 'empty_mask_clipped', 'filename': filename}

        composited = clean_bg.copy()
        composited[py1:py2, px1:px2][mask_bool] = gen_resized[sy1:sy2, sx1:sx2][mask_bool]

        # 전체 크기 마스크 + YOLO bbox
        roi_bbox_jittered = (x1_j, y1, x2_j, y1 + scaled_h)
        full_mask = blender.generate_full_mask(mask_resized, roi_bbox_jittered, (target_h, target_w))
        bboxes, labels = blender.compute_yolo_bboxes(full_mask, class_id)

        if not bboxes:
            return {'__fail__': 'no_bbox', 'filename': filename}

        # 출력 저장
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        cv2.imwrite(out_img_path_str, composited, png_params)
        cv2.imwrite(out_mask_path_str, full_mask, png_params)

        out_img_name = Path(out_img_path_str).name
        out_mask_name = Path(out_mask_path_str).name

        entry = {
            "image_path": f"images/{out_img_name}",
            "class_id": class_id,
            "suitability_score": round(quality_score, 6),
            "mask_path": f"masks/{out_mask_name}",
            "bboxes": [[round(v, 6) for v in bbox] for bbox in bboxes],
            "labels": labels,
            "bbox_format": "yolo",
            "image_width": composited.shape[1],
            "image_height": composited.shape[0],
            "source_generated": filename,
            "source_background": bg_name,
            "blend_method": "direct_paste",
            "roi_bbox": list(roi_bbox),
            "jitter_x": jitter_x,
            "scale_factor": round(scale_factor, 6),
        }
        if defect_bbox_original:
            entry["defect_bbox_original"] = list(defect_bbox_original)
        if defect_subtype != 'unknown':
            entry["defect_subtype"] = defect_subtype
        if background_type != 'unknown':
            entry["background_type"] = background_type
        return entry

    result = blender.compose_from_paths(
        generated_path=img_path_str,
        hint_path=hint_path_str,
        clean_bg_path=bg_path_str,
        roi_bbox=roi_bbox,
        class_id=class_id,
        jitter_x=jitter_x,
        scale_factor=scale_factor,
        use_smooth_mask=use_smooth_mask,
        smooth_ksize=smooth_ksize,
        smooth_sigma=smooth_sigma,
    )

    if not result.success:
        return {'__fail__': 'blend', 'filename': filename}
    
    # 출력 저장 (PNG 압축 레벨 적용)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    cv2.imwrite(out_img_path_str, result.composited_image, png_params)
    cv2.imwrite(out_mask_path_str, result.full_mask, png_params)
    
    out_img_name = Path(out_img_path_str).name
    out_mask_name = Path(out_mask_path_str).name
    
    entry = {
        "image_path": f"images/{out_img_name}",
        "class_id": class_id,
        "suitability_score": round(quality_score, 6),
        "mask_path": f"masks/{out_mask_name}",
        "bboxes": [[round(v, 6) for v in bbox] for bbox in result.bboxes],
        "labels": result.labels,
        "bbox_format": "yolo",
        "image_width": result.composited_image.shape[1],
        "image_height": result.composited_image.shape[0],
        "source_generated": filename,
        "source_background": bg_name,
        "blend_method": result.blend_method,
        "roi_bbox": list(roi_bbox),
        "jitter_x": jitter_x,
        "scale_factor": round(scale_factor, 6),
    }
    
    # 합성 변형 인덱스 (compositions_per_roi > 1일 때 추적용)
    composition_variant = args.get('composition_variant', 0)
    if composition_variant > 0:
        entry["composition_variant"] = composition_variant
    
    if defect_bbox_original:
        entry["defect_bbox_original"] = list(defect_bbox_original)
    if defect_subtype != 'unknown':
        entry["defect_subtype"] = defect_subtype
    if background_type != 'unknown':
        entry["background_type"] = background_type
    
    return entry


def compose_all(
    generated_dir: Path,
    hint_dir: Path,
    metadata_csv: Path,
    summary_json: Path,
    clean_images_dir: Path,
    train_csv: Path,
    output_dir: Path,
    quality_json: Optional[Path] = None,
    dilation_px: int = 8,
    blend_mode: int = cv2.NORMAL_CLONE,
    mask_threshold: int = 127,
    seed: int = 42,
    max_backgrounds: int = 5000,
    default_quality_score: float = 0.5,
    num_workers: int = 0,
    bg_cache_path: Optional[Path] = None,
    png_compression: int = 1,
    jitter_range: int = 100,
    scale_min: float = 0.875,
    scale_max: float = 1.0,
    use_smooth_mask: bool = True,
    smooth_ksize: int = 21,
    smooth_sigma: float = 7.0,
    brightness_tolerance: float = 30.0,
    compositions_per_roi: int = 1,
    no_blend: bool = False,
):
    """
    전체 합성 파이프라인 실행.
    
    Args:
        generated_dir: 생성 이미지 디렉토리 (512x512 PNG)
        hint_dir: 힌트 이미지 디렉토리
        metadata_csv: packaged_roi_metadata.csv 경로
        summary_json: generation_summary.json 경로
        clean_images_dir: 원본 이미지 디렉토리 (1600x256)
        train_csv: train.csv 경로 (결함 이미지 식별용)
        output_dir: 출력 디렉토리 (casda_composed/)
        quality_json: 품질 점수 JSON (선택)
        dilation_px: 마스크 확장 픽셀 수
        blend_mode: cv2.NORMAL_CLONE 또는 cv2.MIXED_CLONE
        mask_threshold: 힌트 Red 채널 이진화 임계값
        seed: 랜덤 시드
        max_backgrounds: 배경 유형 분석할 최대 이미지 수
        default_quality_score: 품질 점수 없을 때 기본값
        num_workers: 합성 병렬 워커 수 (0이면 순차 처리)
        bg_cache_path: 배경 유형+밝기 캐시 JSON 경로 (None이면 캐시 미사용)
        png_compression: PNG 압축 레벨 (0-9, 낮을수록 빠름, 기본 1)
        jitter_range: x축 위치 랜덤 오프셋 최대값 (±N px, 0이면 비활성, 기본 100)
        scale_min: 다운스케일 최소 비율 (기본 0.875)
        scale_max: 다운스케일 최대 비율 (기본 1.0)
        use_smooth_mask: 마스크 경계 가우시안 블러 적용 여부 (기본 True)
        smooth_ksize: 가우시안 커널 크기 (홀수, 기본 21)
        smooth_sigma: 가우시안 시그마 (기본 7.0)
        brightness_tolerance: 배경 밝기 매칭 허용 오차 (±, 기본 30.0, 0이면 비활성)
        compositions_per_roi: 각 생성 이미지당 합성 변형 수 (기본 1).
            N>1이면 각 ROI에 대해 서로 다른 (배경, jitter, scale) 조합으로
            N개의 합성 이미지를 생성한다. GPU 비용 없이 pruning 풀을 N배 확대.
    """
    pipeline_start = time.time()
    rng = random.Random(seed)
    
    if no_blend:
        logger.info("⚠️  --no-blend 모드: Poisson Blending 없이 직접 paste (w/o Blending ablation)")

    # ── Tier-1 증강 옵션 로깅 ──
    augmentation_active = jitter_range > 0 or scale_min < 1.0 or use_smooth_mask
    if augmentation_active:
        logger.info("Tier-1 증강 옵션 활성화:")
        if jitter_range > 0:
            logger.info(f"  Positional Jittering: ±{jitter_range}px")
        if scale_min < 1.0:
            logger.info(f"  Multi-Scale: [{scale_min:.4f}, {scale_max:.4f}]")
        if use_smooth_mask:
            logger.info(f"  Smooth Mask: ksize={smooth_ksize}, sigma={smooth_sigma}")
    if compositions_per_roi > 1:
        logger.info(f"  Compositions per ROI: {compositions_per_roi} "
                    f"(pruning 풀 {compositions_per_roi}x 확대)")
    elif compositions_per_roi < 1:
        raise ValueError(
            f"compositions_per_roi는 1 이상이어야 합니다 (입력: {compositions_per_roi})"
        )
    if brightness_tolerance > 0:
        logger.info(f"밝기 매칭 활성화: tolerance=±{brightness_tolerance:.1f}")
    else:
        logger.info("밝기 매칭 비활성화")
    
    # ── Step 1: 메타데이터 로딩 ──
    logger.info("=" * 60)
    logger.info("Step 1: 메타데이터 로딩")
    logger.info("=" * 60)
    
    roi_lookup = load_roi_metadata(metadata_csv)
    summary, sample_name_map = load_generation_summary(summary_json)
    quality_map = build_quality_map(summary, quality_json)
    
    # ── Step 2: 생성 이미지 탐색 ──
    logger.info("=" * 60)
    logger.info("Step 2: 생성 이미지 탐색")
    logger.info("=" * 60)
    
    generated_images = sorted(generated_dir.glob("*.png"))
    logger.info(f"생성 이미지 발견: {len(generated_images)}장")
    
    if not generated_images:
        logger.error(f"생성 이미지 없음: {generated_dir}")
        sys.exit(1)
    
    # ── Step 3: 결함 없는 배경 이미지 풀 구축 ──
    logger.info("=" * 60)
    logger.info("Step 3: 결함 없는 배경 이미지 풀 구축")
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
    
    # ── Step 4: 출력 디렉토리 생성 ──
    out_img_dir = output_dir / "images"
    out_mask_dir = output_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Step 4.5: 생성 이미지 밝기 사전 계산 ──
    # 배경 밝기 매칭을 위해 모든 생성 이미지의 평균 밝기를 미리 계산한다.
    # Step 5 루프에서 매번 cv2.imread를 호출하지 않도록 분리.
    gen_brightness: Dict[str, float] = {}
    
    if brightness_tolerance > 0:
        logger.info("=" * 60)
        logger.info(f"Step 4.5: 생성 이미지 밝기 사전 계산 ({len(generated_images)}장)")
        logger.info("=" * 60)
        
        brightness_start = time.time()
        gen_paths_str = [str(p) for p in generated_images]
        
        if num_workers > 1 and len(generated_images) > 10:
            # 멀티프로세싱 병렬 밝기 측정
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                brightness_results = list(tqdm(
                    executor.map(_compute_gen_brightness, gen_paths_str, chunksize=32),
                    total=len(gen_paths_str),
                    desc=f"생성 이미지 밝기 측정 (workers={num_workers})",
                ))
        else:
            # 순차 밝기 측정
            brightness_results = [
                _compute_gen_brightness(p) 
                for p in tqdm(gen_paths_str, desc="생성 이미지 밝기 측정")
            ]
        
        for fname, bval in brightness_results:
            gen_brightness[fname] = bval
        
        brightness_elapsed = time.time() - brightness_start
        bvals = list(gen_brightness.values())
        logger.info(
            f"밝기 사전 계산 완료: {len(gen_brightness)}장, "
            f"{brightness_elapsed:.1f}초 소요, "
            f"밝기 범위 [{min(bvals):.0f} ~ {max(bvals):.0f}], "
            f"평균 {sum(bvals)/len(bvals):.1f}"
        )
    else:
        logger.info("밝기 매칭 비활성 (brightness_tolerance=0) — 밝기 사전 계산 건너뜀")
    
    # ── Step 5: 합성 작업 목록 사전 준비 ──
    # 메인 프로세스에서 메타데이터 조회 + 배경 선택을 수행하고,
    # 실제 합성(I/O + seamlessClone)은 워커로 분배한다.
    logger.info("=" * 60)
    logger.info("Step 5: 합성 작업 준비")
    logger.info("=" * 60)
    
    tasks = []
    stats = {
        'total': len(generated_images),
        'total_tasks': 0,  # compositions_per_roi 반영된 실제 task 수
        'compositions_per_roi': compositions_per_roi,
        'success': 0,
        'fail_no_roi_meta': 0,
        'fail_no_hint': 0,
        'fail_no_background': 0,
        'fail_blend': 0,
        'fail_class_parse': 0,
        'blend_methods': {},
        'class_counts': {},
    }
    
    for img_path in tqdm(generated_images, desc="합성 작업 준비"):
        filename = img_path.name
        sample_name = filename_to_sample_name(filename)
        
        # class_id 추출
        try:
            class_id = parse_class_id_from_filename(filename)
        except ValueError:
            stats['fail_class_parse'] += 1
            continue
        
        # ROI 메타데이터 조회
        roi_meta = roi_lookup.get(sample_name)
        if roi_meta is None:
            stats['fail_no_roi_meta'] += 1
            logger.debug(f"ROI 메타 없음: {sample_name}")
            continue
        
        roi_bbox = roi_meta['roi_bbox']
        defect_subtype = roi_meta.get('defect_subtype', 'unknown')
        background_type = roi_meta.get('background_type', 'unknown')
        
        # 힌트 이미지 경로 결정
        hint_filename = f"{sample_name}_hint.png"
        hint_path = hint_dir / hint_filename
        
        if not hint_path.exists():
            result_entry = sample_name_map.get(sample_name, {})
            hint_path_str = result_entry.get("hint_path", "")
            if hint_path_str:
                alt_name = Path(hint_path_str).name
                hint_path = hint_dir / alt_name
            
            if not hint_path.exists():
                stats['fail_no_hint'] += 1
                logger.debug(f"힌트 없음: {hint_filename}")
                continue
        
        # 호환 배경 이미지 선택 (밝기 매칭 포함)
        # Step 4.5에서 사전 계산된 밝기 dict에서 O(1) lookup
        target_brightness = gen_brightness.get(filename)  # None이면 밝기 필터 비활성
        
        roi_x_center = (roi_bbox[0] + roi_bbox[2]) // 2
        
        # 품질 점수 조회 (composition variant 간 공유)
        quality_score = get_quality_score(
            quality_map, filename, default=default_quality_score
        )
        
        # 각 생성 이미지에 대해 compositions_per_roi 개의 합성 변형 생성
        # 매 반복마다 다른 (배경, jitter_x, scale_factor) 조합 사용
        stem = filename.replace(".png", "")
        
        for comp_idx in range(compositions_per_roi):
            bg_name = bg_pool.get_compatible_background(
                defect_subtype=defect_subtype,
                roi_x_center=roi_x_center,
                target_brightness=target_brightness,
                brightness_tolerance=brightness_tolerance,
                rng=rng,
            )
            
            if bg_name is None:
                stats['fail_no_background'] += 1
                continue
            
            bg_path = clean_images_dir / bg_name
            
            # 출력 파일명: N=1이면 기존과 동일, N>1이면 _comp{idx} 접미사 추가
            if compositions_per_roi > 1:
                out_img_name = f"{stem}_comp{comp_idx}.png"
                out_mask_name = f"{stem}_comp{comp_idx}_mask.png"
            else:
                out_img_name = filename
                out_mask_name = filename.replace(".png", "_mask.png")
            
            # Tier-1 증강: 변형별 랜덤 jitter_x, scale_factor 생성
            img_jitter_x = rng.randint(-jitter_range, jitter_range) if jitter_range > 0 else 0
            img_scale_factor = rng.uniform(scale_min, scale_max) if scale_min < scale_max else scale_min
            
            task = {
                'img_path': str(img_path),
                'hint_path': str(hint_path),
                'bg_path': str(bg_path),
                'roi_bbox': list(roi_bbox),
                'class_id': class_id,
                'filename': filename,
                'bg_name': bg_name,
                'quality_score': quality_score,
                'defect_subtype': defect_subtype,
                'background_type': background_type,
                'defect_bbox_original': list(roi_meta['defect_bbox']) if roi_meta.get('defect_bbox') else None,
                'out_img_path': str(out_img_dir / out_img_name),
                'out_mask_path': str(out_mask_dir / out_mask_name),
                'dilation_px': dilation_px,
                'blend_mode': blend_mode,
                'mask_threshold': mask_threshold,
                'png_compression': png_compression,
                # Tier-1 증강 파라미터
                'jitter_x': img_jitter_x,
                'scale_factor': img_scale_factor,
                'use_smooth_mask': use_smooth_mask,
                'smooth_ksize': smooth_ksize,
                'smooth_sigma': smooth_sigma,
                # 합성 변형 인덱스 (metadata 추적용)
                'composition_variant': comp_idx,
                'no_blend': no_blend,
            }
            tasks.append(task)
    
    stats['total_tasks'] = len(tasks)
    
    if compositions_per_roi > 1:
        logger.info(f"합성 작업 준비 완료: {len(tasks)}개 "
                    f"({stats['total']}장 × {compositions_per_roi}변형, "
                    f"사전 필터링 실패 포함)")
    else:
        logger.info(f"합성 작업 준비 완료: {len(tasks)}개 (사전 필터링 실패: "
                    f"{stats['total'] - len(tasks) - stats['fail_class_parse']}개)")
    
    # ── Step 6: 합성 실행 (순차 또는 병렬) ──
    logger.info("=" * 60)
    blend_label = "DirectPaste" if no_blend else "PoissonBlending"
    logger.info(f"Step 6: {blend_label} 합성 시작 "
                f"(workers={num_workers if num_workers > 1 else 'sequential'}, "
                f"png_compression={png_compression})")
    logger.info("=" * 60)
    
    all_metadata = []
    compose_start = time.time()
    
    if num_workers > 1 and len(tasks) > 1:
        # 멀티프로세싱 병렬 합성
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            blend_tqdm = f"{blend_label} (workers={num_workers})"
            results = list(tqdm(
                executor.map(_compose_single_task, tasks, chunksize=4),
                total=len(tasks),
                desc=blend_tqdm,
            ))
        
        for result in results:
            if result is None:
                stats['fail_blend'] += 1
            elif '__fail__' in result:
                stats['fail_blend'] += 1
            else:
                all_metadata.append(result)
                stats['success'] += 1
                bm = result.get('blend_method', 'unknown')
                stats['blend_methods'][bm] = stats['blend_methods'].get(bm, 0) + 1
                cid = result['class_id']
                stats['class_counts'][cid] = stats['class_counts'].get(cid, 0) + 1
    else:
        # 순차 합성 (싱글 프로세스, 배경 이미지 LRU 캐시 활용)
        blender = PoissonBlender(
            dilation_px=dilation_px,
            blend_mode=blend_mode,
            mask_threshold=mask_threshold,
        )
        
        # 배경 이미지 LRU 캐시 (순차 모드에서만 유효)
        _bg_cache: Dict[str, np.ndarray] = {}
        _BG_CACHE_MAX = 128
        
        def _load_bg_cached(path_str: str) -> Optional[np.ndarray]:
            """배경 이미지를 LRU 캐시에서 로드. 없으면 디스크에서 읽고 캐시."""
            if path_str in _bg_cache:
                return _bg_cache[path_str]
            img = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if img is not None and len(_bg_cache) < _BG_CACHE_MAX:
                _bg_cache[path_str] = img
            return img
        
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        
        tqdm_label = "DirectPaste 합성" if no_blend else "Poisson 합성"
        for task in tqdm(tasks, desc=tqdm_label):
            # 이미지 로드 (배경만 캐시)
            generated = cv2.imread(task['img_path'], cv2.IMREAD_COLOR)
            if generated is None:
                stats['fail_blend'] += 1
                continue
            
            hint = cv2.imread(task['hint_path'], cv2.IMREAD_COLOR)
            if hint is None:
                stats['fail_blend'] += 1
                continue
            
            clean_bg = _load_bg_cached(task['bg_path'])
            if clean_bg is None:
                stats['fail_blend'] += 1
                continue
            
            if no_blend:
                # no_blend 모드: 워커 함수 재사용 (직접 paste)
                worker_result = _compose_single_task(task)
                if worker_result is None or '__fail__' in worker_result:
                    stats['fail_blend'] += 1
                    logger.debug(f"직접 paste 실패: {task['filename']}")
                    continue
                all_metadata.append(worker_result)
                stats['success'] += 1
                bm = worker_result.get('blend_method', 'direct_paste')
                stats['blend_methods'][bm] = stats['blend_methods'].get(bm, 0) + 1
                cid = worker_result['class_id']
                stats['class_counts'][cid] = stats['class_counts'].get(cid, 0) + 1
                continue

            result = blender.compose_single(
                generated, hint, clean_bg,
                tuple(task['roi_bbox']), task['class_id'],
                jitter_x=task['jitter_x'],
                scale_factor=task['scale_factor'],
                use_smooth_mask=task['use_smooth_mask'],
                smooth_ksize=task['smooth_ksize'],
                smooth_sigma=task['smooth_sigma'],
            )

            if not result.success:
                stats['fail_blend'] += 1
                logger.debug(f"합성 실패: {task['filename']} — {result.message}")
                continue

            # 출력 저장
            cv2.imwrite(task['out_img_path'], result.composited_image, png_params)
            cv2.imwrite(task['out_mask_path'], result.full_mask, png_params)

            out_img_name = Path(task['out_img_path']).name
            out_mask_name = Path(task['out_mask_path']).name

            entry = {
                "image_path": f"images/{out_img_name}",
                "class_id": task['class_id'],
                "suitability_score": round(task['quality_score'], 6),
                "mask_path": f"masks/{out_mask_name}",
                "bboxes": [[round(v, 6) for v in bbox] for bbox in result.bboxes],
                "labels": result.labels,
                "bbox_format": "yolo",
                "image_width": result.composited_image.shape[1],
                "image_height": result.composited_image.shape[0],
                "source_generated": task['filename'],
                "source_background": task['bg_name'],
                "blend_method": result.blend_method,
                "roi_bbox": task['roi_bbox'],
                "jitter_x": task['jitter_x'],
                "scale_factor": round(task['scale_factor'], 6),
            }
            
            # 합성 변형 인덱스 (compositions_per_roi > 1일 때 추적용)
            comp_variant = task.get('composition_variant', 0)
            if comp_variant > 0:
                entry["composition_variant"] = comp_variant
            
            if task.get('defect_bbox_original'):
                entry["defect_bbox_original"] = task['defect_bbox_original']
            if task['defect_subtype'] != 'unknown':
                entry["defect_subtype"] = task['defect_subtype']
            if task['background_type'] != 'unknown':
                entry["background_type"] = task['background_type']
            
            all_metadata.append(entry)
            
            stats['success'] += 1
            stats['blend_methods'][result.blend_method] = \
                stats['blend_methods'].get(result.blend_method, 0) + 1
            stats['class_counts'][task['class_id']] = \
                stats['class_counts'].get(task['class_id'], 0) + 1
    
    compose_elapsed = time.time() - compose_start
    
    # ── Step 7: metadata.json 저장 ──
    logger.info("=" * 60)
    logger.info("Step 7: 메타데이터 저장")
    logger.info("=" * 60)
    
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    # ── Step 8: 패키징 리포트 ──
    pipeline_elapsed = time.time() - pipeline_start
    
    report = {
        "source": {
            "generated_dir": str(generated_dir),
            "hint_dir": str(hint_dir),
            "metadata_csv": str(metadata_csv),
            "summary_json": str(summary_json),
            "quality_json": str(quality_json) if quality_json else None,
            "clean_images_dir": str(clean_images_dir),
            "train_csv": str(train_csv),
        },
        "parameters": {
            "dilation_px": dilation_px,
            "blend_mode": "NORMAL_CLONE" if blend_mode == cv2.NORMAL_CLONE
                          else "MIXED_CLONE",
            "mask_threshold": mask_threshold,
            "seed": seed,
            "max_backgrounds": max_backgrounds,
            "default_quality_score": default_quality_score,
            "num_workers": num_workers,
            "png_compression": png_compression,
            "jitter_range": jitter_range,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "use_smooth_mask": use_smooth_mask,
            "smooth_ksize": smooth_ksize,
            "smooth_sigma": smooth_sigma,
            "brightness_tolerance": brightness_tolerance,
            "compositions_per_roi": compositions_per_roi,
        },
        "statistics": {
            "total_generated": stats['total'],
            "total_tasks": stats['total_tasks'],
            "compositions_per_roi": compositions_per_roi,
            "success": stats['success'],
            "fail_no_roi_meta": stats['fail_no_roi_meta'],
            "fail_no_hint": stats['fail_no_hint'],
            "fail_no_background": stats['fail_no_background'],
            "fail_blend": stats['fail_blend'],
            "fail_class_parse": stats['fail_class_parse'],
            "success_rate": round(
                stats['success'] / max(stats['total_tasks'], 1) * 100, 1
            ),
            "blend_methods": stats['blend_methods'],
            "class_distribution": {
                str(k): v for k, v in sorted(stats['class_counts'].items())
            },
        },
        "performance": {
            "total_pipeline_sec": round(pipeline_elapsed, 1),
            "compose_sec": round(compose_elapsed, 1),
            "images_per_sec": round(
                stats['success'] / max(compose_elapsed, 0.001), 1
            ),
        },
        "output": {
            "output_dir": str(output_dir),
            "total_images": len(all_metadata),
            "total_bboxes": sum(
                len(m.get("bboxes", [])) for m in all_metadata
            ),
        },
    }
    
    report_path = output_dir / "composition_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # ── 결과 출력 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("CASDA Composed 합성 완료")
    logger.info("=" * 60)
    logger.info(f"  입력 생성 이미지: {stats['total']}장")
    if compositions_per_roi > 1:
        logger.info(f"  합성 변형 수: {compositions_per_roi}x → "
                    f"총 {stats['total_tasks']}개 task 생성")
    logger.info(f"  합성 성공: {stats['success']}장 "
                f"({stats['success']/max(stats['total_tasks'],1)*100:.1f}%)")
    logger.info(f"  실패 — ROI 메타 없음: {stats['fail_no_roi_meta']}")
    logger.info(f"  실패 — 힌트 없음: {stats['fail_no_hint']}")
    logger.info(f"  실패 — 배경 없음: {stats['fail_no_background']}")
    logger.info(f"  실패 — 블렌딩 실패: {stats['fail_blend']}")
    logger.info(f"  실패 — 클래스 파싱: {stats['fail_class_parse']}")
    logger.info(f"  블렌딩 방식: {stats['blend_methods']}")
    logger.info(f"  클래스 분포: {dict(sorted(stats['class_counts'].items()))}")
    logger.info(f"  총 bbox 수: {report['output']['total_bboxes']}")
    logger.info(f"  출력 디렉토리: {output_dir}")
    logger.info(f"  리포트: {report_path}")
    logger.info(f"  --- 성능 ---")
    logger.info(f"  전체 파이프라인: {pipeline_elapsed:.1f}초")
    logger.info(f"  합성 처리: {compose_elapsed:.1f}초 "
                f"({report['performance']['images_per_sec']} img/s)")
    
    # 품질 점수 통계
    if all_metadata:
        scores = [m['suitability_score'] for m in all_metadata]
        logger.info(f"  품질 점수: min={min(scores):.4f}, "
                    f"max={max(scores):.4f}, "
                    f"mean={sum(scores)/len(scores):.4f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compose CASDA images using Poisson Blending"
    )
    parser.add_argument(
        "--generated-dir", type=str, required=True,
        help="생성 이미지 디렉토리 경로 (512x512 PNG)",
    )
    parser.add_argument(
        "--hint-dir", type=str, required=True,
        help="힌트 이미지 디렉토리 경로",
    )
    parser.add_argument(
        "--metadata-csv", type=str, required=True,
        help="packaged_roi_metadata.csv 경로",
    )
    parser.add_argument(
        "--summary-json", type=str, required=True,
        help="generation_summary.json 경로",
    )
    parser.add_argument(
        "--clean-images-dir", type=str, required=True,
        help="원본 이미지 디렉토리 (train_images/, 1600x256)",
    )
    parser.add_argument(
        "--train-csv", type=str, required=True,
        help="train.csv 경로 (결함 이미지 식별용)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="출력 디렉토리 (casda_composed/ 생성)",
    )
    parser.add_argument(
        "--quality-json", type=str, default=None,
        help="품질 점수 JSON 파일 경로 (선택)",
    )
    parser.add_argument(
        "--dilation-px", type=int, default=8,
        help="마스크 확장 픽셀 수 (기본: 8, 이전 기본 15는 과도한 smoothing 유발)",
    )
    parser.add_argument(
        "--blend-mode", type=str, default="NORMAL_CLONE",
        choices=["NORMAL_CLONE", "MIXED_CLONE"],
        help="Poisson 블렌딩 모드 (기본: NORMAL_CLONE). "
             "MIXED_CLONE: 배경 텍스처가 강한 강재에서 배경 보존이 우수하며 "
             "FID 개선에 유리할 수 있음",
    )
    parser.add_argument(
        "--mask-threshold", type=int, default=127,
        help="힌트 Red 채널 이진화 임계값 (기본: 127)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    parser.add_argument(
        "--max-backgrounds", type=int, default=5000,
        help="배경 유형 분석할 최대 이미지 수 (기본: 5000)",
    )
    parser.add_argument(
        "--default-quality-score", type=float, default=0.5,
        help="품질 점수 없을 때 기본값 (기본: 0.5)",
    )
    # ── 성능 최적화 옵션 ──
    parser.add_argument(
        "--workers", type=int, default=0,
        help="병렬 처리 워커 수 (0=순차, 권장: CPU코어수, 기본: 0)",
    )
    parser.add_argument(
        "--bg-cache", type=str, default=None,
        help="배경 유형 캐시 JSON 경로 (지정 시 재실행에서 분석 건너뜀)",
    )
    parser.add_argument(
        "--png-compression", type=int, default=1,
        help="PNG 압축 레벨 0-9 (낮을수록 빠름, 기본: 1, OpenCV 기본: 3)",
    )
    # ── Tier-1 합성 품질 개선 옵션 (기본값 = 권장값 활성화) ──
    parser.add_argument(
        "--jitter-range", type=int, default=100,
        help="x축 위치 랜덤 오프셋 최대값 ±N px (0=비활성, 권장: 50~200, 기본: 100)",
    )
    parser.add_argument(
        "--scale-min", type=float, default=0.875,
        help="다운스케일 최소 비율 (권장: 0.875, 기본: 0.875)",
    )
    parser.add_argument(
        "--scale-max", type=float, default=1.0,
        help="다운스케일 최대 비율 (기본: 1.0)",
    )
    parser.add_argument(
        "--smooth-mask", action="store_true", default=True,
        help="마스크 경계 가우시안 블러 적용 (기본: 활성화, --no-smooth-mask로 비활성화)",
    )
    parser.add_argument(
        "--no-smooth-mask", dest="smooth_mask", action="store_false",
        help="마스크 경계 가우시안 블러 비활성화",
    )
    parser.add_argument(
        "--smooth-ksize", type=int, default=21,
        help="가우시안 커널 크기 (홀수, 기본: 21)",
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=7.0,
        help="가우시안 시그마 (기본: 7.0)",
    )
    # ── 밝기 매칭 옵션 ──
    parser.add_argument(
        "--brightness-tolerance", type=float, default=30.0,
        help="배경 밝기 매칭 허용 오차 ±N (0=비활성, 기본: 30.0, 권장: 25~40)",
    )
    # ── Pruning 풀 확대 옵션 ──
    parser.add_argument(
        "--compositions-per-roi", type=int, default=1,
        help="각 생성 이미지당 합성 변형 수 (기본: 1). "
             "N>1이면 각 ROI에 대해 서로 다른 (배경, jitter, scale) 조합으로 "
             "N개의 합성 이미지를 생성하여 pruning 풀을 N배 확대. GPU 비용 없음",
    )
    # ── Ablation Study 옵션 ──
    parser.add_argument(
        "--no-blend", action="store_true", default=False,
        help="Poisson Blending 없이 직접 paste (w/o Blending ablation study용). "
             "생성 이미지를 마스크 기반으로 배경에 직접 덮어씌움. "
             "출력 디렉토리를 casda_no_blend 등으로 변경 권장.",
    )

    args = parser.parse_args()
    
    # blend_mode 문자열 → OpenCV 상수 변환
    blend_mode = (
        cv2.NORMAL_CLONE if args.blend_mode == "NORMAL_CLONE"
        else cv2.MIXED_CLONE
    )
    
    # workers 자동 설정 (0이면 순차)
    num_workers = args.workers
    if num_workers < 0:
        cpu_count = os.cpu_count() or 4
        num_workers = max(1, cpu_count - 1)
        logger.info(f"워커 수 자동 설정: {num_workers} (CPU: {cpu_count})")
    
    compose_all(
        generated_dir=Path(args.generated_dir),
        hint_dir=Path(args.hint_dir),
        metadata_csv=Path(args.metadata_csv),
        summary_json=Path(args.summary_json),
        clean_images_dir=Path(args.clean_images_dir),
        train_csv=Path(args.train_csv),
        output_dir=Path(args.output_dir),
        quality_json=Path(args.quality_json) if args.quality_json else None,
        dilation_px=args.dilation_px,
        blend_mode=blend_mode,
        mask_threshold=args.mask_threshold,
        seed=args.seed,
        max_backgrounds=args.max_backgrounds,
        default_quality_score=args.default_quality_score,
        num_workers=num_workers,
        bg_cache_path=Path(args.bg_cache) if args.bg_cache else None,
        png_compression=args.png_compression,
        jitter_range=args.jitter_range,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        use_smooth_mask=args.smooth_mask,
        smooth_ksize=args.smooth_ksize,
        smooth_sigma=args.smooth_sigma,
        brightness_tolerance=args.brightness_tolerance,
        compositions_per_roi=args.compositions_per_roi,
        no_blend=args.no_blend,
    )


if __name__ == "__main__":
    main()
