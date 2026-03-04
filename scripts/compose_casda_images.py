#!/usr/bin/env python3
"""
Compose CASDA images using Poisson Blending.
ControlNet 생성 512x512 결함 ROI를 Poisson Blending으로 결함 없는 원본(1600x256)에 합성.

파이프라인:
  1. 메타데이터 로딩 (generation_summary.json + packaged_roi_metadata.csv)
  2. 결함 없는 배경 이미지 풀 구축 (train.csv에 없는 이미지)
  3. 배경 유형 분석 + 호환성 매칭
  4. PoissonBlender로 각 생성 이미지를 1600x256 배경에 합성
  5. 합성 이미지 + 전체 크기 마스크 + metadata.json 저장

출력:
  casda_composed/
  ├── images/          # 1600x256 합성 이미지 (.png)
  ├── masks/           # 1600x256 전체 크기 마스크 (.png)
  └── metadata.json    # 메타데이터 (YOLO bbox, suitability_score 포함)

Usage:
  python scripts/compose_casda_images.py \
    --generated-dir outputs/v5.1/test_results_v5.1/generated \
    --hint-dir data/processed/controlnet_dataset/hints \
    --metadata-csv data/processed/controlnet_dataset/packaged_roi_metadata.csv \
    --summary-json outputs/v5.1/test_results_v5.1/generation_summary.json \
    --clean-images-dir data/raw/train_images \
    --train-csv data/raw/train.csv \
    --output-dir data/augmented/casda_composed \
    --dilation-px 15 \
    --blend-mode NORMAL_CLONE

  # With quality scores (for suitability_score in metadata):
  python scripts/compose_casda_images.py \
    --generated-dir outputs/v5.1/test_results_v5.1/generated \
    --hint-dir data/processed/controlnet_dataset/hints \
    --metadata-csv data/processed/controlnet_dataset/packaged_roi_metadata.csv \
    --summary-json outputs/v5.1/test_results_v5.1/generation_summary.json \
    --quality-json outputs/v5.1/test_results_v5.1/quality_scores.json \
    --clean-images-dir data/raw/train_images \
    --train-csv data/raw/train.csv \
    --output-dir data/augmented/casda_composed
"""

import argparse
import ast
import json
import logging
import random
import re
import sys
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
    1600x256 이미지의 중앙 256x256 영역에서 배경 유형을 간단히 분류.
    
    BackgroundCharacterizer의 전체 파이프라인을 사용하지 않고,
    엣지 방향성과 분산으로 빠르게 분류한다.
    
    Returns:
        'smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern'
    """
    h, w = image.shape[:2]
    
    # 중앙 256x256 패치 추출
    cx = w // 2
    x1 = max(0, cx - 128)
    x2 = min(w, x1 + 256)
    patch = image[:, x1:x2]
    
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
    """
    
    def __init__(
        self,
        clean_image_names: List[str],
        images_dir: Path,
        cache_bg_types: bool = True,
        max_analyze: int = 5000,
    ):
        """
        Args:
            clean_image_names: 결함 없는 이미지 파일명 목록
            images_dir: 이미지 디렉토리 경로
            cache_bg_types: 배경 유형을 캐싱할지 여부
            max_analyze: 배경 유형 분석할 최대 이미지 수
        """
        self.images_dir = images_dir
        self.clean_names = clean_image_names
        
        # 배경 유형별 인덱스: {bg_type: [filename, ...]}
        self.type_index: Dict[str, List[str]] = {t: [] for t in ALL_BACKGROUND_TYPES}
        self.bg_types: Dict[str, str] = {}  # filename → bg_type
        
        if cache_bg_types:
            self._analyze_backgrounds(max_analyze)
    
    def _analyze_backgrounds(self, max_analyze: int):
        """배경 이미지들의 유형을 분석하여 인덱스 구축."""
        names_to_analyze = self.clean_names[:max_analyze]
        logger.info(f"배경 유형 분석 시작: {len(names_to_analyze)}장...")
        
        for name in tqdm(names_to_analyze, desc="배경 유형 분석"):
            img_path = self.images_dir / name
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            bg_type = classify_background_simple(img)
            self.bg_types[name] = bg_type
            self.type_index[bg_type].append(name)
        
        # 분석 통계 출력
        total = sum(len(v) for v in self.type_index.values())
        logger.info(f"배경 유형 분석 완료: {total}장")
        for bg_type in ALL_BACKGROUND_TYPES:
            count = len(self.type_index[bg_type])
            pct = 100.0 * count / max(total, 1)
            logger.info(f"  {bg_type:20s}: {count:5d} ({pct:5.1f}%)")
    
    def get_compatible_background(
        self,
        defect_subtype: str,
        roi_x_center: Optional[int] = None,
        min_compatibility: float = 0.3,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """
        호환성 기반으로 배경 이미지 파일명을 선택한다.
        
        Args:
            defect_subtype: 결함 하위 유형 (COMPATIBILITY_MATRIX 키)
            roi_x_center: ROI의 x 중심점 (미사용, 향후 위치 기반 매칭 확장용)
            min_compatibility: 최소 호환 점수
            rng: 난수 생성기 (None이면 모듈 random 사용)
            
        Returns:
            배경 이미지 파일명 또는 None
        """
        if rng is None:
            rng = random
        
        # 호환 배경 유형 수집 (점수순 내림차순)
        compat_scores = COMPATIBILITY_MATRIX.get(defect_subtype, {})
        
        if not compat_scores:
            # 알 수 없는 defect_subtype → 모든 유형에서 랜덤 선택
            all_analyzed = [n for n in self.clean_names if n in self.bg_types]
            if all_analyzed:
                return rng.choice(all_analyzed)
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
            # 가중 랜덤 선택
            names, weights = zip(*candidates)
            return rng.choices(names, weights=weights, k=1)[0]
        
        # 호환 후보가 없으면 전체에서 랜덤
        all_analyzed = [n for n in self.clean_names if n in self.bg_types]
        if all_analyzed:
            return rng.choice(all_analyzed)
        return rng.choice(self.clean_names) if self.clean_names else None
    
    def get_random_background(
        self, rng: Optional[random.Random] = None
    ) -> Optional[str]:
        """배경 유형 무관하게 랜덤 선택."""
        if rng is None:
            rng = random
        return rng.choice(self.clean_names) if self.clean_names else None


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
    for _, row in df.iterrows():
        image_id = row['image_id']
        class_id = int(row['class_id'])
        region_id = int(row['region_id'])
        sample_name = f"{image_id}_class{class_id}_region{region_id}"
        
        # roi_bbox 파싱
        roi_bbox = parse_bbox_string(row['roi_bbox'])
        
        # defect_bbox 파싱 (존재하는 경우)
        defect_bbox = None
        if 'defect_bbox' in row and pd.notna(row['defect_bbox']):
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


def compose_all(
    generated_dir: Path,
    hint_dir: Path,
    metadata_csv: Path,
    summary_json: Path,
    clean_images_dir: Path,
    train_csv: Path,
    output_dir: Path,
    quality_json: Optional[Path] = None,
    dilation_px: int = 15,
    blend_mode: int = cv2.NORMAL_CLONE,
    mask_threshold: int = 127,
    seed: int = 42,
    max_backgrounds: int = 5000,
    default_quality_score: float = 0.5,
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
    """
    rng = random.Random(seed)
    
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
    )
    
    # ── Step 4: PoissonBlender 초기화 ──
    blender = PoissonBlender(
        dilation_px=dilation_px,
        blend_mode=blend_mode,
        mask_threshold=mask_threshold,
    )
    
    # ── Step 5: 출력 디렉토리 생성 ──
    out_img_dir = output_dir / "images"
    out_mask_dir = output_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Step 6: 합성 처리 ──
    logger.info("=" * 60)
    logger.info("Step 6: Poisson Blending 합성 시작")
    logger.info("=" * 60)
    
    all_metadata = []
    stats = {
        'total': len(generated_images),
        'success': 0,
        'fail_no_roi_meta': 0,
        'fail_no_hint': 0,
        'fail_no_background': 0,
        'fail_blend': 0,
        'fail_class_parse': 0,
        'blend_methods': {},
        'class_counts': {},
    }
    
    for img_path in tqdm(generated_images, desc="Poisson 합성"):
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
            # generation_summary에서 힌트 경로 폴백
            result_entry = sample_name_map.get(sample_name, {})
            hint_path_str = result_entry.get("hint_path", "")
            if hint_path_str:
                alt_name = Path(hint_path_str).name
                hint_path = hint_dir / alt_name
            
            if not hint_path.exists():
                stats['fail_no_hint'] += 1
                logger.debug(f"힌트 없음: {hint_filename}")
                continue
        
        # 호환 배경 이미지 선택
        roi_x_center = (roi_bbox[0] + roi_bbox[2]) // 2
        bg_name = bg_pool.get_compatible_background(
            defect_subtype=defect_subtype,
            roi_x_center=roi_x_center,
            rng=rng,
        )
        
        if bg_name is None:
            stats['fail_no_background'] += 1
            continue
        
        bg_path = clean_images_dir / bg_name
        
        # Poisson 합성 실행
        result = blender.compose_from_paths(
            generated_path=str(img_path),
            hint_path=str(hint_path),
            clean_bg_path=str(bg_path),
            roi_bbox=roi_bbox,
            class_id=class_id,
        )
        
        if not result.success:
            stats['fail_blend'] += 1
            logger.debug(f"합성 실패: {filename} — {result.message}")
            continue
        
        # ── 출력 저장 ──
        # 파일명 구성: 원본 생성 파일명 유지 (추적 용이성)
        out_img_name = filename
        out_mask_name = filename.replace(".png", "_mask.png")
        
        out_img_path = out_img_dir / out_img_name
        out_mask_path = out_mask_dir / out_mask_name
        
        cv2.imwrite(str(out_img_path), result.composited_image)
        cv2.imwrite(str(out_mask_path), result.full_mask)
        
        # 품질 점수 조회
        quality_score = get_quality_score(
            quality_map, filename, default=default_quality_score
        )
        
        # 메타데이터 엔트리 구성
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
        }
        
        # 선택적 메타데이터
        if roi_meta.get('defect_bbox'):
            entry["defect_bbox_original"] = list(roi_meta['defect_bbox'])
        if defect_subtype != 'unknown':
            entry["defect_subtype"] = defect_subtype
        if background_type != 'unknown':
            entry["background_type"] = background_type
        
        all_metadata.append(entry)
        
        # 통계 갱신
        stats['success'] += 1
        stats['blend_methods'][result.blend_method] = \
            stats['blend_methods'].get(result.blend_method, 0) + 1
        stats['class_counts'][class_id] = \
            stats['class_counts'].get(class_id, 0) + 1
    
    # ── Step 7: metadata.json 저장 ──
    logger.info("=" * 60)
    logger.info("Step 7: 메타데이터 저장")
    logger.info("=" * 60)
    
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    # ── Step 8: 패키징 리포트 ──
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
        },
        "statistics": {
            "total_generated": stats['total'],
            "success": stats['success'],
            "fail_no_roi_meta": stats['fail_no_roi_meta'],
            "fail_no_hint": stats['fail_no_hint'],
            "fail_no_background": stats['fail_no_background'],
            "fail_blend": stats['fail_blend'],
            "fail_class_parse": stats['fail_class_parse'],
            "success_rate": round(
                stats['success'] / max(stats['total'], 1) * 100, 1
            ),
            "blend_methods": stats['blend_methods'],
            "class_distribution": {
                str(k): v for k, v in sorted(stats['class_counts'].items())
            },
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
    logger.info(f"  합성 성공: {stats['success']}장 "
                f"({stats['success']/max(stats['total'],1)*100:.1f}%)")
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
        "--dilation-px", type=int, default=15,
        help="마스크 확장 픽셀 수 (기본: 15)",
    )
    parser.add_argument(
        "--blend-mode", type=str, default="NORMAL_CLONE",
        choices=["NORMAL_CLONE", "MIXED_CLONE"],
        help="Poisson 블렌딩 모드 (기본: NORMAL_CLONE)",
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
    
    args = parser.parse_args()
    
    # blend_mode 문자열 → OpenCV 상수 변환
    blend_mode = (
        cv2.NORMAL_CLONE if args.blend_mode == "NORMAL_CLONE"
        else cv2.MIXED_CLONE
    )
    
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
    )


if __name__ == "__main__":
    main()
