"""
ControlNet Dataset Packaging Module

This module packages the processed ROI data into ControlNet training format.
Creates:
- Multi-channel hint images
- train.jsonl with image paths and prompts
- Organized directory structure for training

Output format matches standard ControlNet training requirements.
"""
import ast
import json
import os
import pandas as pd
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .hint_generator import HintImageGenerator
from .prompt_generator import PromptGenerator
from ..utils.rle_utils import get_all_masks_for_image


# ── 병렬 워커 함수 (ProcessPoolExecutor용 — 모듈 레벨 정의 필수) ──

def _package_single_roi_worker(args_tuple):
    """
    단일 ROI를 패키징하는 워커 함수.

    ProcessPoolExecutor에서 pickle 직렬화가 가능하도록
    모듈 레벨에 정의. HintImageGenerator와 PromptGenerator를
    워커 내부에서 재생성한다.

    Args:
        args_tuple: (roi_dict, output_dir_str, create_hints, prompt_style)

    Returns:
        Updated roi_dict, 또는 None (실패 시)
    """
    roi_dict, output_dir_str, create_hints, prompt_style = args_tuple
    output_dir = Path(output_dir_str)

    # ROI 이미지 로드
    roi_image_path = Path(roi_dict['roi_image_path'])
    if not roi_image_path.exists():
        return None
    roi_image = cv2.imread(str(roi_image_path))
    if roi_image is None:
        return None
    roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

    # ROI 마스크 로드
    roi_mask_path = Path(roi_dict['roi_mask_path'])
    if not roi_mask_path.exists():
        return None
    roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        return None
    roi_mask = (roi_mask > 0).astype(np.uint8)

    # 워커 내부에서 generator 재생성
    hint_gen = HintImageGenerator()
    prompt_gen = PromptGenerator(style=prompt_style)

    if create_hints:
        # hint 이미지 생성
        hint_image = hint_gen.generate_hint_image(
            roi_image=roi_image_rgb,
            roi_mask=roi_mask,
            defect_metrics=roi_dict,
            background_type=roi_dict.get('background_type', 'smooth'),
            stability_score=roi_dict.get('stability_score', 0.5),
        )

        hint_dir = output_dir / 'hints'
        hint_dir.mkdir(parents=True, exist_ok=True)

        image_id = roi_dict['image_id']
        class_id = roi_dict['class_id']
        region_id = roi_dict['region_id']
        hint_filename = f"{image_id}_class{class_id}_region{region_id}_hint.png"
        hint_path = hint_dir / hint_filename

        hint_gen.save_hint_image(hint_image, hint_path)
        roi_dict['hint_path'] = str(hint_path)

    # prompt 생성
    prompt = prompt_gen.generate_prompt(
        defect_subtype=roi_dict.get('defect_subtype', 'general'),
        background_type=roi_dict.get('background_type', 'smooth'),
        class_id=roi_dict.get('class_id', 1),
        stability_score=roi_dict.get('stability_score', 0.5),
        defect_metrics=roi_dict,
        suitability_score=roi_dict.get('suitability_score', 0.5),
    )
    roi_dict['prompt'] = prompt
    roi_dict['negative_prompt'] = prompt_gen.generate_negative_prompt()

    return roi_dict


class ControlNetDatasetPackager:
    """
    Packages ROI data for ControlNet training.
    """
    
    def __init__(self, hint_generator: Optional[HintImageGenerator] = None,
                 prompt_generator: Optional[PromptGenerator] = None,
                 prompt_style: str = 'detailed'):
        """
        Initialize dataset packager.
        
        Args:
            hint_generator: HintImageGenerator instance
            prompt_generator: PromptGenerator instance
            prompt_style: Style for prompts ('simple', 'detailed', 'technical')
        """
        self.hint_generator = hint_generator or HintImageGenerator()
        self.prompt_generator = prompt_generator or PromptGenerator(style=prompt_style)
    
    def _edge_filter(self, df: pd.DataFrame,
                     edge_margin_x: float = 0.1,
                     edge_margin_y: float = 0.05,
                     rare_class_margin: float = 0.02,
                     rare_class_threshold: int = 50,
                     class_margin_overrides: Optional[Dict[int, Dict[str, float]]] = None,
                     ) -> pd.DataFrame:
        """
        ROI 경계에 너무 가까운 결함을 가진 샘플을 제외합니다.

        DatasetValidator.visual_check_sample()에서 경고만 출력하던
        edge proximity 검사를 패키징 시점에 실제 제외 로직으로 적용합니다.

        v5: optimize_roi_position이 adaptive sizing을 사용하여
        roi_bbox != defect_bbox가 정상적으로 됩니다 (256x256 패치).
        v4에서는 512x512 ROI가 256px 높이 이미지에 맞지 않아
        roi_bbox == defect_bbox로 fallback되었습니다.

        v5.1 개선사항:
        - 좌우(x)/상하(y) 마진을 분리. Severstal 이미지(1600x256)에서
          ROI 높이=이미지 높이=256이므로 상하 마진은 구조적으로 확보가 어려움.
          좌우 10%, 상하 5%로 차등 적용하여 제외율을 67% → ~35%로 감소.
        - 희소 클래스 보호: 전체 가용 샘플이 rare_class_threshold 이하인
          클래스는 마진을 rare_class_margin(2%)으로 더 완화하여
          학습 데이터 최소 확보를 보장 (v5에서 Class 2가 10개로 감소한 문제).

        B2 개선사항 (v5.4):
        - class_margin_overrides: 클래스별 마진 개별 지정.
          Class 4 결함은 Severstal 이미지 전체 높이(256px)를 차지하는 것이
          형태학적으로 정상이므로, Y-margin을 0으로 설정하여 불필요한 제외를 방지.
          예: {4: {'x': 0.05, 'y': 0.0}}

        Args:
            df: ROI metadata DataFrame (roi_bbox, defect_bbox 컬럼 필요)
            edge_margin_x: 좌우 마진 비율 (기본 0.1 = 10%, ROI 너비 기준)
            edge_margin_y: 상하 마진 비율 (기본 0.05 = 5%, ROI 높이 기준)
            rare_class_margin: 희소 클래스에 적용할 완화 마진 (기본 0.02 = 2%)
            rare_class_threshold: 이 수 이하인 클래스를 희소로 간주 (기본 50)
            class_margin_overrides: 클래스별 마진 오버라이드 dict.
                키: class_id (1-based, train.csv ClassId와 동일)
                값: {'x': float, 'y': float} (미지정 키는 기본 마진 사용)
                예: {4: {'x': 0.05, 'y': 0.0}}  → Class 4는 Y 마진 없음

        Returns:
            edge-flagged 샘플이 제거된 DataFrame
        """
        if 'roi_bbox' not in df.columns or 'defect_bbox' not in df.columns:
            print("Edge filter: roi_bbox/defect_bbox columns not found, skipping")
            return df

        # 희소 클래스 식별
        rare_classes = set()
        if 'class_id' in df.columns:
            class_counts = df['class_id'].value_counts()
            rare_classes = set(
                class_counts[class_counts <= rare_class_threshold].index
            )
            if rare_classes:
                print(f"  Rare classes (n<={rare_class_threshold}): "
                      f"{dict(class_counts[class_counts <= rare_class_threshold])} "
                      f"-> edge margin relaxed to {rare_class_margin}")

        # B2: 클래스별 마진 오버라이드 로그
        if class_margin_overrides:
            print(f"  Class margin overrides: {class_margin_overrides}")

        original_len = len(df)
        exclude_indices = []
        skipped_identical = 0

        for idx, row in df.iterrows():
            roi_bbox = row['roi_bbox']
            defect_bbox = row['defect_bbox']

            # Convert string tuples if needed (안전한 리터럴 파싱)
            if isinstance(roi_bbox, str):
                roi_bbox = ast.literal_eval(roi_bbox)
            if isinstance(defect_bbox, str):
                defect_bbox = ast.literal_eval(defect_bbox)

            # roi_bbox == defect_bbox이면 ROI 최적화 실패로 결함 bbox를
            # 그대로 사용한 것이므로 edge 검사가 무의미 (margin 항상 0).
            # 이 경우 검사를 건너뛰고 샘플을 유지합니다.
            if tuple(roi_bbox) == tuple(defect_bbox):
                skipped_identical += 1
                continue

            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
            def_x1, def_y1, def_x2, def_y2 = defect_bbox

            roi_width = roi_x2 - roi_x1
            roi_height = roi_y2 - roi_y1

            # Skip if dimensions are zero to avoid division issues
            if roi_width <= 0 or roi_height <= 0:
                exclude_indices.append(idx)
                continue

            # 마진 결정 우선순위: class_margin_overrides > rare_classes > 기본값
            class_id = row.get('class_id', None)
            if class_margin_overrides and class_id in class_margin_overrides:
                # B2: 클래스별 개별 마진 (예: Class 4 → y=0.0)
                overrides = class_margin_overrides[class_id]
                mx = overrides.get('x', edge_margin_x)
                my = overrides.get('y', edge_margin_y)
            elif class_id in rare_classes:
                mx = rare_class_margin
                my = rare_class_margin
            else:
                mx = edge_margin_x
                my = edge_margin_y

            edge_issues = []
            if (def_x1 - roi_x1) < roi_width * mx:
                edge_issues.append("left")
            if (roi_x2 - def_x2) < roi_width * mx:
                edge_issues.append("right")
            if (def_y1 - roi_y1) < roi_height * my:
                edge_issues.append("top")
            if (roi_y2 - def_y2) < roi_height * my:
                edge_issues.append("bottom")

            if edge_issues:
                exclude_indices.append(idx)
                image_id = row.get('image_id', 'unknown')
                cls_id = row.get('class_id', '?')
                print(f"  Edge filter excluded: {image_id} (Class {cls_id}) "
                      f"- too close to {', '.join(edge_issues)} edge")

        df = df.drop(index=exclude_indices).reset_index(drop=True)
        removed = original_len - len(df)
        print(f"Edge filter: {original_len} -> {len(df)} "
              f"({removed} removed, {removed/max(original_len,1)*100:.1f}%)")
        print(f"  Margins: x={edge_margin_x}, y={edge_margin_y}, "
              f"rare={rare_class_margin}"
              f"{' (+ class overrides active)' if class_margin_overrides else ''}")
        if skipped_identical > 0:
            print(f"  ({skipped_identical} samples with roi_bbox==defect_bbox, "
                  f"edge check skipped)")

        return df

    def _quality_filter(self, df: pd.DataFrame,
                        min_area: int = 100,
                        min_stability: float = 0.3,
                        min_matching: float = 0.5,
                        allowed_recommendations: Optional[List[str]] = None,
                        ) -> pd.DataFrame:
        """
        품질 기준으로 ROI를 필터링합니다.

        v2에서 50개 샘플만 사용하여 overfitting이 발생했으므로,
        v3에서는 500개 샘플을 목표로 하되 품질 기준을 충족하는 ROI만 사용합니다.

        Args:
            df: ROI metadata DataFrame
            min_area: 최소 결함 영역 (px). 너무 작은 결함은 hint가 불명확함.
            min_stability: 최소 stability_score. 낮으면 마스크 품질이 불안정.
            min_matching: 최소 matching_score. 낮으면 hint-target 불일치 가능.
            allowed_recommendations: 허용할 recommendation 값 목록.
                기본값: ['suitable', 'acceptable']

        Returns:
            필터링된 DataFrame
        """
        if allowed_recommendations is None:
            allowed_recommendations = ['suitable', 'acceptable']

        original_len = len(df)

        # 1. recommendation 필터
        if 'recommendation' in df.columns:
            df = df[df['recommendation'].isin(allowed_recommendations)]

        # 2. area 필터
        if 'area' in df.columns:
            df = df[df['area'] >= min_area]

        # 3. stability_score 필터
        if 'stability_score' in df.columns:
            df = df[df['stability_score'] >= min_stability]

        # 4. matching_score 필터
        if 'matching_score' in df.columns:
            df = df[df['matching_score'] >= min_matching]

        filtered_len = len(df)
        removed = original_len - filtered_len
        print(f"Quality filter: {original_len} -> {filtered_len} "
              f"({removed} removed, {removed/max(original_len,1)*100:.1f}%)")
        print(f"  Criteria: area>={min_area}, stability>={min_stability}, "
              f"matching>={min_matching}, recommendation in {allowed_recommendations}")

        if 'class_id' in df.columns:
            class_dist = df['class_id'].value_counts().sort_index()
            print(f"  Post-filter class distribution: {dict(class_dist)}")

        return df.reset_index(drop=True)

    def _stratified_sample(self, df: pd.DataFrame, n_samples: int,
                           per_class_cap: Optional[int] = None,
                           rare_class_threshold_count: int = 200) -> pd.DataFrame:
        """
        클래스별 샘플링 (품질 우선 + 다양성 고려).
        
        v5.2 개선: class-aware capping 모드 추가.
        - per_class_cap이 지정되면 class-aware capping 모드 사용:
          * 희소 클래스 (count <= rare_class_threshold_count): 전수 포함
          * 풍부한 클래스: min(count, per_class_cap) 만큼 다양성 샘플링
          * n_samples 파라미터는 무시됨 (cap 기반으로 자동 결정)
        - per_class_cap이 None이면 기존 균등 분배 로직 사용 (v3~v5.1 호환)
        
        Args:
            df: ROI metadata DataFrame (class_id 컬럼 필요)
            n_samples: 추출할 총 샘플 수 (균등 분배 모드에서만 사용)
            per_class_cap: 풍부한 클래스의 최대 샘플 수 (v5.2 class-aware capping)
            rare_class_threshold_count: 이 수 이하인 클래스를 희소로 간주하여
                전수 포함 (기본 200)
            
        Returns:
            샘플링된 DataFrame
        """
        if 'class_id' not in df.columns:
            # class_id가 없으면 suitability_score 기반 상위 선택
            if 'suitability_score' in df.columns:
                return df.nlargest(min(n_samples, len(df)), 'suitability_score')
            return df.sample(n=min(n_samples, len(df)), random_state=42)
        
        classes = sorted(df['class_id'].unique())
        has_suitability = 'suitability_score' in df.columns
        has_subtype = 'defect_subtype' in df.columns
        has_bg = 'background_type' in df.columns
        
        if per_class_cap is not None:
            # ===== v5.2 Class-aware capping 모드 =====
            # 희소 클래스는 전수 포함, 풍부한 클래스는 cap 적용
            sampled_parts = []
            
            print(f"  Class-aware capping mode: per_class_cap={per_class_cap}, "
                  f"rare_threshold={rare_class_threshold_count}")
            
            for cls in classes:
                cls_df = df[df['class_id'] == cls].copy()
                cls_count = len(cls_df)
                
                if cls_count <= rare_class_threshold_count:
                    # 희소 클래스: 전수 포함
                    sampled_parts.append(cls_df)
                    print(f"  Class {cls}: {cls_count} samples (RARE -> all included)")
                elif cls_count <= per_class_cap:
                    # cap 미만: 전수 포함
                    sampled_parts.append(cls_df)
                    print(f"  Class {cls}: {cls_count} samples (under cap -> all included)")
                else:
                    # 풍부한 클래스: cap까지 다양성 샘플링
                    selected = self._diverse_select(
                        cls_df, per_class_cap,
                        has_suitability, has_subtype, has_bg
                    )
                    sampled_parts.append(selected)
                    print(f"  Class {cls}: {cls_count} -> {per_class_cap} samples "
                          f"(capped, diversity-selected)")
            
            result = pd.concat(sampled_parts).reset_index(drop=True)
            print(f"  Total after capping: {len(result)} samples")
        else:
            # ===== 기존 균등 분배 모드 (v3~v5.1 호환) =====
            n_classes = len(classes)
            per_class = n_samples // n_classes
            remainder = n_samples % n_classes
            
            sampled_parts = []
            deficit = 0
            
            for i, cls in enumerate(classes):
                cls_df = df[df['class_id'] == cls].copy()
                target = per_class + (1 if i < remainder else 0)
                
                if len(cls_df) <= target:
                    sampled_parts.append(cls_df)
                    deficit += target - len(cls_df)
                else:
                    selected = self._diverse_select(cls_df, target,
                                                    has_suitability, has_subtype, has_bg)
                    sampled_parts.append(selected)
            
            # 부족분 재분배
            if deficit > 0:
                already_sampled_idx = pd.concat(sampled_parts).index
                remaining_df = df[~df.index.isin(already_sampled_idx)]
                
                if len(remaining_df) > 0:
                    if has_suitability:
                        additional = remaining_df.nlargest(
                            min(deficit, len(remaining_df)), 'suitability_score'
                        )
                    else:
                        additional = remaining_df.sample(
                            n=min(deficit, len(remaining_df)), random_state=42
                        )
                    sampled_parts.append(additional)
            
            result = pd.concat(sampled_parts).reset_index(drop=True)
        
        # 다양성 통계 출력
        if has_subtype:
            subtype_counts = result['defect_subtype'].value_counts()
            print(f"  Defect subtype diversity: {len(subtype_counts)} types")
        if has_bg:
            bg_counts = result['background_type'].value_counts()
            print(f"  Background type diversity: {len(bg_counts)} types")
        
        # 클래스 분포 출력
        class_dist = result['class_id'].value_counts().sort_index()
        print(f"  Final class distribution: {dict(class_dist)}")
        
        return result
    
    def _diverse_select(self, cls_df: pd.DataFrame, target: int,
                        has_suitability: bool, has_subtype: bool,
                        has_bg: bool) -> pd.DataFrame:
        """
        다양성을 고려한 샘플 선택.
        
        전략:
        1. defect_subtype과 background_type의 조합별로 그룹화
        2. 각 그룹에서 suitability_score 상위 샘플을 라운드로빈으로 선택
        3. 그룹이 없으면 suitability_score 상위 선택
        """
        if not (has_subtype or has_bg):
            # 다양성 기준 없음 - suitability_score 상위 선택
            if has_suitability:
                return cls_df.nlargest(target, 'suitability_score')
            return cls_df.sample(n=target, random_state=42)
        
        # 다양성 그룹 키 생성
        if has_subtype and has_bg:
            cls_df = cls_df.copy()
            cls_df['_diversity_key'] = (
                cls_df['defect_subtype'].astype(str) + '|' +
                cls_df['background_type'].astype(str)
            )
        elif has_subtype:
            cls_df = cls_df.copy()
            cls_df['_diversity_key'] = cls_df['defect_subtype'].astype(str)
        else:
            cls_df = cls_df.copy()
            cls_df['_diversity_key'] = cls_df['background_type'].astype(str)
        
        # 각 그룹을 suitability_score 내림차순으로 정렬
        groups = {}
        for key, group_df in cls_df.groupby('_diversity_key'):
            if has_suitability:
                groups[key] = group_df.sort_values(
                    'suitability_score', ascending=False
                ).index.tolist()
            else:
                groups[key] = group_df.index.tolist()
        
        # 라운드로빈으로 선택
        selected_indices = []
        group_keys = sorted(groups.keys())
        group_pointers = {k: 0 for k in group_keys}
        
        while len(selected_indices) < target:
            added_this_round = False
            for key in group_keys:
                if len(selected_indices) >= target:
                    break
                ptr = group_pointers[key]
                if ptr < len(groups[key]):
                    selected_indices.append(groups[key][ptr])
                    group_pointers[key] = ptr + 1
                    added_this_round = True
            if not added_this_round:
                break  # 모든 그룹 소진
        
        result = cls_df.loc[selected_indices]
        if '_diversity_key' in result.columns:
            result = result.drop(columns=['_diversity_key'])
        return result
    
    def package_single_roi(self, roi_data: Dict, 
                          roi_image: np.ndarray,
                          roi_mask: np.ndarray,
                          output_dir: Path) -> Dict:
        """
        Package a single ROI with hint image and metadata.
        
        Args:
            roi_data: ROI metadata dictionary
            roi_image: ROI image array (H, W, 3)
            roi_mask: ROI mask array (H, W)
            output_dir: Output directory
            
        Returns:
            Updated roi_data with hint_path and prompt
        """
        # Generate hint image
        hint_image = self.hint_generator.generate_hint_image(
            roi_image=roi_image,
            roi_mask=roi_mask,
            defect_metrics=roi_data,
            background_type=roi_data.get('background_type', 'smooth'),
            stability_score=roi_data.get('stability_score', 0.5)
        )
        
        # Save hint image
        hint_dir = output_dir / 'hints'
        hint_dir.mkdir(parents=True, exist_ok=True)
        
        image_id = roi_data['image_id']
        class_id = roi_data['class_id']
        region_id = roi_data['region_id']
        hint_filename = f"{image_id}_class{class_id}_region{region_id}_hint.png"
        hint_path = hint_dir / hint_filename
        
        self.hint_generator.save_hint_image(hint_image, hint_path)
        
        # Generate prompt
        prompt = self.prompt_generator.generate_prompt(
            defect_subtype=roi_data.get('defect_subtype', 'general'),
            background_type=roi_data.get('background_type', 'smooth'),
            class_id=class_id,
            stability_score=roi_data.get('stability_score', 0.5),
            defect_metrics=roi_data,
            suitability_score=roi_data.get('suitability_score', 0.5)
        )
        
        negative_prompt = self.prompt_generator.generate_negative_prompt()
        
        # Update roi_data
        roi_data['hint_path'] = str(hint_path)
        roi_data['prompt'] = prompt
        roi_data['negative_prompt'] = negative_prompt
        
        return roi_data
    
    def create_train_jsonl(self, roi_metadata: List[Dict], 
                          output_path: Path,
                          relative_paths: bool = True,
                          base_dir: Optional[Path] = None):
        """
        Create train.jsonl file for ControlNet training.
        
        필드 역할 (train_controlnet.py 기준):
        - target: 학습 타겟 이미지 (VAE encode → 노이즈 예측 대상).
                  결함이 포함된 ROI 패치 이미지.
        - hint:   ControlNet 조건부 입력 (conditioning_pixel_values).
                  3채널 합성 이미지 (R=마스크, G=구조선, B=텍스처).
        - prompt: 텍스트 설명 (결함 특성 + 배경 유형).
        - source: 레거시 필드. 학습/추론 코드에서 사용하지 않음.
                  target과 동일값으로 설정 (하위 호환용).
        
        Format per line:
        {
            "source": "path/to/roi_image.png",    (미사용, 레거시 호환)
            "target": "path/to/roi_image.png",     (학습 타겟)
            "prompt": "a linear scratch on ...",
            "hint": "path/to/hint_image.png",      (ControlNet 조건부 입력)
            "negative_prompt": "blurry, low quality..."
        }
        
        Args:
            roi_metadata: List of ROI metadata dictionaries
            output_path: Path to save train.jsonl
            relative_paths: Use relative paths instead of absolute
            base_dir: Base directory for relative paths
        """
        jsonl_lines = []
        
        for roi_data in roi_metadata:
            # target = 결함 포함 ROI 이미지 (학습 대상)
            target_path = roi_data.get('roi_image_path', '')
            # hint = 3채널 conditioning 이미지 (ControlNet 입력)
            hint_path = roi_data.get('hint_path', '')
            
            if relative_paths and base_dir:
                try:
                    target_path = Path(target_path).relative_to(base_dir)
                    hint_path = Path(hint_path).relative_to(base_dir)
                except ValueError:
                    pass  # Keep absolute if relative conversion fails
            
            entry = {
                "source": str(target_path),   # 레거시 호환 — 학습/추론에서 미사용
                "target": str(target_path),    # 학습 타겟 (VAE encode 대상)
                "prompt": roi_data.get('prompt', ''),
                "hint": str(hint_path),        # ControlNet conditioning 입력
                "negative_prompt": roi_data.get('negative_prompt', '')
            }
            
            jsonl_lines.append(entry)
        
        # Write JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_lines:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Created train.jsonl with {len(jsonl_lines)} entries at: {output_path}")
    
    def create_metadata_json(self, roi_metadata: List[Dict], 
                            output_path: Path):
        """
        Create comprehensive metadata JSON file.
        
        Args:
            roi_metadata: List of ROI metadata dictionaries
            output_path: Path to save metadata.json
        """
        metadata = {
            'dataset_name': 'Severstal Steel Defect Detection - ControlNet Training Set',
            'total_samples': len(roi_metadata),
            'format': 'ControlNet training format with multi-channel hints',
            'channels': {
                'red': 'Defect mask with 4-indicator enhancement',
                'green': 'Background structure lines (edge information)',
                'blue': 'Background fine texture'
            },
            'prompt_structure': '[Defect characteristics] + [Background type] + [Surface condition]',
            'samples': roi_metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Created metadata.json at: {output_path}")
    
    def package_dataset(self, roi_metadata_df: pd.DataFrame,
                       train_images_dir: Path,
                       train_csv: Path,
                       output_dir: Path,
                       create_hints: bool = True,
                       max_samples: Optional[int] = None,
                       quality_filter: bool = True,
                       min_area: int = 100,
                       min_stability: float = 0.3,
                       min_matching: float = 0.5,
                       per_class_cap: Optional[int] = None,
                       rare_class_threshold_count: int = 200,
                       class_margin_overrides: Optional[Dict[int, Dict[str, float]]] = None,
                       edge_margin_x: float = 0.1,
                       edge_margin_y: float = 0.05,
                       num_workers: int = 0,
                       ) -> Path:
        """
        Package complete dataset for ControlNet training.
        
        Args:
            roi_metadata_df: DataFrame with ROI metadata from ROI extraction
            train_images_dir: Directory with original training images
            train_csv: Path to train.csv with RLE annotations
            output_dir: Output directory for packaged dataset
            create_hints: Whether to generate hint images
            max_samples: Maximum number of samples to package (균등 분배 모드).
                v2에서 50개로 overfitting이 발생했으므로, v3에서는 500 권장.
                per_class_cap이 지정되면 무시됨.
            quality_filter: 품질 필터 적용 여부 (기본: True)
            min_area: 최소 결함 영역 (px, 기본: 100)
            min_stability: 최소 stability_score (기본: 0.3)
            min_matching: 최소 matching_score (기본: 0.5)
            per_class_cap: v5.2 class-aware capping. 풍부한 클래스의 최대 샘플 수.
                지정 시 희소 클래스는 전수 포함, 풍부한 클래스는 cap까지만 선택.
            rare_class_threshold_count: 이 수 이하인 클래스를 희소로 간주 (기본 200).
                per_class_cap 모드에서만 사용.
            class_margin_overrides: B2 클래스별 엣지 마진 오버라이드.
                키: class_id (1-based), 값: {'x': float, 'y': float}.
                예: {4: {'x': 0.05, 'y': 0.0}} → Class 4는 Y 마진 없음.
                기본 None이면 기존 동작 유지.
            edge_margin_x: 좌우 엣지 마진 비율 (기본 0.1 = 10%).
                _edge_filter의 기본 마진을 외부에서 조정 가능.
            edge_margin_y: 상하 엣지 마진 비율 (기본 0.05 = 5%).
                _edge_filter의 기본 마진을 외부에서 조정 가능.
            num_workers: 병렬 워커 수 (0=순차 처리, >=2=ProcessPoolExecutor).
                ROI별 hint 생성 + 이미지 I/O를 병렬화.
            
        Returns:
            Path to output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("ControlNet Dataset Packaging")
        print("="*80)
        print(f"Input: {len(roi_metadata_df)} ROIs")
        print(f"Output: {output_dir}")
        print(f"Create hints: {create_hints}")
        print(f"Quality filter: {quality_filter}")
        print("="*80)
        
        # Edge proximity 필터 적용 (결함이 ROI 경계에 너무 가까운 샘플 제외)
        roi_metadata_df = self._edge_filter(
            roi_metadata_df,
            edge_margin_x=edge_margin_x,
            edge_margin_y=edge_margin_y,
            class_margin_overrides=class_margin_overrides,
        )

        # 품질 필터 적용
        if quality_filter:
            roi_metadata_df = self._quality_filter(
                roi_metadata_df,
                min_area=min_area,
                min_stability=min_stability,
                min_matching=min_matching,
            )
        
        # 필터 후 빈 DataFrame 방어
        if len(roi_metadata_df) == 0:
            print("\nWARNING: 모든 ROI가 필터링되었습니다. 패키징할 샘플이 없습니다.")
            print("  - edge_margin_x/y 값을 낮추거나")
            print("  - quality_filter=False로 설정하거나")
            print("  - class_margin_overrides로 특정 클래스 마진을 완화해 보세요.")
            # 빈 출력 디렉토리라도 반환하여 파이프라인 중단 방지
            return output_dir
        
        # Limit samples: class-aware capping (v5.2) 또는 균등 분배 (v3~v5.1)
        if per_class_cap is not None:
            # v5.2 class-aware capping: 희소 클래스 전수 + 풍부한 클래스 cap
            roi_metadata_df = self._stratified_sample(
                roi_metadata_df, n_samples=0,  # n_samples는 capping 모드에서 무시
                per_class_cap=per_class_cap,
                rare_class_threshold_count=rare_class_threshold_count
            )
            print(f"Class-aware capping: {len(roi_metadata_df)} samples selected")
            if 'class_id' in roi_metadata_df.columns:
                class_dist = roi_metadata_df['class_id'].value_counts().sort_index()
                print(f"  Class distribution: {dict(class_dist)}")
        elif max_samples and max_samples < len(roi_metadata_df):
            roi_metadata_df = self._stratified_sample(roi_metadata_df, max_samples)
            print(f"Stratified sampling: {max_samples} samples selected")
            if 'class_id' in roi_metadata_df.columns:
                class_dist = roi_metadata_df['class_id'].value_counts().sort_index()
                print(f"  Class distribution: {dict(class_dist)}")
        
        packaged_data = []
        
        use_parallel = num_workers > 1 and len(roi_metadata_df) > 10
        
        if use_parallel:
            # ── 병렬 경로: ProcessPoolExecutor ──
            print(f"\n  병렬 모드: {num_workers} workers, {len(roi_metadata_df)} ROIs")
            
            prompt_style = self.prompt_generator.style if hasattr(
                self.prompt_generator, 'style') else 'detailed'
            
            tasks = []
            for _, row in roi_metadata_df.iterrows():
                roi_dict = row.to_dict()
                tasks.append((roi_dict, str(output_dir), create_hints, prompt_style))
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_package_single_roi_worker, t): i
                    for i, t in enumerate(tasks)
                }
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc="Packaging ROIs (parallel)"):
                    result = future.result()
                    if result is not None:
                        packaged_data.append(result)
        else:
            # ── 순차 경로 (기존 동작) ──
            # Process each ROI
            for idx, row in tqdm(roi_metadata_df.iterrows(), 
                                total=len(roi_metadata_df),
                                desc="Packaging ROIs"):
                
                roi_data = row.to_dict()
                
                # Load ROI image
                roi_image_path = Path(row['roi_image_path'])
                if not roi_image_path.exists():
                    print(f"Warning: Image not found: {roi_image_path}")
                    continue
                
                roi_image = cv2.imread(str(roi_image_path))
                if roi_image is None:
                    print(f"Warning: Failed to read image (corrupt?): {roi_image_path}")
                    continue
                roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                
                # Load ROI mask
                roi_mask_path = Path(row['roi_mask_path'])
                if not roi_mask_path.exists():
                    print(f"Warning: Mask not found: {roi_mask_path}")
                    continue
                
                roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
                if roi_mask is None:
                    print(f"Warning: Failed to read mask (corrupt?): {roi_mask_path}")
                    continue
                roi_mask = (roi_mask > 0).astype(np.uint8)
                
                # Package this ROI
                if create_hints:
                    roi_data = self.package_single_roi(
                        roi_data=roi_data,
                        roi_image=roi_image_rgb,
                        roi_mask=roi_mask,
                        output_dir=output_dir
                    )
                else:
                    # Just generate prompts
                    prompt = self.prompt_generator.generate_prompt(
                        defect_subtype=roi_data.get('defect_subtype', 'general'),
                        background_type=roi_data.get('background_type', 'smooth'),
                        class_id=roi_data['class_id'],
                        stability_score=roi_data.get('stability_score', 0.5),
                        defect_metrics=roi_data,
                        suitability_score=roi_data.get('suitability_score', 0.5)
                    )
                    roi_data['prompt'] = prompt
                    roi_data['negative_prompt'] = self.prompt_generator.generate_negative_prompt()
                
                packaged_data.append(roi_data)
        
        print(f"\nSuccessfully packaged {len(packaged_data)} ROIs")
        
        # Create train.jsonl
        print("\nCreating train.jsonl...")
        train_jsonl_path = output_dir / 'train.jsonl'
        self.create_train_jsonl(
            packaged_data, 
            train_jsonl_path,
            relative_paths=True,
            base_dir=output_dir
        )
        
        # Create metadata.json
        print("\nCreating metadata.json...")
        metadata_path = output_dir / 'metadata.json'
        self.create_metadata_json(packaged_data, metadata_path)
        
        # Save updated ROI metadata
        print("\nSaving updated ROI metadata...")
        packaged_df = pd.DataFrame(packaged_data)
        packaged_csv = output_dir / 'packaged_roi_metadata.csv'
        packaged_df.to_csv(packaged_csv, index=False)
        print(f"Saved to: {packaged_csv}")
        
        # Create summary
        summary = {
            'total_packaged': len(packaged_data),
            'hints_created': create_hints,
            'output_directory': str(output_dir),
            'train_jsonl': str(train_jsonl_path),
            'metadata_json': str(metadata_path)
        }
        
        summary_path = output_dir / 'packaging_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ControlNet Dataset Packaging Summary\n")
            f.write("="*80 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nSummary saved to: {summary_path}")
        print("\n" + "="*80)
        print("Packaging complete!")
        print("="*80)
        
        return output_dir
