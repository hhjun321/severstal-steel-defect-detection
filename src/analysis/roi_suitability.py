"""
ROI Suitability Assessment Module

This module evaluates ROI suitability by matching defect characteristics
with background contexts. According to PROJECT(roi).md:
- Linear scratches are most natural on vertical/horizontal stripe backgrounds
- Compact blobs prefer smooth backgrounds
- Complex defects match complex pattern backgrounds

The suitability score guides optimal ROI selection for data augmentation.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from .defect_characterization import DefectCharacterizer
from .background_characterization import BackgroundAnalyzer, BackgroundType


class ROISuitabilityEvaluator:
    """
    Evaluates ROI suitability by matching defect-background combinations.
    """
    
    # Defect-Background matching rules (higher score = better match)
    MATCHING_RULES = {
        'linear_scratch': {
            BackgroundType.VERTICAL_STRIPE.value: 1.0,
            BackgroundType.HORIZONTAL_STRIPE.value: 1.0,
            BackgroundType.SMOOTH.value: 0.7,
            BackgroundType.TEXTURED.value: 0.5,
            BackgroundType.COMPLEX_PATTERN.value: 0.3,
        },
        'elongated': {
            BackgroundType.VERTICAL_STRIPE.value: 0.9,
            BackgroundType.HORIZONTAL_STRIPE.value: 0.9,
            BackgroundType.SMOOTH.value: 0.8,
            BackgroundType.TEXTURED.value: 0.6,
            BackgroundType.COMPLEX_PATTERN.value: 0.4,
        },
        'compact_blob': {
            BackgroundType.SMOOTH.value: 1.0,
            BackgroundType.TEXTURED.value: 0.7,
            BackgroundType.VERTICAL_STRIPE.value: 0.5,
            BackgroundType.HORIZONTAL_STRIPE.value: 0.5,
            BackgroundType.COMPLEX_PATTERN.value: 0.6,
        },
        'irregular': {
            BackgroundType.COMPLEX_PATTERN.value: 1.0,
            BackgroundType.TEXTURED.value: 0.8,
            BackgroundType.SMOOTH.value: 0.6,
            BackgroundType.VERTICAL_STRIPE.value: 0.5,
            BackgroundType.HORIZONTAL_STRIPE.value: 0.5,
        },
        'general': {
            BackgroundType.SMOOTH.value: 0.7,
            BackgroundType.TEXTURED.value: 0.7,
            BackgroundType.VERTICAL_STRIPE.value: 0.7,
            BackgroundType.HORIZONTAL_STRIPE.value: 0.7,
            BackgroundType.COMPLEX_PATTERN.value: 0.7,
        }
    }
    
    def __init__(self, defect_analyzer: DefectCharacterizer, 
                 background_analyzer: BackgroundAnalyzer):
        """
        Initialize ROI suitability evaluator.
        
        Args:
            defect_analyzer: DefectCharacterizer instance
            background_analyzer: BackgroundAnalyzer instance
        """
        self.defect_analyzer = defect_analyzer
        self.background_analyzer = background_analyzer
    
    def compute_matching_score(self, defect_subtype: str, background_type: str) -> float:
        """
        Compute matching score between defect subtype and background type.
        
        Args:
            defect_subtype: Defect sub-classification
            background_type: Background type
            
        Returns:
            Matching score (0-1)
        """
        if defect_subtype not in self.MATCHING_RULES:
            return 0.5  # Neutral score for unknown subtypes
        
        return self.MATCHING_RULES[defect_subtype].get(background_type, 0.5)
    
    def evaluate_roi_suitability(self, defect_metrics: Dict, 
                                 background_analysis: Dict, 
                                 bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Evaluate ROI suitability for a defect region.
        
        Args:
            defect_metrics: Defect characterization metrics
            background_analysis: Background analysis result from BackgroundAnalyzer
            bbox: ROI bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary containing:
            - 'suitability_score': Overall suitability (0-1)
            - 'matching_score': Defect-background matching score
            - 'continuity_score': Background continuity score
            - 'stability_score': Average background stability
            - 'recommendation': 'suitable', 'acceptable', or 'unsuitable'
        """
        # Get defect subtype
        defect_subtype = self.defect_analyzer.classify_defect_subtype(defect_metrics)
        
        # Get background continuity
        continuity_score = self.background_analyzer.check_continuity(
            background_analysis, bbox
        )
        
        # Get background type at defect centroid
        cx, cy = defect_metrics['centroid']
        bg_info = self.background_analyzer.get_background_at_location(
            background_analysis, int(cx), int(cy)
        )
        
        if bg_info is None:
            return {
                'suitability_score': 0.0,
                'matching_score': 0.0,
                'continuity_score': 0.0,
                'stability_score': 0.0,
                'recommendation': 'unsuitable',
                'defect_subtype': defect_subtype,
                'background_type': 'unknown'
            }
        
        background_type = bg_info['background_type']
        stability_score = bg_info['stability_score']
        
        # Compute matching score
        matching_score = self.compute_matching_score(defect_subtype, background_type)
        
        # Compute overall suitability (weighted combination)
        suitability_score = (
            0.5 * matching_score +       # Defect-background match is most important
            0.3 * continuity_score +     # Background should be continuous
            0.2 * stability_score        # Background should be stable
        )
        
        # Make recommendation
        if suitability_score >= 0.7:
            recommendation = 'suitable'
        elif suitability_score >= 0.5:
            recommendation = 'acceptable'
        else:
            recommendation = 'unsuitable'
        
        return {
            'suitability_score': float(suitability_score),
            'matching_score': float(matching_score),
            'continuity_score': float(continuity_score),
            'stability_score': float(stability_score),
            'recommendation': recommendation,
            'defect_subtype': defect_subtype,
            'background_type': background_type
        }
    
    def optimize_roi_position(self, image: np.ndarray, 
                             defect_metrics: Dict,
                             background_analysis: Dict,
                             roi_size: int = 512,
                             search_radius: int = 32,
                             min_context_pixels: int = 64) -> Optional[Tuple[int, int, int, int]]:
        """
        Optimize ROI position with adaptive sizing for non-square images.
        
        Severstal 이미지는 1600x256 (WxH)으로, 기존 512x512 정사각 ROI는
        이미지 높이(256px)를 초과하여 항상 실패했습니다 (v4 근본 원인 #1).
        
        v5 전략:
        1. 이미지 높이가 roi_size보다 작으면 높이를 이미지 높이로 제한
        2. 정사각형 ROI를 우선 시도 (min(roi_size, h) x min(roi_size, h))
        3. 결함이 ROI보다 크면 결함+padding으로 확장
        4. 결함 주변에 최소 min_context_pixels만큼 컨텍스트 보장
        5. 결과적으로 256x256 패치 → 2x 업스케일 (기존 10-34x 대비 대폭 개선)
        
        Args:
            image: Original image (H, W, C)
            defect_metrics: Defect characterization metrics
            background_analysis: Background analysis result
            roi_size: Target ROI size (will be adapted to image dimensions)
            search_radius: Maximum shift distance in pixels
            min_context_pixels: Minimum context padding around defect
            
        Returns:
            Optimized bounding box (x1, y1, x2, y2) or None if no valid ROI
        """
        h, w = image.shape[:2]
        cx, cy = defect_metrics['centroid']
        cx, cy = int(cx), int(cy)
        
        defect_bbox = defect_metrics['bbox']
        dx1, dy1, dx2, dy2 = defect_bbox
        defect_w = dx2 - dx1
        defect_h = dy2 - dy1
        
        # Adaptive ROI size: constrained by image dimensions
        # Use square ROI with side = min(roi_size, image_height, image_width)
        effective_size_h = min(roi_size, h)
        effective_size_w = min(roi_size, w)
        
        # Prefer square ROI using the smaller dimension
        base_size = min(effective_size_h, effective_size_w)
        
        # Ensure ROI is large enough to contain defect + context padding
        roi_h = max(base_size, defect_h + 2 * min_context_pixels)
        roi_w = max(base_size, defect_w + 2 * min_context_pixels)
        
        # Force square ROI to prevent train/inference resolution mismatch
        # Training: Resize(512) + CenterCrop(512) loses data on non-square
        # Inference: always generates 512x512 square
        # Use the smaller dimension to guarantee image bounds compliance
        roi_side = min(roi_h, roi_w, h, w)
        roi_h = roi_side
        roi_w = roi_side
        
        # Center ROI on defect centroid
        x1 = cx - roi_w // 2
        y1 = cy - roi_h // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        # Clamp to image bounds
        if x1 < 0:
            x1 = 0
            x2 = roi_w
        if y1 < 0:
            y1 = 0
            y2 = roi_h
        if x2 > w:
            x2 = w
            x1 = max(0, w - roi_w)
        if y2 > h:
            y2 = h
            y1 = max(0, h - roi_h)
        
        # Verify defect is fully contained
        # Note: with forced square ROI, very wide defects (> roi_side) may
        # not be fully contained. This is acceptable — partial coverage of
        # extremely wide defects is better than non-square ROIs that cause
        # train/inference mismatch. The defect center is always captured.
        if dx1 < x1 or dy1 < y1 or dx2 > x2 or dy2 > y2:
            # Try to shift ROI to contain as much of the defect as possible
            # WITHOUT changing ROI dimensions (keep square)
            shift_x = 0
            shift_y = 0
            if dx1 < x1:
                shift_x = dx1 - x1  # negative: shift left
            elif dx2 > x2:
                shift_x = dx2 - x2  # positive: shift right
            if dy1 < y1:
                shift_y = dy1 - y1
            elif dy2 > y2:
                shift_y = dy2 - y2
            
            x1 = max(0, min(x1 + shift_x, w - roi_w))
            y1 = max(0, min(y1 + shift_y, h - roi_h))
            x2 = x1 + roi_w
            y2 = y1 + roi_h
        
        initial_bbox = (x1, y1, x2, y2)
        initial_continuity = self.background_analyzer.check_continuity(
            background_analysis, initial_bbox
        )
        
        # Search for better position within radius
        best_bbox = initial_bbox
        best_continuity = initial_continuity
        
        # Search grid
        step = max(1, search_radius // 4)
        for ddx in range(-search_radius, search_radius + 1, step):
            for ddy in range(-search_radius, search_radius + 1, step):
                new_x1 = x1 + ddx
                new_y1 = y1 + ddy
                new_x2 = new_x1 + roi_w
                new_y2 = new_y1 + roi_h
                
                # Check bounds
                if new_x1 < 0 or new_y1 < 0 or new_x2 > w or new_y2 > h:
                    continue
                
                # Check that defect is still fully in ROI
                if dx1 < new_x1 or dy1 < new_y1 or dx2 > new_x2 or dy2 > new_y2:
                    continue
                
                # Evaluate continuity
                candidate_bbox = (new_x1, new_y1, new_x2, new_y2)
                continuity = self.background_analyzer.check_continuity(
                    background_analysis, candidate_bbox
                )
                
                if continuity > best_continuity:
                    best_continuity = continuity
                    best_bbox = candidate_bbox
        
        return best_bbox
    
    def generate_prompt_for_roi(self, defect_subtype: str, background_type: str, 
                                class_id: int) -> str:
        """
        Generate text prompt for ControlNet training/inference.
        
        Args:
            defect_subtype: Defect sub-classification
            background_type: Background type
            class_id: Defect class (1-4)
            
        Returns:
            Text prompt describing the defect-background combination
        """
        # Defect descriptions
        defect_desc = {
            'linear_scratch': 'a linear scratch defect',
            'elongated': 'an elongated defect',
            'compact_blob': 'a compact blob defect',
            'irregular': 'an irregular defect',
            'general': 'a defect'
        }
        
        # Background descriptions
        bg_desc = {
            BackgroundType.SMOOTH.value: 'smooth metal surface',
            BackgroundType.TEXTURED.value: 'textured metal surface',
            BackgroundType.VERTICAL_STRIPE.value: 'vertical striped metal surface',
            BackgroundType.HORIZONTAL_STRIPE.value: 'horizontal striped metal surface',
            BackgroundType.COMPLEX_PATTERN.value: 'complex patterned metal surface'
        }
        
        defect_text = defect_desc.get(defect_subtype, 'a defect')
        bg_text = bg_desc.get(background_type, 'metal surface')
        
        prompt = f"{defect_text} on {bg_text}, class {class_id}, industrial steel defect detection"
        
        return prompt
