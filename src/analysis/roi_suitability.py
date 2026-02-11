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
                             search_radius: int = 32) -> Optional[Tuple[int, int, int, int]]:
        """
        Optimize ROI position by shifting to maximize background continuity.
        
        According to PROJECT(roi).md, if the defect centroid is near a background
        boundary, shift the ROI window toward more uniform background.
        
        Args:
            image: Original image
            defect_metrics: Defect characterization metrics
            background_analysis: Background analysis result
            roi_size: Size of ROI patch
            search_radius: Maximum shift distance in pixels
            
        Returns:
            Optimized bounding box (x1, y1, x2, y2) or None if no valid ROI
        """
        h, w = image.shape[:2]
        cx, cy = defect_metrics['centroid']
        cx, cy = int(cx), int(cy)
        
        # Initial ROI centered on defect
        half_size = roi_size // 2
        x1 = cx - half_size
        y1 = cy - half_size
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Check if initial ROI is valid
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            # Try to fit within image bounds
            x1 = max(0, min(cx - half_size, w - roi_size))
            y1 = max(0, min(cy - half_size, h - roi_size))
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return None  # Cannot fit ROI
        
        initial_bbox = (x1, y1, x2, y2)
        initial_continuity = self.background_analyzer.check_continuity(
            background_analysis, initial_bbox
        )
        
        # Search for better position within radius
        best_bbox = initial_bbox
        best_continuity = initial_continuity
        
        # Search grid
        step = max(1, search_radius // 4)
        for dx in range(-search_radius, search_radius + 1, step):
            for dy in range(-search_radius, search_radius + 1, step):
                new_x1 = x1 + dx
                new_y1 = y1 + dy
                new_x2 = new_x1 + roi_size
                new_y2 = new_y1 + roi_size
                
                # Check bounds
                if new_x1 < 0 or new_y1 < 0 or new_x2 > w or new_y2 > h:
                    continue
                
                # Check that defect is still in ROI
                defect_bbox = defect_metrics['bbox']
                dx1, dy1, dx2, dy2 = defect_bbox
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
