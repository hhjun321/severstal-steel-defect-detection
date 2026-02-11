"""
ROI Extraction and Data Packaging Module

This module implements the complete pipeline from PROJECT(roi).md:
1. Background analysis (grid-based labeling)
2. Defect analysis (4 indicators)
3. ROI suitability assessment
4. ROI extraction with position optimization
5. Data packaging for ControlNet training
"""
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ..utils.rle_utils import decode_mask_from_csv, get_all_masks_for_image
from src.analysis.defect_characterization import DefectCharacterizer
from src.analysis.background_characterization import BackgroundAnalyzer
from src.analysis.roi_suitability import ROISuitabilityEvaluator


class ROIExtractor:
    """
    Complete ROI extraction pipeline.
    """
    
    def __init__(self, 
                 defect_analyzer: Optional[DefectCharacterizer] = None,
                 background_analyzer: Optional[BackgroundAnalyzer] = None,
                 roi_evaluator: Optional[ROISuitabilityEvaluator] = None,
                 roi_size: int = 512,
                 min_suitability: float = 0.5):
        """
        Initialize ROI extractor.
        
        Args:
            defect_analyzer: DefectCharacterizer instance
            background_analyzer: BackgroundAnalyzer instance
            roi_evaluator: ROISuitabilityEvaluator instance
            roi_size: Size of ROI patches
            min_suitability: Minimum suitability score to accept ROI
        """
        self.defect_analyzer = defect_analyzer or DefectCharacterizer()
        self.background_analyzer = background_analyzer or BackgroundAnalyzer(
            grid_size=64, 
            variance_threshold=100.0,
            edge_threshold=0.3
        )
        self.roi_evaluator = roi_evaluator or ROISuitabilityEvaluator(
            self.defect_analyzer,
            self.background_analyzer
        )
        self.roi_size = roi_size
        self.min_suitability = min_suitability
    
    def process_single_image(self, image_path: str, train_df: pd.DataFrame, 
                            image_id: str) -> List[Dict]:
        """
        Process a single image and extract all suitable ROIs.
        
        Args:
            image_path: Path to image file
            train_df: Training dataframe with mask annotations
            image_id: Image identifier
            
        Returns:
            List of ROI metadata dictionaries
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Step 1: Background analysis
        background_analysis = self.background_analyzer.analyze_image(image_rgb)
        
        # Step 2: Get all defect masks for this image
        masks = get_all_masks_for_image(image_id, train_df, shape=(h, w))
        
        if len(masks) == 0:
            return []
        
        # Step 3: Analyze each defect class
        roi_results = []
        
        for class_id, mask in masks.items():
            # Analyze all separate defects in this class mask
            defect_regions = self.defect_analyzer.analyze_all_defects_in_mask(
                mask, class_id
            )
            
            for defect_metrics in defect_regions:
                # Get initial bbox from defect
                defect_bbox = defect_metrics['bbox']
                
                # Evaluate suitability with initial bbox
                suitability = self.roi_evaluator.evaluate_roi_suitability(
                    defect_metrics,
                    background_analysis,
                    defect_bbox
                )
                
                # If unsuitable, try to optimize position
                if suitability['suitability_score'] < self.min_suitability:
                    optimized_bbox = self.roi_evaluator.optimize_roi_position(
                        image_rgb,
                        defect_metrics,
                        background_analysis,
                        roi_size=self.roi_size,
                        search_radius=32
                    )
                    
                    if optimized_bbox is not None:
                        # Re-evaluate with optimized position
                        suitability = self.roi_evaluator.evaluate_roi_suitability(
                            defect_metrics,
                            background_analysis,
                            optimized_bbox
                        )
                        roi_bbox = optimized_bbox
                    else:
                        roi_bbox = None
                else:
                    # Optimize anyway to potentially improve
                    optimized_bbox = self.roi_evaluator.optimize_roi_position(
                        image_rgb,
                        defect_metrics,
                        background_analysis,
                        roi_size=self.roi_size,
                        search_radius=32
                    )
                    roi_bbox = optimized_bbox if optimized_bbox is not None else defect_bbox
                
                # Only keep ROIs above threshold
                if roi_bbox is not None and suitability['suitability_score'] >= self.min_suitability:
                    # Generate prompt
                    prompt = self.roi_evaluator.generate_prompt_for_roi(
                        suitability['defect_subtype'],
                        suitability['background_type'],
                        class_id
                    )
                    
                    roi_data = {
                        'image_id': image_id,
                        'class_id': class_id,
                        'region_id': defect_metrics['region_id'],
                        'roi_bbox': roi_bbox,
                        'defect_bbox': defect_bbox,
                        'centroid': defect_metrics['centroid'],
                        'area': defect_metrics['area'],
                        'linearity': defect_metrics['linearity'],
                        'solidity': defect_metrics['solidity'],
                        'extent': defect_metrics['extent'],
                        'aspect_ratio': defect_metrics['aspect_ratio'],
                        'defect_subtype': suitability['defect_subtype'],
                        'background_type': suitability['background_type'],
                        'suitability_score': suitability['suitability_score'],
                        'matching_score': suitability['matching_score'],
                        'continuity_score': suitability['continuity_score'],
                        'stability_score': suitability['stability_score'],
                        'recommendation': suitability['recommendation'],
                        'prompt': prompt
                    }
                    
                    roi_results.append(roi_data)
        
        return roi_results
    
    def extract_roi_patch(self, image: np.ndarray, mask: np.ndarray, 
                         roi_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ROI patch from image and mask.
        
        Args:
            image: Original image (H, W, 3)
            mask: Binary mask (H, W)
            roi_bbox: (x1, y1, x2, y2)
            
        Returns:
            Tuple of (roi_image, roi_mask)
        """
        x1, y1, x2, y2 = roi_bbox
        roi_image = image[y1:y2, x1:x2].copy()
        roi_mask = mask[y1:y2, x1:x2].copy()
        
        return roi_image, roi_mask
    
    def save_roi_data(self, image: np.ndarray, mask: np.ndarray,
                     roi_data: Dict, output_dir: Path, 
                     save_patches: bool = True) -> Dict:
        """
        Save ROI patch and update metadata.
        
        Args:
            image: Original image
            mask: Binary mask for this class
            roi_data: ROI metadata dictionary
            output_dir: Output directory
            save_patches: Whether to save image/mask patches
            
        Returns:
            Updated roi_data with file paths
        """
        roi_bbox = roi_data['roi_bbox']
        
        if save_patches:
            # Extract patches
            roi_image, roi_mask = self.extract_roi_patch(image, mask, roi_bbox)
            
            # Create filename
            image_id = roi_data['image_id']
            class_id = roi_data['class_id']
            region_id = roi_data['region_id']
            filename = f"{image_id}_class{class_id}_region{region_id}"
            
            # Save image patch
            image_dir = output_dir / 'images'
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{filename}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
            
            # Save mask patch
            mask_dir = output_dir / 'masks'
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{filename}.png"
            cv2.imwrite(str(mask_path), roi_mask * 255)
            
            roi_data['roi_image_path'] = str(image_path)
            roi_data['roi_mask_path'] = str(mask_path)
        
        return roi_data
    
    def process_dataset(self, image_dir: Path, train_csv: Path, 
                       output_dir: Path, save_patches: bool = True,
                       max_images: Optional[int] = None) -> pd.DataFrame:
        """
        Process entire dataset and extract all ROIs.
        
        Args:
            image_dir: Directory containing training images
            train_csv: Path to train.csv with annotations
            output_dir: Output directory for ROI data
            save_patches: Whether to save image/mask patches
            max_images: Maximum number of images to process (for testing)
            
        Returns:
            DataFrame with all ROI metadata
        """
        # Load training data
        train_df = pd.read_csv(train_csv)
        
        # Get unique images that have defects
        image_ids = train_df['ImageId'].unique()
        
        if max_images is not None:
            image_ids = image_ids[:max_images]
        
        all_roi_data = []
        
        # Process each image
        for image_id in tqdm(image_ids, desc="Processing images"):
            image_path = str(image_dir / image_id)
            
            if not Path(image_path).exists():
                continue
            
            # Extract ROIs
            roi_results = self.process_single_image(image_path, train_df, image_id)
            
            if len(roi_results) == 0:
                continue
            
            # Load image and masks for saving
            if save_patches:
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image_rgb.shape[:2]
                masks = get_all_masks_for_image(image_id, train_df, shape=(h, w))
            
            # Save each ROI
            for roi_data in roi_results:
                if save_patches:
                    class_id = roi_data['class_id']
                    mask = masks.get(class_id)
                    if mask is not None:
                        roi_data = self.save_roi_data(
                            image_rgb, mask, roi_data, output_dir, save_patches
                        )
                
                all_roi_data.append(roi_data)
        
        # Convert to DataFrame
        roi_df = pd.DataFrame(all_roi_data)
        
        return roi_df
