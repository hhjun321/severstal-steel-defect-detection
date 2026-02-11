"""
Background ROI Extraction for Augmentation Generation (Stage 3)
증강 생성을 위한 배경 ROI 추출 (3단계)

This module extracts clean background regions from images to use as templates
for placing synthetic defects during augmentation generation.

깨끗한 배경 영역을 추출하여 증강 생성 시 합성 결함을 배치할 템플릿으로 사용합니다.

Key Features:
- Extracts backgrounds from clean (defect-free) images
- Falls back to minimal-defect images if no clean images exist
- Classifies backgrounds into 5 types (smooth, vertical_stripe, etc.)
- Ensures diversity by selecting one ROI per background type
- Computes stability scores for background quality
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from ..analysis.background_characterization import BackgroundCharacterizer


@dataclass
class BackgroundROI:
    """
    Metadata for an extracted background ROI.
    추출된 배경 ROI의 메타데이터
    """
    image_id: str
    roi_index: int
    x_start: int
    y_start: int
    width: int
    height: int
    background_type: str
    stability_score: float
    num_defects_in_image: int
    patch_path: str  # Path where patch is saved
    

class BackgroundExtractor:
    """
    Extracts clean background regions from steel images for augmentation.
    증강을 위해 강철 이미지에서 깨끗한 배경 영역을 추출합니다.
    
    Algorithm:
    1. Find clean images (no defects) or minimal-defect images (1 defect)
    2. Analyze background using grid-based classification
    3. Extract diverse background ROIs (one per type)
    4. Compute stability scores
    5. Save patches and metadata
    """
    
    def __init__(self,
                 roi_size: int = 512,
                 grid_size: int = 64,
                 min_stability: float = 0.6,
                 rois_per_image: int = 5):
        """
        Initialize background extractor.
        
        Args:
            roi_size: Size of extracted ROI patches (default: 512x512)
            grid_size: Grid size for background analysis (default: 64x64)
            min_stability: Minimum stability score to accept ROI (default: 0.6)
            rois_per_image: Number of diverse ROIs to extract per image (default: 5)
        """
        self.roi_size = roi_size
        self.grid_size = grid_size
        self.min_stability = min_stability
        self.rois_per_image = rois_per_image
        
        # Initialize background characterizer
        self.bg_characterizer = BackgroundCharacterizer(
            grid_size=grid_size,
            variance_threshold=50.0,
            edge_threshold=0.3
        )
        
        # Background types in priority order
        self.background_types = [
            'smooth',
            'vertical_stripe',
            'horizontal_stripe',
            'textured',
            'complex_pattern'
        ]
    
    def find_clean_images(self, train_csv_path: Path, train_images_dir: Path) -> List[str]:
        """
        Find images with NO defects (images NOT in train.csv).
        결함이 없는 이미지 찾기 (train.csv에 없는 이미지)
        
        RESEARCH PROTOCOL:
        - train.csv contains ONLY images WITH defects
        - Clean images are NOT listed in train.csv
        - We find clean images by: all_images - images_in_train_csv
        
        연구 원칙:
        - train.csv는 결함이 있는 이미지만 포함
        - 깨끗한 이미지는 train.csv에 기록되지 않음
        - 깨끗한 이미지 = 전체 이미지 - train.csv 이미지
        
        Args:
            train_csv_path: Path to train.csv
            train_images_dir: Path to train_images/ directory
            
        Returns:
            List of clean image filenames (not in train.csv)
        """
        # Get all images from directory
        all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
        
        # Get images with defects (in train.csv)
        train_df = pd.read_csv(train_csv_path)
        images_with_defects = set(train_df['ImageId'].unique())
        
        # Clean images = all images - images with defects
        clean_images = list(all_images - images_with_defects)
        
        return clean_images
    
    def analyze_image_background(self, image: np.ndarray) -> Dict:
        """
        Analyze background characteristics across entire image.
        전체 이미지의 배경 특성 분석
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Dictionary with grid-based background analysis
        """
        return self.bg_characterizer.analyze_background(image)
    
    def select_diverse_rois(self, 
                           image: np.ndarray,
                           background_grid: np.ndarray,
                           defect_mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Select diverse background ROIs covering different background types.
        다양한 배경 타입을 포함하는 배경 ROI 선택
        
        Args:
            image: RGB image (H, W, 3)
            background_grid: Grid of background type labels (grid_h, grid_w)
            defect_mask: Optional binary mask of defects to avoid
            
        Returns:
            List of ROI dictionaries with position and metadata
        """
        h, w = image.shape[:2]
        grid_h, grid_w = background_grid.shape
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        selected_rois = []
        
        # For each background type, find the best ROI
        for bg_type_idx, bg_type in enumerate(self.background_types):
            # Find all grid cells with this background type
            matching_cells = np.argwhere(background_grid == bg_type_idx)
            
            if len(matching_cells) == 0:
                continue
            
            best_roi = None
            best_score = -1
            
            # Try multiple candidate positions
            for _ in range(min(20, len(matching_cells))):
                # Random sampling for efficiency
                cell_idx = np.random.randint(len(matching_cells))
                cell_y, cell_x = matching_cells[cell_idx]
                
                # Convert grid cell to pixel coordinates
                center_y = int((cell_y + 0.5) * cell_h)
                center_x = int((cell_x + 0.5) * cell_w)
                
                # Calculate ROI bounds
                y_start = max(0, center_y - self.roi_size // 2)
                x_start = max(0, center_x - self.roi_size // 2)
                y_end = min(h, y_start + self.roi_size)
                x_end = min(w, x_start + self.roi_size)
                
                # Adjust if ROI extends beyond image
                if y_end - y_start < self.roi_size:
                    y_start = max(0, y_end - self.roi_size)
                if x_end - x_start < self.roi_size:
                    x_start = max(0, x_end - self.roi_size)
                
                # Skip if still too small
                if y_end - y_start < self.roi_size or x_end - x_start < self.roi_size:
                    continue
                
                # Extract ROI
                roi_patch = image[y_start:y_end, x_start:x_end]
                
                # Check for defect overlap if mask provided
                if defect_mask is not None:
                    roi_defect = defect_mask[y_start:y_end, x_start:x_end]
                    defect_ratio = roi_defect.sum() / (self.roi_size * self.roi_size)
                    if defect_ratio > 0.05:  # Reject if >5% defect coverage
                        continue
                
                # Compute stability score
                stability = self._compute_stability_score(roi_patch)
                
                if stability > best_score:
                    best_score = stability
                    best_roi = {
                        'x_start': x_start,
                        'y_start': y_start,
                        'width': x_end - x_start,
                        'height': y_end - y_start,
                        'background_type': bg_type,
                        'stability_score': stability
                    }
            
            # Add best ROI for this background type
            if best_roi is not None and best_roi['stability_score'] >= self.min_stability:
                selected_rois.append(best_roi)
                
                # Stop if we have enough ROIs
                if len(selected_rois) >= self.rois_per_image:
                    break
        
        return selected_rois
    
    def _compute_stability_score(self, roi_patch: np.ndarray) -> float:
        """
        Compute stability score for a background ROI.
        배경 ROI의 안정성 점수 계산
        
        Stability measures background uniformity and consistency.
        Higher scores indicate more stable, predictable backgrounds.
        
        Args:
            roi_patch: RGB patch (roi_size, roi_size, 3)
            
        Returns:
            Stability score [0, 1]
        """
        gray = cv2.cvtColor(roi_patch, cv2.COLOR_RGB2GRAY)
        
        # 1. Variance uniformity (lower variance = more stable)
        variance = np.var(gray)
        variance_score = 1.0 / (1.0 + variance / 1000.0)  # Normalize to [0, 1]
        
        # 2. Local consistency (split into 4 quadrants, compare variances)
        h, w = gray.shape
        q1 = gray[:h//2, :w//2]
        q2 = gray[:h//2, w//2:]
        q3 = gray[h//2:, :w//2]
        q4 = gray[h//2:, w//2:]
        
        quadrant_vars = [np.var(q) for q in [q1, q2, q3, q4]]
        var_std = np.std(quadrant_vars)
        consistency_score = 1.0 / (1.0 + var_std / 500.0)
        
        # 3. Edge density (fewer edges = more stable)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (h * w * 255.0)
        edge_score = 1.0 - min(1.0, edge_density * 10.0)
        
        # Weighted combination
        stability = (
            0.4 * variance_score +
            0.3 * consistency_score +
            0.3 * edge_score
        )
        
        return float(stability)
    
    def process_single_image(self,
                            image_path: Path,
                            image_id: str,
                            output_dir: Path) -> List[BackgroundROI]:
        """
        Extract background ROIs from a single CLEAN image (no defects).
        깨끗한 이미지에서 배경 ROI 추출 (결함 없음)
        
        RESEARCH PROTOCOL:
        - This function processes CLEAN images (not in train.csv)
        - No defect masks needed (images are defect-free)
        - Extracts diverse background regions for synthetic defect placement
        
        연구 원칙:
        - 이 함수는 깨끗한 이미지 처리 (train.csv에 없음)
        - 결함 마스크 불필요 (이미지에 결함 없음)
        - 합성 결함 배치를 위한 다양한 배경 영역 추출
        
        Args:
            image_path: Path to clean image file
            image_id: Image identifier (filename)
            output_dir: Directory to save background patches
            
        Returns:
            List of BackgroundROI metadata objects
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # No defect masks - image is clean
        combined_defect_mask = None
        num_defects = 0
        
        # Analyze background
        bg_analysis = self.analyze_image_background(image_rgb)
        background_grid = bg_analysis['background_types']  # (grid_h, grid_w)
        
        # Select diverse ROIs (no defect mask to avoid)
        roi_candidates = self.select_diverse_rois(
            image_rgb,
            background_grid,
            defect_mask=None  # Clean image - no defects to avoid
        )
        
        # Save patches and create metadata
        background_rois = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for roi_idx, roi_data in enumerate(roi_candidates):
            # Extract patch
            x1, y1 = roi_data['x_start'], roi_data['y_start']
            x2 = x1 + roi_data['width']
            y2 = y1 + roi_data['height']
            patch = image_rgb[y1:y2, x1:x2]
            
            # Save patch
            patch_filename = f"{image_id}_bg_roi_{roi_idx}_{roi_data['background_type']}.png"
            patch_path = output_dir / patch_filename
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(patch_path), patch_bgr)
            
            # Create metadata
            bg_roi = BackgroundROI(
                image_id=image_id,
                roi_index=roi_idx,
                x_start=roi_data['x_start'],
                y_start=roi_data['y_start'],
                width=roi_data['width'],
                height=roi_data['height'],
                background_type=roi_data['background_type'],
                stability_score=roi_data['stability_score'],
                num_defects_in_image=num_defects,
                patch_path=str(patch_path)
            )
            background_rois.append(bg_roi)
        
        return background_rois
    
    def process_dataset(self,
                       train_csv_path: Path,
                       train_images_dir: Path,
                       output_dir: Path,
                       max_images: Optional[int] = None) -> pd.DataFrame:
        """
        Process CLEAN images to extract background library.
        깨끗한 이미지를 처리하여 배경 라이브러리 추출
        
        RESEARCH PROTOCOL:
        - Only processes images NOT in train.csv (clean images)
        - train.csv images have defects and are used for ROI extraction
        - Clean images provide defect-free backgrounds for augmentation
        
        연구 원칙:
        - train.csv에 없는 이미지만 처리 (깨끗한 이미지)
        - train.csv 이미지는 결함이 있어 ROI 추출에 사용
        - 깨끗한 이미지는 증강을 위한 결함 없는 배경 제공
        
        Args:
            train_csv_path: Path to train.csv (to identify images WITH defects)
            train_images_dir: Directory containing all training images
            output_dir: Directory to save background patches
            max_images: Maximum number of CLEAN images to process (None = all)
            
        Returns:
            DataFrame with all background ROI metadata
        """
        # Find clean images (NOT in train.csv)
        clean_images = self.find_clean_images(train_csv_path, train_images_dir)
        
        print(f"\n" + "="*60)
        print("CLEAN IMAGE IDENTIFICATION")
        print("="*60)
        print(f"Total images in directory:  {len(list(train_images_dir.glob('*.jpg')))}")
        
        train_df = pd.read_csv(train_csv_path)
        images_with_defects = train_df['ImageId'].nunique()
        print(f"Images WITH defects (train.csv): {images_with_defects}")
        print(f"Clean images (NOT in train.csv): {len(clean_images)}")
        
        if len(clean_images) == 0:
            print("\nERROR: No clean images found!")
            print("This should not happen - check data directory")
            return pd.DataFrame()
        
        if max_images is not None:
            clean_images = clean_images[:max_images]
            print(f"\nProcessing first {max_images} clean images")
        else:
            print(f"\nProcessing all {len(clean_images)} clean images")
        
        # Process each clean image
        all_background_rois = []
        
        for image_id in tqdm(clean_images, desc="Extracting backgrounds"):
            image_path = train_images_dir / image_id
            
            if not image_path.exists():
                continue
            
            rois = self.process_single_image(
                image_path,
                image_id,
                output_dir
            )
            
            all_background_rois.extend(rois)
        
        # Convert to DataFrame
        if len(all_background_rois) > 0:
            metadata_df = pd.DataFrame([
                {
                    'image_id': roi.image_id,
                    'roi_index': roi.roi_index,
                    'x_start': roi.x_start,
                    'y_start': roi.y_start,
                    'width': roi.width,
                    'height': roi.height,
                    'background_type': roi.background_type,
                    'stability_score': roi.stability_score,
                    'num_defects_in_image': roi.num_defects_in_image,
                    'patch_path': roi.patch_path
                }
                for roi in all_background_rois
            ])
            
            # Save metadata
            metadata_path = output_dir / 'background_metadata.csv'
            metadata_df.to_csv(metadata_path, index=False)
            print(f"\nExtracted {len(metadata_df)} background ROIs")
            print(f"Metadata saved to: {metadata_path}")
            
            # Print statistics
            print("\nBackground Type Distribution:")
            for bg_type in self.background_types:
                count = (metadata_df['background_type'] == bg_type).sum()
                pct = 100.0 * count / len(metadata_df)
                print(f"  {bg_type:20s}: {count:4d} ({pct:5.1f}%)")
            
            print(f"\nStability Score Statistics:")
            print(f"  Mean:   {metadata_df['stability_score'].mean():.3f}")
            print(f"  Median: {metadata_df['stability_score'].median():.3f}")
            print(f"  Min:    {metadata_df['stability_score'].min():.3f}")
            print(f"  Max:    {metadata_df['stability_score'].max():.3f}")
            
            return metadata_df
        else:
            print("\nNo background ROIs extracted!")
            return pd.DataFrame()


def main():
    """
    Example usage of BackgroundExtractor.
    
    RESEARCH PROTOCOL:
    This extracts backgrounds from CLEAN images (NOT in train.csv).
    """
    # Paths
    project_root = Path(r"D:\project\severstal-steel-defect-detection")
    train_csv = project_root / "train.csv"
    train_images = project_root / "train_images"
    output_dir = project_root / "data" / "processed" / "background_patches"
    
    print("="*80)
    print("BACKGROUND EXTRACTION FROM CLEAN IMAGES")
    print("결함 없는 이미지에서 배경 추출")
    print("="*80)
    print("\nRESEARCH PROTOCOL:")
    print("  - Extract backgrounds from images NOT in train.csv")
    print("  - train.csv contains images WITH defects (used for ROI extraction)")
    print("  - Clean images provide defect-free backgrounds for augmentation")
    print("\n연구 원칙:")
    print("  - train.csv에 없는 이미지에서 배경 추출")
    print("  - train.csv는 결함 있는 이미지 포함 (ROI 추출용)")
    print("  - 깨끗한 이미지는 증강용 결함 없는 배경 제공")
    
    # Initialize extractor
    extractor = BackgroundExtractor(
        roi_size=512,
        grid_size=64,
        min_stability=0.6,
        rois_per_image=5
    )
    
    # Process clean images (NOT in train.csv)
    metadata_df = extractor.process_dataset(
        train_csv_path=train_csv,
        train_images_dir=train_images,
        output_dir=output_dir,
        max_images=100  # Process first 100 clean images
    )
    
    print("\nBackground extraction complete!")
    print(f"Total backgrounds extracted: {len(metadata_df)}")


if __name__ == '__main__':
    main()
