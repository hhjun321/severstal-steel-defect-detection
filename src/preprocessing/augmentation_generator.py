"""
Augmentation Generation (Stage 3)
증강 생성 (3단계)

This module implements the augmentation generation pipeline that combines:
1. Background templates from clean images
2. Defect templates from ROI extraction
3. ControlNet-based defect generation

이 모듈은 다음을 결합하는 증강 생성 파이프라인을 구현합니다:
1. 깨끗한 이미지의 배경 템플릿
2. ROI 추출의 결함 템플릿
3. ControlNet 기반 결함 생성

Architecture:
    Background Library + Defect Library → Template Matching → ControlNet Generation
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm

from .background_library import BackgroundLibrary, BackgroundTemplate
from ..utils.rle_utils import decode_mask_from_csv


@dataclass
class DefectTemplate:
    """
    Metadata for a defect template from ROI extraction.
    ROI 추출의 결함 템플릿 메타데이터
    """
    image_id: str
    roi_index: int
    class_id: int
    defect_type: str
    background_type: str
    suitability_score: float
    matching_score: float
    patch_path: Path
    mask_path: Path


@dataclass
class AugmentationSpec:
    """
    Specification for generating one augmented sample.
    증강 샘플 생성 사양
    """
    aug_id: str
    background_template: BackgroundTemplate
    defect_template: DefectTemplate
    compatibility_score: float
    generation_params: Dict


class AugmentationGenerator:
    """
    Generates synthetic defect images by combining backgrounds and defect templates.
    배경과 결함 템플릿을 결합하여 합성 결함 이미지 생성
    
    This is Stage 3 of the CASDA pipeline.
    """
    
    def __init__(self,
                 background_library: BackgroundLibrary,
                 defect_metadata_path: Path,
                 output_dir: Path,
                 min_compatibility: float = 0.5):
        """
        Initialize augmentation generator.
        
        Args:
            background_library: BackgroundLibrary instance
            defect_metadata_path: Path to roi_metadata.csv from Stage 1
            output_dir: Directory for augmented samples
            min_compatibility: Minimum compatibility score for matching
        """
        self.bg_library = background_library
        self.output_dir = output_dir
        self.min_compatibility = min_compatibility
        
        # Load defect library
        self.defect_df = pd.read_csv(defect_metadata_path)
        print(f"Loaded {len(self.defect_df)} defect templates")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "hints").mkdir(exist_ok=True)
    
    def load_defect_templates(self,
                             class_id: Optional[int] = None,
                             min_suitability: float = 0.5) -> List[DefectTemplate]:
        """
        Load defect templates from ROI metadata.
        ROI 메타데이터에서 결함 템플릿 로드
        
        Args:
            class_id: Filter by defect class (None = all classes)
            min_suitability: Minimum suitability score
            
        Returns:
            List of DefectTemplate objects
        """
        df = self.defect_df
        
        # Filter by class
        if class_id is not None:
            df = df[df['class_id'] == class_id]
        
        # Filter by suitability
        df = df[df['suitability_score'] >= min_suitability]
        
        # Convert to DefectTemplate objects
        templates = []
        for _, row in df.iterrows():
            template = DefectTemplate(
                image_id=row['image_id'],
                roi_index=row['roi_index'],
                class_id=row['class_id'],
                defect_type=row['defect_type'],
                background_type=row['background_type'],
                suitability_score=row['suitability_score'],
                matching_score=row['matching_score'],
                patch_path=Path(row['patch_path']),
                mask_path=Path(row['mask_path'])
            )
            templates.append(template)
        
        return templates
    
    def match_templates(self,
                       defect_template: DefectTemplate,
                       max_backgrounds: int = 10) -> List[Tuple[BackgroundTemplate, float]]:
        """
        Find compatible backgrounds for a defect template.
        결함 템플릿에 대한 호환 가능한 배경 찾기
        
        Uses compatibility matrix from BackgroundLibrary.
        
        Args:
            defect_template: DefectTemplate to match
            max_backgrounds: Maximum number of backgrounds to return
            
        Returns:
            List of (BackgroundTemplate, compatibility_score) tuples
        """
        return self.bg_library.get_compatible_backgrounds(
            defect_type=defect_template.defect_type,
            min_compatibility=self.min_compatibility,
            min_stability=0.6,
            max_results=max_backgrounds
        )
    
    def create_augmentation_specs(self,
                                 n_samples: int,
                                 class_distribution: Optional[Dict[int, float]] = None) -> List[AugmentationSpec]:
        """
        Create specifications for N augmented samples.
        N개의 증강 샘플 사양 생성
        
        Args:
            n_samples: Number of augmented samples to generate
            class_distribution: Optional class distribution (class_id -> probability)
                               If None, uses uniform distribution
            
        Returns:
            List of AugmentationSpec objects
        """
        # Default: uniform distribution across classes
        if class_distribution is None:
            unique_classes = self.defect_df['class_id'].unique()
            class_distribution = {c: 1.0/len(unique_classes) for c in unique_classes}
        
        # Normalize distribution
        total = sum(class_distribution.values())
        class_distribution = {k: v/total for k, v in class_distribution.items()}
        
        # Calculate samples per class
        samples_per_class = {
            class_id: int(n_samples * prob)
            for class_id, prob in class_distribution.items()
        }
        
        # Adjust for rounding errors
        total_allocated = sum(samples_per_class.values())
        if total_allocated < n_samples:
            # Add remaining to most common class
            most_common = max(class_distribution.keys(), key=lambda k: class_distribution[k])
            samples_per_class[most_common] += n_samples - total_allocated
        
        print(f"\nGenerating {n_samples} augmentation specifications:")
        for class_id, count in samples_per_class.items():
            print(f"  Class {class_id}: {count} samples ({100*count/n_samples:.1f}%)")
        
        # Generate specs
        all_specs = []
        
        for class_id, n_class_samples in samples_per_class.items():
            if n_class_samples == 0:
                continue
            
            # Load defect templates for this class
            defect_templates = self.load_defect_templates(
                class_id=class_id,
                min_suitability=0.5
            )
            
            if len(defect_templates) == 0:
                print(f"Warning: No defect templates found for class {class_id}")
                continue
            
            # Generate specs for this class
            for i in range(n_class_samples):
                # Select defect template (cycle through available templates)
                defect_idx = i % len(defect_templates)
                defect_template = defect_templates[defect_idx]
                
                # Find compatible backgrounds
                compatible_bgs = self.match_templates(defect_template, max_backgrounds=10)
                
                if len(compatible_bgs) == 0:
                    print(f"Warning: No compatible backgrounds for defect {defect_template.defect_type}")
                    continue
                
                # Select background (cycle through compatible backgrounds)
                bg_idx = i % len(compatible_bgs)
                bg_template, compatibility = compatible_bgs[bg_idx]
                
                # Create augmentation spec
                aug_id = f"aug_c{class_id}_{len(all_specs):05d}"
                
                spec = AugmentationSpec(
                    aug_id=aug_id,
                    background_template=bg_template,
                    defect_template=defect_template,
                    compatibility_score=compatibility,
                    generation_params={
                        'class_id': class_id,
                        'defect_type': defect_template.defect_type,
                        'background_type': bg_template.background_type,
                        'defect_suitability': defect_template.suitability_score,
                        'background_stability': bg_template.stability_score
                    }
                )
                
                all_specs.append(spec)
        
        print(f"\nCreated {len(all_specs)} augmentation specifications")
        return all_specs
    
    def generate_controlnet_hint(self,
                                defect_mask: np.ndarray,
                                background_patch: np.ndarray) -> np.ndarray:
        """
        Generate multi-channel ControlNet hint image.
        다중 채널 ControlNet 힌트 이미지 생성
        
        Hint format (3 channels):
        - R: Defect region (binary mask)
        - G: Edge information (Canny edges)
        - B: Texture information (gradient magnitude)
        
        Args:
            defect_mask: Binary mask of defect (H, W)
            background_patch: RGB background image (H, W, 3)
            
        Returns:
            Multi-channel hint image (H, W, 3)
        """
        h, w = defect_mask.shape
        hint = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Channel 0 (R): Defect region
        hint[:, :, 0] = (defect_mask * 255).astype(np.uint8)
        
        # Channel 1 (G): Edges from background
        gray = cv2.cvtColor(background_patch, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        hint[:, :, 1] = edges
        
        # Channel 2 (B): Texture (gradient magnitude)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)
        hint[:, :, 2] = grad_mag
        
        return hint
    
    def generate_sample(self, spec: AugmentationSpec) -> Dict:
        """
        Generate one augmented sample.
        증강 샘플 하나 생성
        
        This creates the input files for ControlNet generation:
        - Background image
        - Defect mask
        - Multi-channel hint
        - Metadata
        
        Args:
            spec: AugmentationSpec defining what to generate
            
        Returns:
            Dictionary with paths to generated files
        """
        # Load background patch
        bg_patch = cv2.imread(str(spec.background_template.patch_path))
        bg_patch = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2RGB)
        
        # Load defect mask
        defect_mask = cv2.imread(str(spec.defect_template.mask_path), cv2.IMREAD_GRAYSCALE)
        defect_mask = (defect_mask > 0).astype(np.uint8)
        
        # Resize if needed (ensure same size)
        if bg_patch.shape[:2] != defect_mask.shape:
            h, w = bg_patch.shape[:2]
            defect_mask = cv2.resize(defect_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Generate ControlNet hint
        hint = self.generate_controlnet_hint(defect_mask, bg_patch)
        
        # Save files
        output_image_path = self.output_dir / "images" / f"{spec.aug_id}.png"
        output_mask_path = self.output_dir / "masks" / f"{spec.aug_id}.png"
        output_hint_path = self.output_dir / "hints" / f"{spec.aug_id}.png"
        
        # Background as input
        bg_bgr = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_image_path), bg_bgr)
        
        # Mask
        cv2.imwrite(str(output_mask_path), defect_mask * 255)
        
        # Hint
        hint_bgr = cv2.cvtColor(hint, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_hint_path), hint_bgr)
        
        # Return metadata
        return {
            'aug_id': spec.aug_id,
            'image_path': str(output_image_path),
            'mask_path': str(output_mask_path),
            'hint_path': str(output_hint_path),
            'class_id': spec.generation_params['class_id'],
            'defect_type': spec.generation_params['defect_type'],
            'background_type': spec.generation_params['background_type'],
            'compatibility_score': spec.compatibility_score,
            'defect_suitability': spec.generation_params['defect_suitability'],
            'background_stability': spec.generation_params['background_stability']
        }
    
    def generate_batch(self,
                      n_samples: int,
                      class_distribution: Optional[Dict[int, float]] = None) -> pd.DataFrame:
        """
        Generate a batch of augmented samples.
        증강 샘플 배치 생성
        
        Args:
            n_samples: Number of samples to generate
            class_distribution: Optional class distribution
            
        Returns:
            DataFrame with augmentation metadata
        """
        # Create specs
        specs = self.create_augmentation_specs(n_samples, class_distribution)
        
        # Generate samples
        print(f"\nGenerating {len(specs)} augmented samples...")
        results = []
        
        for spec in tqdm(specs, desc="Generating"):
            try:
                result = self.generate_sample(spec)
                results.append(result)
            except Exception as e:
                print(f"Error generating {spec.aug_id}: {e}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save metadata
        metadata_path = self.output_dir / "augmentation_metadata.csv"
        results_df.to_csv(metadata_path, index=False)
        
        print(f"\nGeneration complete!")
        print(f"  Samples generated: {len(results_df)}")
        print(f"  Metadata saved to: {metadata_path}")
        
        # Print statistics
        print(f"\nClass distribution:")
        for class_id in sorted(results_df['class_id'].unique()):
            count = (results_df['class_id'] == class_id).sum()
            pct = 100.0 * count / len(results_df)
            print(f"  Class {class_id}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nQuality metrics:")
        print(f"  Mean compatibility:  {results_df['compatibility_score'].mean():.3f}")
        print(f"  Mean suitability:    {results_df['defect_suitability'].mean():.3f}")
        print(f"  Mean stability:      {results_df['background_stability'].mean():.3f}")
        
        return results_df


def main():
    """
    Example usage of AugmentationGenerator.
    """
    from .background_library import BackgroundLibrary
    
    # Paths
    project_root = Path(r"D:\project\severstal-steel-defect-detection")
    bg_metadata = project_root / "data" / "processed" / "background_patches" / "background_metadata.csv"
    defect_metadata = project_root / "data" / "processed" / "roi_patches" / "roi_metadata.csv"
    output_dir = project_root / "data" / "augmented"
    
    # Check paths
    if not bg_metadata.exists():
        print(f"Background metadata not found: {bg_metadata}")
        print("Run background extraction first:")
        print("  python scripts/run_background_extraction.py")
        return
    
    if not defect_metadata.exists():
        print(f"Defect metadata not found: {defect_metadata}")
        print("Run ROI extraction first")
        return
    
    # Initialize
    print("Loading background library...")
    bg_library = BackgroundLibrary(bg_metadata)
    
    print("\nInitializing augmentation generator...")
    generator = AugmentationGenerator(
        background_library=bg_library,
        defect_metadata_path=defect_metadata,
        output_dir=output_dir,
        min_compatibility=0.5
    )
    
    # Generate samples
    # Example: 1000 samples with class distribution matching original dataset
    class_dist = {
        1: 0.25,  # 25% class 1
        2: 0.25,  # 25% class 2
        3: 0.35,  # 35% class 3
        4: 0.15   # 15% class 4
    }
    
    results_df = generator.generate_batch(
        n_samples=1000,
        class_distribution=class_dist
    )
    
    print("\nAugmentation generation complete!")
    print(f"Next step: Train ControlNet using generated hints")


if __name__ == '__main__':
    main()
