"""
Background Library - Indexing and Search System
배경 라이브러리 - 인덱싱 및 검색 시스템

This module provides indexing and search capabilities for the extracted
background library, enabling efficient background selection during augmentation.

추출된 배경 라이브러리에 대한 인덱싱 및 검색 기능을 제공하여
증강 중 효율적인 배경 선택을 가능하게 합니다.

Key Features:
- Index backgrounds by type, stability, and characteristics
- Fast search by background type
- Compatibility matching with defect templates
- Stratified sampling for diverse augmentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class BackgroundTemplate:
    """
    A background template from the library.
    라이브러리의 배경 템플릿
    """
    image_id: str
    roi_index: int
    patch_path: Path
    background_type: str
    stability_score: float
    num_defects_in_image: int
    

class BackgroundLibrary:
    """
    Background library with indexing and search capabilities.
    인덱싱 및 검색 기능이 있는 배경 라이브러리
    
    Provides efficient access to background templates for augmentation.
    """
    
    # Compatibility rules: which defect types work best with which background types
    # Perfect compatibility: 1.0, Good: 0.8, Acceptable: 0.5, Poor: 0.2
    COMPATIBILITY_MATRIX = {
        # Defect type: {background_type: compatibility_score}
        'compact_blob': {
            'smooth': 1.0,           # Perfect for isolated blobs
            'vertical_stripe': 0.8,   # Good contrast
            'horizontal_stripe': 0.8, # Good contrast
            'textured': 0.5,          # May blend in
            'complex_pattern': 0.2    # Poor visibility
        },
        'linear_scratch': {
            'smooth': 0.8,            # Good visibility
            'vertical_stripe': 1.0,   # Perfect if horizontal scratch
            'horizontal_stripe': 1.0, # Perfect if vertical scratch
            'textured': 0.5,          # May blend
            'complex_pattern': 0.2    # Poor contrast
        },
        'scattered_defects': {
            'smooth': 1.0,            # Perfect visibility
            'vertical_stripe': 0.8,   # Good
            'horizontal_stripe': 0.8, # Good
            'textured': 0.5,          # Acceptable
            'complex_pattern': 0.2    # Poor
        },
        'elongated_region': {
            'smooth': 0.8,            # Good
            'vertical_stripe': 1.0,   # Perfect contrast
            'horizontal_stripe': 1.0, # Perfect contrast
            'textured': 0.5,          # Acceptable
            'complex_pattern': 0.2    # Poor
        }
    }
    
    def __init__(self, metadata_csv_path: Path):
        """
        Initialize background library from metadata CSV.
        
        Args:
            metadata_csv_path: Path to background_metadata.csv
        """
        self.metadata_path = metadata_csv_path
        self.df = pd.read_csv(metadata_csv_path)
        
        # Build indexes
        self._build_indexes()
        
        print(f"Loaded background library with {len(self.df)} templates")
        print(f"Background type distribution:")
        for bg_type, count in self.df['background_type'].value_counts().items():
            print(f"  {bg_type:20s}: {count:4d}")
    
    def _build_indexes(self):
        """
        Build indexes for fast lookup.
        빠른 조회를 위한 인덱스 구축
        """
        # Index by background type
        self.type_index = {}
        for bg_type in self.df['background_type'].unique():
            mask = self.df['background_type'] == bg_type
            self.type_index[bg_type] = self.df[mask].index.tolist()
        
        # Index by stability tier
        self.stability_index = {
            'high': self.df[self.df['stability_score'] >= 0.8].index.tolist(),
            'medium': self.df[(self.df['stability_score'] >= 0.6) & 
                             (self.df['stability_score'] < 0.8)].index.tolist(),
            'low': self.df[self.df['stability_score'] < 0.6].index.tolist()
        }
    
    def get_by_type(self, 
                   background_type: str,
                   min_stability: float = 0.6,
                   max_results: Optional[int] = None) -> List[BackgroundTemplate]:
        """
        Get backgrounds by type.
        타입별 배경 가져오기
        
        Args:
            background_type: Type of background to retrieve
            min_stability: Minimum stability score
            max_results: Maximum number of results (None = all)
            
        Returns:
            List of BackgroundTemplate objects
        """
        if background_type not in self.type_index:
            return []
        
        # Filter by type and stability
        indices = self.type_index[background_type]
        candidates = self.df.iloc[indices]
        candidates = candidates[candidates['stability_score'] >= min_stability]
        
        # Sort by stability (best first)
        candidates = candidates.sort_values('stability_score', ascending=False)
        
        # Limit results
        if max_results is not None:
            candidates = candidates.head(max_results)
        
        # Convert to BackgroundTemplate objects
        templates = []
        for _, row in candidates.iterrows():
            template = BackgroundTemplate(
                image_id=row['image_id'],
                roi_index=row['roi_index'],
                patch_path=Path(row['patch_path']),
                background_type=row['background_type'],
                stability_score=row['stability_score'],
                num_defects_in_image=row['num_defects_in_image']
            )
            templates.append(template)
        
        return templates
    
    def get_compatible_backgrounds(self,
                                   defect_type: str,
                                   min_compatibility: float = 0.5,
                                   min_stability: float = 0.6,
                                   max_results: Optional[int] = None) -> List[Tuple[BackgroundTemplate, float]]:
        """
        Get backgrounds compatible with a defect type.
        결함 유형과 호환되는 배경 가져오기
        
        Args:
            defect_type: Type of defect to match
            min_compatibility: Minimum compatibility score
            min_stability: Minimum stability score
            max_results: Maximum results to return
            
        Returns:
            List of (BackgroundTemplate, compatibility_score) tuples
        """
        if defect_type not in self.COMPATIBILITY_MATRIX:
            print(f"Warning: Unknown defect type '{defect_type}', using defaults")
            defect_type = 'compact_blob'
        
        compatibility_scores = self.COMPATIBILITY_MATRIX[defect_type]
        
        # Collect compatible backgrounds
        compatible = []
        
        for bg_type, compat_score in compatibility_scores.items():
            if compat_score < min_compatibility:
                continue
            
            # Get backgrounds of this type
            templates = self.get_by_type(
                bg_type,
                min_stability=min_stability,
                max_results=None
            )
            
            # Add with compatibility score
            for template in templates:
                compatible.append((template, compat_score))
        
        # Sort by compatibility * stability (combined quality score)
        compatible.sort(
            key=lambda x: x[1] * x[0].stability_score,
            reverse=True
        )
        
        # Limit results
        if max_results is not None:
            compatible = compatible[:max_results]
        
        return compatible
    
    def sample_diverse(self,
                      n_samples: int,
                      min_stability: float = 0.6,
                      ensure_type_diversity: bool = True) -> List[BackgroundTemplate]:
        """
        Sample diverse backgrounds from library.
        라이브러리에서 다양한 배경 샘플링
        
        Uses stratified sampling to ensure diversity across background types.
        
        Args:
            n_samples: Number of samples to draw
            min_stability: Minimum stability score
            ensure_type_diversity: Ensure at least one of each type if possible
            
        Returns:
            List of BackgroundTemplate objects
        """
        # Filter by stability
        candidates = self.df[self.df['stability_score'] >= min_stability]
        
        if len(candidates) == 0:
            print("Warning: No backgrounds meet stability requirement")
            return []
        
        if ensure_type_diversity:
            # Stratified sampling: one from each type, then random
            samples = []
            types_available = candidates['background_type'].unique()
            
            # First, get one of each type
            for bg_type in types_available:
                type_candidates = candidates[candidates['background_type'] == bg_type]
                if len(type_candidates) > 0:
                    # Pick the best one (highest stability)
                    best = type_candidates.nlargest(1, 'stability_score')
                    samples.append(best)
                    
                    if len(samples) >= n_samples:
                        break
            
            # If we need more, sample randomly from remaining
            if len(samples) < n_samples:
                remaining_indices = set(candidates.index) - set([s.index[0] for s in samples])
                remaining = candidates.loc[list(remaining_indices)]
                
                n_additional = min(n_samples - len(samples), len(remaining))
                if n_additional > 0:
                    additional = remaining.sample(n=n_additional)
                    samples.append(additional)
            
            # Combine all samples
            if len(samples) > 0:
                sampled_df = pd.concat(samples).head(n_samples)
            else:
                sampled_df = pd.DataFrame()
        else:
            # Simple random sampling
            n_to_sample = min(n_samples, len(candidates))
            sampled_df = candidates.sample(n=n_to_sample)
        
        # Convert to BackgroundTemplate objects
        templates = []
        for _, row in sampled_df.iterrows():
            template = BackgroundTemplate(
                image_id=row['image_id'],
                roi_index=row['roi_index'],
                patch_path=Path(row['patch_path']),
                background_type=row['background_type'],
                stability_score=row['stability_score'],
                num_defects_in_image=row['num_defects_in_image']
            )
            templates.append(template)
        
        return templates
    
    def get_statistics(self) -> Dict:
        """
        Get library statistics.
        라이브러리 통계 가져오기
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_backgrounds': len(self.df),
            'unique_source_images': self.df['image_id'].nunique(),
            'background_types': dict(self.df['background_type'].value_counts()),
            'stability_distribution': {
                'high (≥0.8)': len(self.stability_index['high']),
                'medium (0.6-0.8)': len(self.stability_index['medium']),
                'low (<0.6)': len(self.stability_index['low'])
            },
            'stability_stats': {
                'mean': float(self.df['stability_score'].mean()),
                'std': float(self.df['stability_score'].std()),
                'min': float(self.df['stability_score'].min()),
                'max': float(self.df['stability_score'].max()),
                'median': float(self.df['stability_score'].median())
            }
        }
        return stats
    
    def print_statistics(self):
        """
        Print library statistics in a readable format.
        통계를 읽기 쉬운 형식으로 출력
        """
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("BACKGROUND LIBRARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal Backgrounds: {stats['total_backgrounds']}")
        print(f"Unique Source Images: {stats['unique_source_images']}")
        
        print(f"\nBackground Type Distribution:")
        for bg_type, count in stats['background_types'].items():
            pct = 100.0 * count / stats['total_backgrounds']
            print(f"  {bg_type:20s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nStability Distribution:")
        for tier, count in stats['stability_distribution'].items():
            pct = 100.0 * count / stats['total_backgrounds']
            print(f"  {tier:20s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nStability Statistics:")
        for key, value in stats['stability_stats'].items():
            print(f"  {key.capitalize():10s}: {value:.3f}")
        
        print("="*60 + "\n")


def demo_search():
    """
    Demonstrate background library search capabilities.
    배경 라이브러리 검색 기능 시연
    """
    # Load library
    project_root = Path(r"D:\project\severstal-steel-defect-detection")
    metadata_path = project_root / "data" / "processed" / "background_patches" / "background_metadata.csv"
    
    if not metadata_path.exists():
        print(f"Background metadata not found at {metadata_path}")
        print("Please run background extraction first:")
        print("  python scripts/run_background_extraction.py")
        return
    
    library = BackgroundLibrary(metadata_path)
    
    # Print statistics
    library.print_statistics()
    
    # Example 1: Get smooth backgrounds
    print("\n" + "="*60)
    print("EXAMPLE 1: Get smooth backgrounds (best for compact blobs)")
    print("="*60)
    smooth_bgs = library.get_by_type('smooth', min_stability=0.7, max_results=5)
    print(f"Found {len(smooth_bgs)} smooth backgrounds with stability ≥ 0.7")
    for i, bg in enumerate(smooth_bgs[:3], 1):
        print(f"  {i}. {bg.image_id} - stability: {bg.stability_score:.3f}")
    
    # Example 2: Get backgrounds compatible with linear scratches
    print("\n" + "="*60)
    print("EXAMPLE 2: Get backgrounds compatible with linear scratches")
    print("="*60)
    compatible = library.get_compatible_backgrounds(
        'linear_scratch',
        min_compatibility=0.8,
        min_stability=0.6,
        max_results=10
    )
    print(f"Found {len(compatible)} highly compatible backgrounds")
    for i, (bg, compat) in enumerate(compatible[:5], 1):
        print(f"  {i}. {bg.background_type:20s} - compat: {compat:.2f}, stability: {bg.stability_score:.3f}")
    
    # Example 3: Sample diverse backgrounds
    print("\n" + "="*60)
    print("EXAMPLE 3: Sample 10 diverse backgrounds")
    print("="*60)
    diverse = library.sample_diverse(n_samples=10, min_stability=0.6)
    print(f"Sampled {len(diverse)} diverse backgrounds")
    type_counts = {}
    for bg in diverse:
        type_counts[bg.background_type] = type_counts.get(bg.background_type, 0) + 1
    for bg_type, count in sorted(type_counts.items()):
        print(f"  {bg_type:20s}: {count}")


if __name__ == '__main__':
    demo_search()
