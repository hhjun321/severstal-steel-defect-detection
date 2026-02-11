#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced ROI Size Strategy
더 정교한 ROI 크기 결정 전략 구현
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).parent
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"

# Add src to path
sys.path.insert(0, str(project_root / "src"))
from analysis.background_characterization import BackgroundAnalyzer

class AdvancedROISizer:
    """
    Advanced ROI sizing strategy that considers:
    - Image dimensions and aspect ratio
    - Background type and uniformity
    - Grid cell distribution
    - Multiple ROI sizes per image
    """
    
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.analyzer = BackgroundAnalyzer(
            grid_size=grid_size,
            variance_threshold=100.0,
            edge_threshold=0.3
        )
    
    def analyze_image_characteristics(self, img):
        """Analyze image to determine optimal ROI characteristics"""
        H, W = img.shape[:2]
        
        # Aspect ratio analysis
        aspect_ratio = W / H
        
        # Background analysis
        analysis = self.analyzer.analyze_image(img)
        bg_map = analysis['background_map']
        stability_map = analysis['stability_map']
        grid_h, grid_w = analysis['grid_shape']
        
        # Count background types
        unique_types, counts = np.unique(bg_map, return_counts=True)
        bg_diversity = len(unique_types)
        
        # Calculate average stability
        avg_stability = np.mean(stability_map)
        
        # Estimate usable area (non-black regions)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        usable_ratio = np.sum(gray > 20) / gray.size
        
        return {
            'width': W,
            'height': H,
            'aspect_ratio': aspect_ratio,
            'bg_diversity': bg_diversity,
            'avg_stability': avg_stability,
            'usable_ratio': usable_ratio,
            'grid_shape': (grid_h, grid_w),
            'analysis': analysis
        }
    
    def recommend_roi_sizes(self, characteristics):
        """
        Recommend multiple ROI sizes based on image characteristics
        
        Returns list of recommended sizes in priority order
        """
        H = characteristics['height']
        W = characteristics['width']
        aspect_ratio = characteristics['aspect_ratio']
        bg_diversity = characteristics['bg_diversity']
        grid_h, grid_w = characteristics['grid_shape']
        
        recommendations = []
        
        # Strategy 1: Maximum fitting square
        # Use 70-80% of smaller dimension to ensure good fit
        max_roi = int(min(H, W) * 0.75)
        max_roi = (max_roi // 32) * 32  # Round to nearest 32
        
        if max_roi >= 128:
            recommendations.append({
                'size': max_roi,
                'strategy': 'max_fit',
                'description': f'Maximum fitting square ({max_roi}x{max_roi})',
                'priority': 1,
                'coverage': 'Full height coverage'
            })
        
        # Strategy 2: Multi-scale approach based on grid
        # ROI size should ideally cover 3-4 grid cells for good context
        grid_based_roi = self.grid_size * 3
        grid_based_roi = (grid_based_roi // 32) * 32
        
        if grid_based_roi <= min(H, W) * 0.8 and grid_based_roi >= 128:
            if grid_based_roi not in [r['size'] for r in recommendations]:
                recommendations.append({
                    'size': grid_based_roi,
                    'strategy': 'grid_based',
                    'description': f'Grid-optimized ({grid_based_roi}x{grid_based_roi}, covers 3x3 cells)',
                    'priority': 2,
                    'coverage': 'Optimal context per ROI'
                })
        
        # Strategy 3: Standard sizes that fit
        standard_sizes = [512, 384, 256, 192, 128]
        for size in standard_sizes:
            if size <= min(H, W) * 0.8:
                if size not in [r['size'] for r in recommendations]:
                    recommendations.append({
                        'size': size,
                        'strategy': 'standard',
                        'description': f'Standard {size}x{size}',
                        'priority': 3,
                        'coverage': f'{(size/min(H,W)*100):.0f}% of smaller dimension'
                    })
                    break  # Only add one standard size
        
        # Strategy 4: Small detail-focused ROI
        # For high diversity backgrounds, add smaller ROI for details
        if bg_diversity >= 3:
            small_roi = 128
            if small_roi <= min(H, W) * 0.5:
                if small_roi not in [r['size'] for r in recommendations]:
                    recommendations.append({
                        'size': small_roi,
                        'strategy': 'detail',
                        'description': 'Small detail-focused (128x128)',
                        'priority': 4,
                        'coverage': 'High detail capture'
                    })
        
        # Strategy 5: Wide-aspect optimized (for panoramic images)
        if aspect_ratio > 4.0:  # Very wide image
            # Multiple smaller ROIs might work better
            wide_roi = int(H * 0.6)
            wide_roi = (wide_roi // 32) * 32
            
            if wide_roi >= 96 and wide_roi not in [r['size'] for r in recommendations]:
                recommendations.append({
                    'size': wide_roi,
                    'strategy': 'wide_optimized',
                    'description': f'Wide-image optimized ({wide_roi}x{wide_roi})',
                    'priority': 2,
                    'coverage': 'Multiple ROIs across width'
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations
    
    def estimate_roi_count(self, img_size, roi_size, bg_diversity):
        """Estimate how many non-overlapping ROIs can fit"""
        H, W = img_size
        
        # Theoretical maximum (without overlap prevention)
        max_rows = H // roi_size
        max_cols = W // roi_size
        theoretical_max = max_rows * max_cols
        
        # Practical estimate (considering overlap prevention and filtering)
        # With 3x3 grid span for overlap prevention: ~1/9 of theoretical
        practical_estimate = max(1, theoretical_max // 9)
        
        # Further reduce based on bg_diversity
        # More diversity = more potential ROIs of different types
        diversity_factor = min(1.0, bg_diversity / 5.0)
        practical_estimate = int(practical_estimate * diversity_factor)
        
        return {
            'theoretical_max': theoretical_max,
            'practical_estimate': max(1, practical_estimate),
            'grid_capacity': f"{max_rows}x{max_cols}"
        }

def analyze_and_recommend(image_id):
    """Analyze single image and provide recommendations"""
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    
    if img is None:
        return None
    
    sizer = AdvancedROISizer(grid_size=64)
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {image_id}")
    print(f"{'='*80}")
    
    # Analyze characteristics
    chars = sizer.analyze_image_characteristics(img)
    
    print(f"\nImage Characteristics:")
    print(f"  Dimensions: {chars['width']}x{chars['height']}")
    print(f"  Aspect Ratio: {chars['aspect_ratio']:.2f}:1")
    print(f"  Background Diversity: {chars['bg_diversity']} types")
    print(f"  Average Stability: {chars['avg_stability']:.3f}")
    print(f"  Usable Area: {chars['usable_ratio']*100:.1f}%")
    print(f"  Grid Shape: {chars['grid_shape'][0]}x{chars['grid_shape'][1]} cells")
    
    # Get recommendations
    recommendations = sizer.recommend_roi_sizes(chars)
    
    print(f"\nRecommended ROI Sizes (in priority order):")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  [{i}] {rec['size']}x{rec['size']} - {rec['strategy'].upper()}")
        print(f"      {rec['description']}")
        print(f"      Coverage: {rec['coverage']}")
        
        # Estimate ROI count
        estimate = sizer.estimate_roi_count(
            (chars['height'], chars['width']),
            rec['size'],
            chars['bg_diversity']
        )
        print(f"      Estimated ROIs: {estimate['practical_estimate']} "
              f"(grid capacity: {estimate['grid_capacity']})")
    
    return recommendations

def main():
    print("\n" + "="*80)
    print("ADVANCED ROI SIZE STRATEGY ANALYSIS")
    print("="*80)
    
    # Find clean images
    all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
    train_df = pd.read_csv(train_csv_path)
    defect_images = set(train_df['ImageId'].unique())
    clean_images = list(all_images - defect_images)
    
    print(f"\nTotal clean images: {len(clean_images)}")
    
    # Analyze sample images
    import random
    random.seed(42)
    samples = random.sample(clean_images, min(5, len(clean_images)))
    
    all_recommendations = {}
    
    for image_id in samples:
        recs = analyze_and_recommend(image_id)
        if recs:
            all_recommendations[image_id] = recs
    
    # Summary
    print(f"\n{'='*80}")
    print("STRATEGY SUMMARY")
    print(f"{'='*80}")
    
    # Collect all recommended sizes
    all_sizes = []
    for recs in all_recommendations.values():
        all_sizes.extend([r['size'] for r in recs])
    
    if all_sizes:
        from collections import Counter
        size_counts = Counter(all_sizes)
        
        print("\nMost Common Recommended Sizes:")
        for size, count in size_counts.most_common(5):
            print(f"  {size}x{size}: recommended for {count}/{len(samples)} images")
        
        print(f"\nRecommended Strategy:")
        print(f"  PRIMARY: Use {size_counts.most_common(1)[0][0]}x{size_counts.most_common(1)[0][0]} "
              f"as default (most versatile)")
        
        if len(size_counts) > 1:
            print(f"  SECONDARY: Use {size_counts.most_common(2)[1][0]}x{size_counts.most_common(2)[1][0]} "
                  f"as alternative")
        
        print(f"\n  ADAPTIVE APPROACH:")
        print(f"    - For each image, calculate optimal size as: min(H, W) * 0.75")
        print(f"    - Round to nearest 32 pixels")
        print(f"    - Clamp between 128 and 512")
        print(f"    - This ensures best fit for each image's unique characteristics")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
