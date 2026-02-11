#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI Size Analysis
Analyze image dimensions and defect sizes to determine optimal ROI sizes
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Setup paths
project_root = Path(__file__).parent
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "roi_analysis"

# Add src to path
sys.path.insert(0, str(project_root / "src"))
from utils.rle_utils import rle_decode

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

def analyze_image_dimensions():
    """Analyze dimensions of all training images"""
    print("\n" + "="*80)
    print("1. IMAGE DIMENSIONS ANALYSIS")
    print("="*80)
    
    dimensions = []
    image_files = list(train_images_dir.glob("*.jpg"))
    
    print(f"\nAnalyzing {len(image_files)} images...")
    
    for img_path in image_files[:100]:  # Sample first 100 for speed
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            dimensions.append({'width': w, 'height': h, 'image_id': img_path.name})
    
    df_dims = pd.DataFrame(dimensions)
    
    print(f"\nImage Dimension Statistics:")
    print(f"  Width:  min={df_dims['width'].min()}, max={df_dims['width'].max()}, mean={df_dims['width'].mean():.0f}")
    print(f"  Height: min={df_dims['height'].min()}, max={df_dims['height'].max()}, mean={df_dims['height'].mean():.0f}")
    print(f"\nUnique dimensions:")
    unique_dims = df_dims.groupby(['width', 'height']).size().reset_index(name='count')
    for _, row in unique_dims.iterrows():
        print(f"  {int(row['width'])}x{int(row['height'])}: {row['count']} images")
    
    return df_dims

def analyze_defect_sizes():
    """Analyze bounding box sizes of defects"""
    print("\n" + "="*80)
    print("2. DEFECT SIZE ANALYSIS")
    print("="*80)
    
    train_df = pd.read_csv(train_csv_path)
    
    # Filter out rows with no defects
    defect_df = train_df[train_df['EncodedPixels'].notna()].copy()
    
    print(f"\nTotal defect annotations: {len(defect_df)}")
    
    defect_sizes = []
    
    # Sample for analysis
    sample_size = min(500, len(defect_df))
    sample_df = defect_df.sample(n=sample_size, random_state=42)
    
    print(f"Analyzing {sample_size} defect samples...")
    
    for idx, row in sample_df.iterrows():
        image_id = row['ImageId']
        rle = row['EncodedPixels']
        
        # Load image to get dimensions
        img_path = train_images_dir / image_id
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Decode RLE to get mask
        mask = rle_decode(rle, (h, w))
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            bbox_width = cmax - cmin + 1
            bbox_height = rmax - rmin + 1
            bbox_area = bbox_width * bbox_height
            defect_area = np.sum(mask)
            
            defect_sizes.append({
                'width': bbox_width,
                'height': bbox_height,
                'area': bbox_area,
                'defect_area': defect_area,
                'class_id': row['ClassId']
            })
    
    df_defects = pd.DataFrame(defect_sizes)
    
    print(f"\nDefect Bounding Box Statistics:")
    print(f"  Width:  min={df_defects['width'].min()}, max={df_defects['width'].max()}, mean={df_defects['width'].mean():.0f}, median={df_defects['width'].median():.0f}")
    print(f"  Height: min={df_defects['height'].min()}, max={df_defects['height'].max()}, mean={df_defects['height'].mean():.0f}, median={df_defects['height'].median():.0f}")
    print(f"  Area:   min={df_defects['area'].min()}, max={df_defects['area'].max()}, mean={df_defects['area'].mean():.0f}, median={df_defects['area'].median():.0f}")
    
    print(f"\nDefect Size Distribution (by class):")
    for class_id in sorted(df_defects['class_id'].unique()):
        class_df = df_defects[df_defects['class_id'] == class_id]
        print(f"  Class {class_id}:")
        print(f"    Width:  mean={class_df['width'].mean():.0f}, median={class_df['width'].median():.0f}")
        print(f"    Height: mean={class_df['height'].mean():.0f}, median={class_df['height'].median():.0f}")
    
    # Percentiles
    print(f"\nDefect Size Percentiles:")
    for percentile in [25, 50, 75, 90, 95, 99]:
        w_p = df_defects['width'].quantile(percentile/100)
        h_p = df_defects['height'].quantile(percentile/100)
        a_p = df_defects['area'].quantile(percentile/100)
        print(f"  {percentile}th: width={w_p:.0f}, height={h_p:.0f}, area={a_p:.0f}")
    
    return df_defects

def recommend_roi_sizes(df_dims, df_defects):
    """Recommend optimal ROI sizes based on analysis"""
    print("\n" + "="*80)
    print("3. ROI SIZE RECOMMENDATIONS")
    print("="*80)
    
    # Image constraints
    min_img_height = df_dims['height'].min()
    min_img_width = df_dims['width'].min()
    
    print(f"\nImage Constraints:")
    print(f"  Minimum image size: {min_img_width}x{min_img_height}")
    print(f"  ROI must fit within: {int(min_img_width * 0.8)}x{int(min_img_height * 0.8)} (80% rule)")
    
    # Defect size analysis
    print(f"\nDefect Coverage Analysis:")
    
    # Test different ROI sizes
    test_sizes = [128, 192, 256, 384, 512]
    
    for roi_size in test_sizes:
        if roi_size > min_img_height or roi_size > min_img_width:
            print(f"\n  ROI {roi_size}x{roi_size}: TOO LARGE (exceeds image dimensions)")
            continue
        
        # How many defects fit within this ROI size?
        fits_completely = ((df_defects['width'] <= roi_size) & 
                          (df_defects['height'] <= roi_size)).sum()
        
        coverage_rate = fits_completely / len(df_defects) * 100
        
        # Margin analysis
        avg_margin_w = roi_size - df_defects['width'].mean()
        avg_margin_h = roi_size - df_defects['height'].mean()
        
        print(f"\n  ROI {roi_size}x{roi_size}:")
        print(f"    Defects fitting completely: {fits_completely}/{len(df_defects)} ({coverage_rate:.1f}%)")
        print(f"    Average margin: width={avg_margin_w:.0f}px, height={avg_margin_h:.0f}px")
        
        # Context ratio
        median_defect_area = df_defects['area'].median()
        roi_area = roi_size * roi_size
        context_ratio = roi_area / median_defect_area
        print(f"    Context ratio: {context_ratio:.1f}x (ROI area / median defect area)")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print("="*80)
    
    print(f"\n1. CURRENT STRATEGY (Dynamic Sizing):")
    print(f"   - Based on image height: {int(min_img_height * 0.8)}px (80% of {min_img_height}px)")
    print(f"   - Pros: Always fits in image")
    print(f"   - Cons: May be too small for larger defects or augmentation")
    
    print(f"\n2. FIXED SIZE OPTIONS:")
    
    # Option A: Conservative (covers 95% of defects)
    size_95 = int(np.ceil(max(
        df_defects['width'].quantile(0.95),
        df_defects['height'].quantile(0.95)
    ) / 32) * 32)  # Round to nearest 32
    print(f"\n   Option A (Conservative): {size_95}x{size_95}")
    print(f"   - Covers 95% of defects completely")
    print(f"   - Good for: Preserving full defect context")
    
    # Option B: Balanced (covers 75% of defects, optimal context)
    size_75 = int(np.ceil(max(
        df_defects['width'].quantile(0.75),
        df_defects['height'].quantile(0.75)
    ) / 32) * 32)
    print(f"\n   Option B (Balanced): {size_75}x{size_75}")
    print(f"   - Covers 75% of defects completely")
    print(f"   - Good for: Balance between defect and context")
    
    # Option C: Standard
    print(f"\n   Option C (Standard): 256x256")
    print(f"   - Industry standard for defect detection")
    print(f"   - Good for: Model training and inference speed")
    
    print(f"\n3. MULTI-SCALE STRATEGY:")
    print(f"   - Small ROIs (128x128): For small defects, high resolution details")
    print(f"   - Medium ROIs (256x256): Standard size, good balance")
    print(f"   - Large ROIs (512x512): For large defects, more context")
    
    print(f"\n4. ADAPTIVE STRATEGY (RECOMMENDED):")
    print(f"   - Defect-aware sizing: ROI size = max(defect_bbox) * 1.5 to 2.0")
    print(f"   - Ensures defect fits with adequate context")
    print(f"   - Minimum: 128x128, Maximum: 512x512")

def create_visualizations(df_dims, df_defects):
    """Create visualization plots"""
    print("\n" + "="*80)
    print("4. CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Image dimension distribution
    ax1 = axes[0, 0]
    ax1.scatter(df_dims['width'], df_dims['height'], alpha=0.5)
    ax1.set_xlabel('Width (pixels)', fontsize=12)
    ax1.set_ylabel('Height (pixels)', fontsize=12)
    ax1.set_title('Image Dimensions Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add ROI size references
    for roi_size in [128, 256, 512]:
        ax1.axhline(y=roi_size, color='red', linestyle='--', alpha=0.3, label=f'ROI {roi_size}')
        ax1.axvline(x=roi_size, color='red', linestyle='--', alpha=0.3)
    ax1.legend()
    
    # 2. Defect size distribution
    ax2 = axes[0, 1]
    ax2.scatter(df_defects['width'], df_defects['height'], 
               c=df_defects['class_id'], cmap='tab10', alpha=0.6)
    ax2.set_xlabel('Defect Width (pixels)', fontsize=12)
    ax2.set_ylabel('Defect Height (pixels)', fontsize=12)
    ax2.set_title('Defect Bounding Box Sizes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add ROI size references
    for roi_size in [128, 256, 512]:
        ax2.axhline(y=roi_size, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.axvline(x=roi_size, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.text(roi_size + 10, roi_size + 10, f'{roi_size}x{roi_size}', 
                color='red', fontweight='bold')
    
    # 3. Defect width histogram
    ax3 = axes[1, 0]
    ax3.hist(df_defects['width'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Defect Width (pixels)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Defect Width Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add vertical lines for ROI sizes
    for roi_size in [128, 256, 512]:
        ax3.axvline(x=roi_size, color='red', linestyle='--', linewidth=2, label=f'ROI {roi_size}')
    ax3.legend()
    
    # 4. Coverage analysis
    ax4 = axes[1, 1]
    roi_sizes = np.arange(64, 640, 32)
    coverage_rates = []
    
    for roi_size in roi_sizes:
        fits = ((df_defects['width'] <= roi_size) & 
               (df_defects['height'] <= roi_size)).sum()
        coverage_rates.append(fits / len(df_defects) * 100)
    
    ax4.plot(roi_sizes, coverage_rates, linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('ROI Size (pixels)', fontsize=12)
    ax4.set_ylabel('Defect Coverage (%)', fontsize=12)
    ax4.set_title('ROI Size vs Defect Coverage', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=95, color='green', linestyle='--', label='95% coverage')
    ax4.axhline(y=75, color='orange', linestyle='--', label='75% coverage')
    ax4.axhline(y=50, color='red', linestyle='--', label='50% coverage')
    
    # Highlight standard sizes
    for roi_size in [128, 256, 512]:
        idx = np.argmin(np.abs(roi_sizes - roi_size))
        ax4.plot(roi_size, coverage_rates[idx], 'ro', markersize=10)
        ax4.text(roi_size, coverage_rates[idx] + 3, f'{roi_size}', 
                ha='center', fontweight='bold', color='red')
    
    ax4.legend()
    
    plt.tight_layout()
    
    output_path = output_dir / "roi_size_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()

def main():
    print("\n" + "="*80)
    print("ROI SIZE RESEARCH AND ANALYSIS")
    print("="*80)
    
    # Analyze image dimensions
    df_dims = analyze_image_dimensions()
    
    # Analyze defect sizes
    df_defects = analyze_defect_sizes()
    
    # Generate recommendations
    recommend_roi_sizes(df_dims, df_defects)
    
    # Create visualizations
    create_visualizations(df_dims, df_defects)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
