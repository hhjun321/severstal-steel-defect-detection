#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug ROI Detection
ROI 검출 문제를 진단하기 위한 스크립트
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

def find_clean_images():
    """Find clean images without defects"""
    all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
    train_df = pd.read_csv(train_csv_path)
    defect_images = set(train_df['ImageId'].unique())
    clean_images = list(all_images - defect_images)
    return clean_images

def debug_analyze_image(image_id, roi_size=512, grid_size=64):
    """Analyze image with detailed debugging output"""
    print("\n" + "="*80)
    print(f"DEBUG ANALYSIS: {image_id}")
    print("="*80)
    
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Could not load image: {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    print(f"\n1. Image Info:")
    print(f"   - Size: {W}x{H}")
    print(f"   - Shape: {img.shape}")
    
    # Analyze background
    print(f"\n2. Initializing BackgroundAnalyzer:")
    print(f"   - grid_size: {grid_size}")
    print(f"   - variance_threshold: 100.0")
    print(f"   - edge_threshold: 0.3")
    
    analyzer = BackgroundAnalyzer(grid_size=grid_size, variance_threshold=100.0, edge_threshold=0.3)
    analysis = analyzer.analyze_image(img)
    
    bg_map = analysis['background_map']
    stability_map = analysis['stability_map']
    grid_h, grid_w = analysis['grid_shape']
    
    print(f"\n3. Background Analysis Results:")
    print(f"   - Grid shape: {grid_h}x{grid_w}")
    print(f"   - Background map shape: {bg_map.shape}")
    
    # Count background types
    unique_types, counts = np.unique(bg_map, return_counts=True)
    print(f"\n4. Background Type Distribution:")
    for bg_type, count in zip(unique_types, counts):
        percentage = (count / bg_map.size) * 100
        print(f"   - {bg_type}: {count} cells ({percentage:.1f}%)")
    
    # Check each background type for ROI extraction
    print(f"\n5. ROI Extraction Attempts:")
    BG_TYPE_ORDER = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']
    
    rois = []
    for target_type in BG_TYPE_ORDER:
        print(f"\n   Checking '{target_type}':")
        matches = np.argwhere(bg_map == target_type)
        print(f"   - Found {len(matches)} cells of this type")
        
        if len(matches) == 0:
            print(f"   - SKIP: No cells found")
            continue
        
        # Show stability scores
        stabilities = [stability_map[m[0], m[1]] for m in matches]
        print(f"   - Stability scores: min={min(stabilities):.3f}, max={max(stabilities):.3f}, avg={np.mean(stabilities):.3f}")
        
        best_idx = np.argmax(stabilities)
        gi, gj = matches[best_idx]
        best_stability = stabilities[best_idx]
        
        print(f"   - Best cell: grid[{gi},{gj}] with stability={best_stability:.3f}")
        
        # Calculate ROI position
        y_center = gi * grid_size + grid_size // 2
        x_center = gj * grid_size + grid_size // 2
        
        y_roi = max(0, min(H - roi_size, y_center - roi_size // 2))
        x_roi = max(0, min(W - roi_size, x_center - roi_size // 2))
        
        print(f"   - Center: ({x_center}, {y_center})")
        print(f"   - ROI position: ({x_roi}, {y_roi}) to ({x_roi+roi_size}, {y_roi+roi_size})")
        
        # Check bounds
        if y_roi + roi_size <= H and x_roi + roi_size <= W:
            rois.append({
                'x': x_roi,
                'y': y_roi,
                'type': target_type,
                'score': float(best_stability)
            })
            print(f"   - ✓ ROI ADDED (total: {len(rois)})")
        else:
            print(f"   - ✗ REJECTED: Out of bounds (H={H}, W={W})")
        
        if len(rois) >= 5:
            print(f"   - Reached maximum 5 ROIs, stopping")
            break
    
    print(f"\n6. FINAL RESULT:")
    print(f"   - Total ROIs extracted: {len(rois)}")
    
    if len(rois) > 0:
        print(f"\n   ROI Details:")
        for i, roi in enumerate(rois, 1):
            print(f"   [{i}] Type: {roi['type']}, Position: ({roi['x']}, {roi['y']}), Score: {roi['score']:.3f}")
    
    print("\n" + "="*80)
    return rois

def main():
    print("\n" + "="*80)
    print("ROI Detection Debug Tool")
    print("="*80)
    
    # Find clean images
    clean_images = find_clean_images()
    print(f"\nTotal clean images found: {len(clean_images)}")
    
    # Test first 3 images
    print("\nTesting first 3 clean images...")
    
    for i, image_id in enumerate(clean_images[:3], 1):
        rois = debug_analyze_image(image_id)
        
        if i < 3:
            input("\nPress Enter to continue to next image...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
