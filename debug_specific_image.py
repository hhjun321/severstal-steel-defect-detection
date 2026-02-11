#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug specific image ROI detection
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).parent
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "debug"

sys.path.insert(0, str(project_root / "src"))
from analysis.background_characterization import BackgroundAnalyzer

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

def visualize_grid_analysis(image_id="b18d448a7.jpg", grid_size=64):
    """Visualize background analysis grid"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {image_id}")
    print(f"{'='*80}\n")
    
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    # Auto ROI size
    roi_size = min(int(min(H, W) * 0.8), 256)
    roi_size = max(roi_size, 128)
    
    print(f"Image size: {W}x{H}")
    print(f"ROI size: {roi_size}x{roi_size}")
    print(f"Grid size: {grid_size}x{grid_size}\n")
    
    # Analyze background
    analyzer = BackgroundAnalyzer(grid_size=grid_size, variance_threshold=100.0, edge_threshold=0.3)
    analysis = analyzer.analyze_image(img)
    
    bg_map = analysis['background_map']
    stability_map = analysis['stability_map']
    grid_h, grid_w = analysis['grid_shape']
    
    print(f"Grid dimensions: {grid_h} rows x {grid_w} cols\n")
    
    # Color mapping
    color_map = {
        'smooth': [0, 255, 0],
        'vertical_stripe': [0, 0, 255],
        'horizontal_stripe': [255, 0, 0],
        'textured': [255, 165, 0],
        'complex_pattern': [255, 255, 0]
    }
    
    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # 1. Original image with grid overlay
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image with Grid', fontsize=14, fontweight='bold')
    
    # Draw grid lines
    for i in range(grid_h + 1):
        y = i * grid_size
        ax1.axhline(y=y, color='cyan', linewidth=0.5, alpha=0.5)
    for j in range(grid_w + 1):
        x = j * grid_size
        ax1.axvline(x=x, color='cyan', linewidth=0.5, alpha=0.5)
    
    ax1.axis('off')
    
    # 2. Background type map
    ax2 = axes[1]
    ax2.imshow(img_rgb, alpha=0.3)
    ax2.set_title('Background Type Classification', fontsize=14, fontweight='bold')
    
    for i in range(grid_h):
        for j in range(grid_w):
            bg_type = bg_map[i, j]
            color = np.array(color_map.get(bg_type, [128, 128, 128])) / 255.0
            
            rect = patches.Rectangle(
                (j * grid_size, i * grid_size), grid_size, grid_size,
                linewidth=1, edgecolor='white', facecolor=color, alpha=0.5
            )
            ax2.add_patch(rect)
            
            # Add text label
            stability = stability_map[i, j]
            ax2.text(j * grid_size + grid_size // 2, i * grid_size + grid_size // 2,
                    f"{bg_type[:3]}\n{stability:.2f}",
                    ha='center', va='center', fontsize=6, color='black',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))
    
    ax2.axis('off')
    
    # 3. ROI selection
    ax3 = axes[2]
    ax3.imshow(img_rgb)
    ax3.set_title('ROI Selection (with overlap prevention)', fontsize=14, fontweight='bold')
    
    # Select ROIs with the same logic as show_clean_roi_boxes.py
    BG_TYPE_ORDER = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']
    rois = []
    used_cells = set()
    
    # Convert to grayscale for brightness checking
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for target_type in BG_TYPE_ORDER:
        matches = np.argwhere(bg_map == target_type)
        if len(matches) == 0:
            continue
        
        candidates = [(m, stability_map[m[0], m[1]]) for m in matches]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for match, stability in candidates:
            gi, gj = match
            cell_key = (gi, gj)
            
            if cell_key in used_cells:
                continue
            
            overlap = False
            roi_cells_span = max(1, roi_size // grid_size)
            for di in range(-roi_cells_span, roi_cells_span + 1):
                for dj in range(-roi_cells_span, roi_cells_span + 1):
                    if (gi + di, gj + dj) in used_cells:
                        overlap = True
                        break
                if overlap:
                    break
            
            if overlap:
                continue
            
            y_center = gi * grid_size + grid_size // 2
            x_center = gj * grid_size + grid_size // 2
            
            y_roi = max(0, min(H - roi_size, y_center - roi_size // 2))
            x_roi = max(0, min(W - roi_size, x_center - roi_size // 2))
            
            if y_roi + roi_size <= H and x_roi + roi_size <= W:
                # Extract ROI patch to check brightness
                roi_patch = gray[y_roi:y_roi+roi_size, x_roi:x_roi+roi_size]
                mean_brightness = np.mean(roi_patch)
                non_black_ratio = np.sum(roi_patch > 20) / roi_patch.size
                
                # Skip if patch is too dark or has too many black pixels
                if mean_brightness < 30:
                    print(f"  SKIPPED - Too dark (brightness={mean_brightness:.1f})")
                    continue
                
                if non_black_ratio < 0.8:
                    print(f"  SKIPPED - Too many black pixels ({non_black_ratio*100:.1f}% valid)")
                    continue
                
                rois.append({
                    'x': x_roi,
                    'y': y_roi,
                    'type': target_type,
                    'score': stability,
                    'grid_pos': (gi, gj)
                })
                
                for di in range(-roi_cells_span, roi_cells_span + 1):
                    for dj in range(-roi_cells_span, roi_cells_span + 1):
                        used_cells.add((gi + di, gj + dj))
                
                print(f"ROI #{len(rois)+1}: {target_type}")
                print(f"  Grid cell: [{gi}, {gj}]")
                print(f"  Grid center: ({x_center}, {y_center})")
                print(f"  ROI position: ({x_roi}, {y_roi}) to ({x_roi+roi_size}, {y_roi+roi_size})")
                print(f"  Stability: {stability:.3f}")
                print(f"  Brightness: {mean_brightness:.1f}, Valid pixels: {non_black_ratio*100:.1f}%\n")
                break
        
        if len(rois) >= 5:
            break
    
    # Draw ROIs
    for idx, roi in enumerate(rois, 1):
        color = np.array(color_map.get(roi['type'], [128, 128, 128])) / 255.0
        
        rect = patches.Rectangle(
            (roi['x'], roi['y']), roi_size, roi_size,
            linewidth=4, edgecolor=color, facecolor='none', linestyle='-'
        )
        ax3.add_patch(rect)
        
        # Draw center cross
        gi, gj = roi['grid_pos']
        x_center = gj * grid_size + grid_size // 2
        y_center = gi * grid_size + grid_size // 2
        ax3.plot(x_center, y_center, 'r+', markersize=20, markeredgewidth=3)
        
        bg_type_display = roi['type'].replace('_', ' ').title()
        label = f"ROI #{idx}\n{bg_type_display}\nGrid[{gi},{gj}]"
        ax3.text(roi['x'] + 10, roi['y'] + 30, label,
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.85, pad=0.6))
    
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{image_id}_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved debug visualization: {output_path}")
    plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("Background Type Distribution:")
    unique_types, counts = np.unique(bg_map, return_counts=True)
    for bg_type, count in zip(unique_types, counts):
        percentage = (count / bg_map.size) * 100
        print(f"  {bg_type}: {count} cells ({percentage:.1f}%)")
    
    print(f"\nTotal ROIs extracted: {len(rois)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    visualize_grid_analysis("b18d448a7.jpg")
