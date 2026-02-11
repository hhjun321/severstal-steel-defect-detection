#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare different ROI size strategies
Generates visualizations for multiple ROI strategies on the same images
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
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "roi_strategy_comparison"

# Add src to path
sys.path.insert(0, str(project_root / "src"))
from analysis.background_characterization import BackgroundAnalyzer

# Background type colors
BG_COLORS = {
    'smooth': (0, 255, 0),
    'vertical_stripe': (0, 0, 255),
    'horizontal_stripe': (255, 0, 0),
    'textured': (255, 165, 0),
    'complex_pattern': (255, 255, 0)
}

BG_TYPE_ORDER = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']

def find_clean_images():
    """Find clean images without defects"""
    all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
    train_df = pd.read_csv(train_csv_path)
    defect_images = set(train_df['ImageId'].unique())
    clean_images = list(all_images - defect_images)
    return clean_images

def get_roi_size_for_strategy(strategy, H, W):
    """Calculate ROI size based on strategy"""
    if strategy == 'adaptive':
        roi_size = min(int(min(H, W) * 0.8), 512)
        roi_size = max(roi_size, 128)
    elif strategy == 'balanced':
        roi_size = 256
    elif strategy == 'large':
        roi_size = 512
    elif strategy == 'small':
        roi_size = 128
    else:
        roi_size = 256
    
    # Ensure ROI fits
    if roi_size > H or roi_size > W:
        roi_size = min(int(min(H, W) * 0.8), 512)
        roi_size = max(roi_size, 128)
    
    return roi_size

def extract_rois(img, strategy, grid_size=64):
    """Extract ROIs for a given strategy"""
    H, W = img.shape[:2]
    roi_size = get_roi_size_for_strategy(strategy, H, W)
    
    # Analyze background
    analyzer = BackgroundAnalyzer(grid_size=grid_size, variance_threshold=100.0, edge_threshold=0.3)
    analysis = analyzer.analyze_image(img)
    
    bg_map = analysis['background_map']
    stability_map = analysis['stability_map']
    
    # Convert to grayscale for brightness checking
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Select ROI regions
    rois = []
    used_cells = set()
    
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
                roi_patch = gray[y_roi:y_roi+roi_size, x_roi:x_roi+roi_size]
                mean_brightness = np.mean(roi_patch)
                non_black_ratio = np.sum(roi_patch > 20) / roi_patch.size
                
                if mean_brightness < 30 or non_black_ratio < 0.8:
                    continue
                
                rois.append({
                    'x': x_roi,
                    'y': y_roi,
                    'size': roi_size,
                    'type': target_type,
                    'score': float(stability)
                })
                
                for di in range(-roi_cells_span, roi_cells_span + 1):
                    for dj in range(-roi_cells_span, roi_cells_span + 1):
                        used_cells.add((gi + di, gj + dj))
                
                break
            
            if len(rois) >= 5:
                break
        
        if len(rois) >= 5:
            break
    
    return rois, roi_size

def compare_strategies_on_image(image_id):
    """Compare all ROI strategies on a single image"""
    print(f"\nComparing strategies on: {image_id}")
    
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    print(f"  Image size: {W}x{H}")
    
    # Test all strategies
    strategies = ['small', 'balanced', 'large', 'adaptive']
    strategy_names = {
        'small': 'Small (128x128)',
        'balanced': 'Balanced (256x256)',
        'large': 'Large (512x512)',
        'adaptive': 'Adaptive (Dynamic)'
    }
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        ax.imshow(img_rgb)
        
        # Extract ROIs for this strategy
        rois, roi_size = extract_rois(img, strategy)
        
        # Draw ROIs
        for roi_idx, roi in enumerate(rois, 1):
            color = np.array(BG_COLORS.get(roi['type'], (128, 128, 128))) / 255.0
            
            rect = patches.Rectangle(
                (roi['x'], roi['y']), roi['size'], roi['size'],
                linewidth=3, edgecolor=color, facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            
            # Add label
            bg_type_display = roi['type'].replace('_', ' ').title()
            label = f"#{roi_idx}\n{bg_type_display[:8]}"
            ax.text(roi['x'] + 5, roi['y'] + 20, label,
                   color='white', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.85, pad=0.4))
        
        # Title
        title = f"{strategy_names[strategy]}\nActual ROI: {roi_size}x{roi_size} | {len(rois)} ROIs found"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        print(f"    {strategy}: ROI={roi_size}x{roi_size}, Count={len(rois)}")
    
    # Main title
    fig.suptitle(f"ROI Strategy Comparison: {image_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = output_dir / f"{image_id}_strategy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path

def main():
    print("\n" + "=" * 80)
    print("ROI STRATEGY COMPARISON")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find clean images
    clean_images = find_clean_images()
    print(f"\nTotal clean images found: {len(clean_images)}")
    
    # Select 3 sample images
    import random
    random.seed(42)
    sample_images = random.sample(clean_images, min(3, len(clean_images)))
    
    print(f"\nComparing strategies on {len(sample_images)} sample images...")
    
    for image_id in sample_images:
        try:
            compare_strategies_on_image(image_id)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nStrategy Summary:")
    print("  • Small (128x128): Best for small defects, high detail")
    print("  • Balanced (256x256): Standard size, good for training")
    print("  • Large (512x512): Best for large defects, more context")
    print("  • Adaptive: Dynamic sizing based on image dimensions")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
