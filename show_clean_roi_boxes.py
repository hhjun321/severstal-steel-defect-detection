#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Show Clean Image with ROI Boxes
결함 없는 이미지에 ROI 박스 표시
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Configure matplotlib to use DejaVu Sans (default, no Korean needed)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# Setup paths
project_root = Path(__file__).parent
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "clean_roi_boxes"

# Add src to path
sys.path.insert(0, str(project_root / "src"))

from analysis.background_characterization import BackgroundAnalyzer

# Background type colors
BG_COLORS = {
    'smooth': (0, 255, 0),           # Green
    'vertical_stripe': (0, 0, 255),   # Blue
    'horizontal_stripe': (255, 0, 0), # Red
    'textured': (255, 165, 0),        # Orange
    'complex_pattern': (255, 255, 0)  # Yellow
}

BG_TYPE_ORDER = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']


def find_clean_images():
    """결함 없는 이미지 찾기"""
    all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
    train_df = pd.read_csv(train_csv_path)
    defect_images = set(train_df['ImageId'].unique())
    clean_images = list(all_images - defect_images)
    return clean_images


def calculate_optimal_roi_size(H, W, strategy='adaptive_smart', grid_size=64):
    """
    Calculate optimal ROI size using advanced strategy
    
    Strategies:
    - adaptive_smart: Intelligent sizing based on image characteristics (RECOMMENDED)
    - adaptive: 75% of smaller dimension
    - balanced: Fixed 256x256
    - large: Fixed 512x512 (if fits)
    - small: Fixed 128x128
    - grid_based: Based on grid cells (3x grid_size)
    """
    
    aspect_ratio = W / H
    
    if strategy == 'adaptive_smart':
        # Advanced adaptive strategy
        # Base size on smaller dimension
        base_size = min(H, W) * 0.75
        
        # Round to nearest 32 for efficiency
        roi_size = int((base_size // 32) * 32)
        
        # Apply constraints
        roi_size = max(128, min(512, roi_size))
        
        # For very wide images, reduce size slightly to allow more ROIs
        if aspect_ratio > 5.0:
            roi_size = int(roi_size * 0.8)
            roi_size = (roi_size // 32) * 32
        
        # Ensure minimum grid coverage (at least 3x3 cells should be available)
        min_for_grid = grid_size * 4  # Need space for 3x3 grid + margins
        if roi_size > min(H, W) - min_for_grid:
            roi_size = max(128, int((min(H, W) - min_for_grid) * 0.9))
            roi_size = (roi_size // 32) * 32
            
    elif strategy == 'adaptive':
        roi_size = int(min(H, W) * 0.75)
        roi_size = (roi_size // 32) * 32
        roi_size = max(128, min(512, roi_size))
        
    elif strategy == 'grid_based':
        # ROI covers 3x3 grid cells for good context
        roi_size = grid_size * 3
        roi_size = (roi_size // 32) * 32
        roi_size = max(128, min(512, roi_size))
        
    elif strategy == 'balanced':
        roi_size = 256
        
    elif strategy == 'large':
        roi_size = 512
        
    elif strategy == 'small':
        roi_size = 128
        
    else:
        # Default to adaptive_smart
        return calculate_optimal_roi_size(H, W, 'adaptive_smart', grid_size)
    
    # Final safety check
    if roi_size > H or roi_size > W:
        roi_size = int(min(H, W) * 0.7)
        roi_size = max(128, (roi_size // 32) * 32)
    
    return roi_size

def analyze_and_visualize(image_id, roi_size=None, roi_strategy='adaptive_smart', grid_size=64, min_rois=2):
    """Analyze image and visualize ROI boxes"""
    print(f"\nProcessing: {image_id}")
    
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    # Calculate optimal ROI size if not specified
    if roi_size is None:
        roi_size = calculate_optimal_roi_size(H, W, roi_strategy, grid_size)
    else:
        # Validate provided size
        if roi_size > H or roi_size > W:
            print(f"  Warning: ROI size {roi_size} too large for image {W}x{H}")
            roi_size = calculate_optimal_roi_size(H, W, 'adaptive_smart', grid_size)
    
    # Calculate how many grid cells the ROI spans
    roi_grid_span = roi_size / grid_size
    
    print(f"  Image: {W}x{H} (aspect {W/H:.1f}:1)")
    print(f"  Strategy: '{roi_strategy}' → ROI: {roi_size}x{roi_size} (spans {roi_grid_span:.1f} grid cells)")
    
    # Analyze background
    # Note: variance_threshold=100.0 is the standard value used across the project
    # Lower values (e.g., 50.0) may result in too few ROIs being detected
    analyzer = BackgroundAnalyzer(grid_size=grid_size, variance_threshold=100.0, edge_threshold=0.3)
    analysis = analyzer.analyze_image(img)
    
    bg_map = analysis['background_map']
    stability_map = analysis['stability_map']
    grid_h, grid_w = analysis['grid_shape']
    
    # Debug: Print background type distribution
    unique_types = np.unique(bg_map)
    print(f"  Background types found: {unique_types}")
    for bg_type in unique_types:
        count = np.sum(bg_map == bg_type)
        print(f"    - {bg_type}: {count} cells")
    
    # Convert to grayscale for brightness checking
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Select ROI regions with better distribution
    rois = []
    used_cells = set()  # Track used grid cells to avoid overlap
    
    for target_type in BG_TYPE_ORDER:
        matches = np.argwhere(bg_map == target_type)
        if len(matches) == 0:
            continue
        
        # Get all candidates with their stability scores
        candidates = [(m, stability_map[m[0], m[1]]) for m in matches]
        # Sort by stability score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Try to find non-overlapping ROIs
        for match, stability in candidates:
            gi, gj = match
            
            # Skip if this cell or nearby cells are already used
            cell_key = (gi, gj)
            if cell_key in used_cells:
                continue
            
            # Check if nearby cells are used (to avoid overlap)
            overlap = False
            roi_cells_span = max(1, roi_size // grid_size)  # How many cells the ROI spans
            for di in range(-roi_cells_span, roi_cells_span + 1):
                for dj in range(-roi_cells_span, roi_cells_span + 1):
                    if (gi + di, gj + dj) in used_cells:
                        overlap = True
                        break
                if overlap:
                    break
            
            if overlap:
                continue
            
            # Calculate ROI position
            y_center = gi * grid_size + grid_size // 2
            x_center = gj * grid_size + grid_size // 2
            
            y_roi = max(0, min(H - roi_size, y_center - roi_size // 2))
            x_roi = max(0, min(W - roi_size, x_center - roi_size // 2))
            
            # Check if ROI fits in image bounds
            if y_roi + roi_size <= H and x_roi + roi_size <= W:
                # Extract ROI patch to check brightness
                roi_patch = gray[y_roi:y_roi+roi_size, x_roi:x_roi+roi_size]
                mean_brightness = np.mean(roi_patch)
                
                # Skip if patch is too dark (likely black background/invalid region)
                # Steel images should have reasonable brightness (> 30)
                if mean_brightness < 30:
                    print(f"    - Skipped {target_type} at grid[{gi},{gj}]: too dark (brightness={mean_brightness:.1f})")
                    continue
                
                # Check if patch has enough non-black pixels (> 80% should be valid)
                non_black_ratio = np.sum(roi_patch > 20) / roi_patch.size
                if non_black_ratio < 0.8:
                    print(f"    - Skipped {target_type} at grid[{gi},{gj}]: too many black pixels ({non_black_ratio*100:.1f}% valid)")
                    continue
                rois.append({
                    'x': x_roi,
                    'y': y_roi,
                    'type': target_type,
                    'score': float(stability),
                    'grid_pos': (gi, gj)
                })
                
                # Mark cells as used
                for di in range(-roi_cells_span, roi_cells_span + 1):
                    for dj in range(-roi_cells_span, roi_cells_span + 1):
                        used_cells.add((gi + di, gj + dj))
                
                print(f"    - Added {target_type} ROI at grid[{gi},{gj}], stability={stability:.3f}")
                break  # Found one ROI for this type, move to next type
        
        if len(rois) >= 5:
            break
    
    print(f"  Selected ROIs: {len(rois)}")
    
    # Skip if not enough ROIs found
    if len(rois) < min_rois:
        print(f"  Skipped: Only {len(rois)} ROI(s) found (minimum: {min_rois})")
        return None
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.imshow(img_rgb)
    
    # Title with clear English text
    title_text = (f"{image_id} - Clean Image (No Defects)\n"
                  f"ROI Strategy: '{roi_strategy}' | ROI Size: {roi_size}x{roi_size} | "
                  f"{len(rois)} regions selected")
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Draw ROI boxes with improved labels
    for idx, roi in enumerate(rois, 1):
        color = np.array(BG_COLORS.get(roi['type'], (128, 128, 128))) / 255.0
        
        rect = patches.Rectangle(
            (roi['x'], roi['y']), roi_size, roi_size,
            linewidth=4, edgecolor=color, facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        
        # Improved label with background type and stability score
        bg_type_display = roi['type'].replace('_', ' ').title()
        label = f"ROI #{idx}\n{bg_type_display}\nStability: {roi['score']:.2f}"
        ax.text(roi['x'] + 10, roi['y'] + 35, label, 
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.85, pad=0.8),
                linespacing=1.5)
    
    # Add comprehensive legend with background type explanations
    legend_items = [
        "Background Types:",
        "  Green = Smooth surface",
        "  Blue = Vertical stripes",
        "  Red = Horizontal stripes",
        "  Orange = Textured pattern",
        "  Yellow = Complex pattern"
    ]
    legend_text = "\n".join(legend_items)
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, pad=1.2, 
                     edgecolor='gray', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{image_id}_roi_boxes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return output_path


def main(roi_strategy='adaptive_smart', num_samples=10):
    print("\n" + "=" * 80)
    print("Clean Image ROI Box Visualization")
    print("=" * 80)
    print(f"\nROI Strategy: '{roi_strategy}'")
    
    # Strategy descriptions
    strategy_info = {
        'adaptive_smart': 'Intelligent adaptive sizing (75% of min dimension, aspect-aware, grid-optimized) [RECOMMENDED]',
        'adaptive': 'Simple adaptive (75% of min dimension, 128-512 range)',
        'grid_based': 'Grid-optimized (3x grid_size, ensures good context per ROI)',
        'balanced': 'Fixed 256x256 (standard, good for training)',
        'large': 'Fixed 512x512 (large defects, more context)',
        'small': 'Fixed 128x128 (small defects, high detail)'
    }
    print(f"Description: {strategy_info.get(roi_strategy, 'Unknown strategy')}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find clean images
    clean_images = find_clean_images()
    print(f"\nTotal clean images found: {len(clean_images)}")
    
    # Process images until we get required valid outputs
    import random
    random.seed(42)  # For reproducibility
    
    # Shuffle and try images until we have enough with multiple ROIs
    shuffled_images = clean_images.copy()
    random.shuffle(shuffled_images)
    
    print(f"\nSearching for images with at least 2 ROIs...")
    print(f"Target: {num_samples} valid images")
    
    output_paths = []
    skipped_count = 0
    
    for i, image_id in enumerate(shuffled_images):
        if len(output_paths) >= num_samples:
            break
            
        print(f"\n[{len(output_paths)+1}/{num_samples}] Trying: {image_id}")
        try:
            output_path = analyze_and_visualize(image_id, roi_strategy=roi_strategy, min_rois=2)
            if output_path is not None:
                output_paths.append(output_path)
                print(f"  ✓ Saved successfully ({len(output_paths)}/{num_samples})")
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {image_id}: {e}")
            skipped_count += 1
            continue
    
    print("\n" + "=" * 80)
    print("COMPLETED!")
    print("=" * 80)
    print(f"\nProcessed {len(output_paths)} valid images (with 2+ ROIs)")
    print(f"Skipped {skipped_count} images (insufficient ROIs or errors)")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nThese ROI box regions are:")
    print(f"   1. Clean backgrounds without defects")
    print(f"   2. Classified into different background types")
    print(f"   3. Used as 'backgrounds' for data augmentation")
    print(f"   4. Synthetic defects will be added to create new training data")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
