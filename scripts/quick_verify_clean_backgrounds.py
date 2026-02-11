"""
Quick Verification: Background Extraction from Clean Images
깨끗한 이미지에서 배경 추출 빠른 검증

This script verifies that background extraction correctly identifies and processes
clean images (those NOT in train.csv) by visualizing ROI extraction results.

이 스크립트는 배경 추출이 깨끗한 이미지(train.csv에 없는 이미지)를 올바르게
식별하고 처리하는지 ROI 추출 결과를 시각화하여 검증합니다.

RESEARCH PROTOCOL:
- train.csv contains ONLY images WITH defects (6,666 images)
- Clean images are NOT in train.csv (5,902 images)
- We extract backgrounds from clean images for augmentation
- Clean images = all_images - train_csv_images

연구 원칙:
- train.csv는 결함이 있는 이미지만 포함 (6,666개)
- 깨끗한 이미지는 train.csv에 없음 (5,902개)
- 증강을 위해 깨끗한 이미지에서 배경을 추출
- 깨끗한 이미지 = 전체 이미지 - train.csv 이미지

Output:
- Individual visualization files for each clean image with ROI boxes
- Comparison grid showing all images side-by-side
- Console statistics confirming correct image selection
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import random

# Setup paths
project_root = Path(__file__).parent.parent
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "clean_bg_verification"

# Add src to path
sys.path.insert(0, str(project_root / "src"))

# Import BackgroundAnalyzer (NOT BackgroundCharacterizer - that doesn't exist!)
from analysis.background_characterization import BackgroundAnalyzer

# Background type colors for visualization
BG_COLORS = {
    'smooth': (0, 255, 0),           # Green
    'vertical_stripe': (0, 0, 255),   # Blue
    'horizontal_stripe': (255, 0, 0), # Red
    'textured': (255, 165, 0),        # Orange
    'complex_pattern': (255, 255, 0)  # Yellow
}

BG_TYPE_ORDER = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']


def print_header():
    """Print script header"""
    print("=" * 80)
    print("QUICK VERIFICATION: Background Extraction from Clean Images")
    print("깨끗한 이미지에서 배경 추출 빠른 검증")
    print("=" * 80)
    print()
    print("RESEARCH PROTOCOL:")
    print("- train.csv contains ONLY images WITH defects")
    print("- Clean images are NOT in train.csv")
    print("- We extract backgrounds from clean images for augmentation")
    print()


def find_clean_images(train_csv_path: Path, train_images_dir: Path) -> list:
    """
    Find images with NO defects (images NOT in train.csv).
    
    CORRECT METHOD:
    - All images in train_images/ directory: 12,568
    - Images listed in train.csv (with defects): 6,666
    - Clean images = all_images - train_csv_images = 5,902
    
    올바른 방법:
    - train_images/ 디렉토리의 모든 이미지: 12,568개
    - train.csv에 있는 이미지 (결함 있음): 6,666개
    - 깨끗한 이미지 = 전체 이미지 - train.csv 이미지 = 5,902개
    
    Args:
        train_csv_path: Path to train.csv
        train_images_dir: Path to train_images/ directory
        
    Returns:
        List of clean image filenames (not in train.csv)
    """
    print("1. Finding clean images...")
    
    # Get all images from directory
    all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
    print(f"   Total images in train_images/: {len(all_images)}")
    
    # Get images with defects (in train.csv)
    train_df = pd.read_csv(train_csv_path)
    defect_images = set(train_df['ImageId'].unique())
    print(f"   Images WITH defects (in train.csv): {len(defect_images)}")
    
    # Clean images = all images - images with defects
    clean_images = list(all_images - defect_images)
    print(f"   Clean images (NOT in train.csv): {len(clean_images)}")
    
    if len(clean_images) == 0:
        print("   ❌ ERROR: No clean images found!")
        print("   This should not happen - expected 5,902 clean images")
        sys.exit(1)
    
    print(f"   ✓ Found {len(clean_images)} clean images")
    print()
    
    return clean_images


def analyze_background_rois(image_path: Path, analyzer: BackgroundAnalyzer, 
                           roi_size: int = 512, grid_size: int = 64) -> tuple:
    """
    Analyze background and extract diverse ROI regions.
    
    Args:
        image_path: Path to image file
        analyzer: BackgroundAnalyzer instance
        roi_size: Size of ROI patches (default: 512x512)
        grid_size: Grid size for analysis (default: 64x64)
        
    Returns:
        Tuple of (img_rgb, rois, bg_stats)
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    print(f"   Image size: {img.shape}")
    
    # Analyze background using BackgroundAnalyzer
    print(f"   Analyzing background grid ({grid_size}×{grid_size})...")
    analysis = analyzer.analyze_image(img)
    
    # Extract results
    bg_map = analysis['background_map']      # (grid_h, grid_w) with string values
    stability_map = analysis['stability_map'] # (grid_h, grid_w) with float scores
    grid_h, grid_w = analysis['grid_shape']
    
    # Count background types
    unique_types, counts = np.unique(bg_map, return_counts=True)
    bg_stats = dict(zip(unique_types, counts))
    
    print(f"   Background distribution:")
    for bg_type in BG_TYPE_ORDER:
        if bg_type in bg_stats:
            count = bg_stats[bg_type]
            pct = (count / (grid_h * grid_w)) * 100
            print(f"     - {bg_type}: {count} cells ({pct:.1f}%)")
    
    # Select diverse ROI regions (one per background type, max 5)
    print(f"   Selecting diverse ROI regions ({roi_size}×{roi_size})...")
    rois = []
    
    for target_type in BG_TYPE_ORDER:
        # Find grid cells with this background type
        matches = np.argwhere(bg_map == target_type)
        if len(matches) == 0:
            continue
        
        # Select cell with highest stability score
        best_idx = np.argmax([stability_map[m[0], m[1]] for m in matches])
        gi, gj = matches[best_idx]
        
        # Convert grid position to pixel position
        # Center the ROI around the grid cell
        y_center = gi * grid_size + grid_size // 2
        x_center = gj * grid_size + grid_size // 2
        
        # Calculate ROI top-left corner
        y_roi = max(0, min(H - roi_size, y_center - roi_size // 2))
        x_roi = max(0, min(W - roi_size, x_center - roi_size // 2))
        
        # Verify ROI is within bounds
        if y_roi + roi_size > H or x_roi + roi_size > W:
            continue
        
        rois.append({
            'x': x_roi,
            'y': y_roi,
            'type': target_type,
            'score': float(stability_map[gi, gj]),
            'grid_pos': (gi, gj)
        })
        
        if len(rois) >= 5:
            break
    
    print(f"   Selected {len(rois)} ROI regions")
    
    return img_rgb, rois, bg_stats


def visualize_single_image(img_rgb: np.ndarray, rois: list, image_id: str, 
                          output_path: Path, roi_size: int = 512):
    """
    Create visualization for a single image with ROI boxes.
    
    Args:
        img_rgb: RGB image array
        rois: List of ROI dictionaries
        image_id: Image filename
        output_path: Path to save visualization
        roi_size: Size of ROI boxes
    """
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: Full image with ROI boxes
    ax1 = plt.subplot(1, len(rois) + 1, 1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"{image_id}\n{len(rois)} Background ROIs\n(Clean Image - NOT in train.csv)", 
                 fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # Draw ROI boxes on full image
    for roi in rois:
        # Get color for this background type
        color = np.array(BG_COLORS.get(roi['type'], (128, 128, 128))) / 255.0
        
        # Draw rectangle
        rect = patches.Rectangle(
            (roi['x'], roi['y']), roi_size, roi_size,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Add label with background type
        label = f"{roi['type'][:8]}"
        ax1.text(roi['x'], roi['y'] - 5, label, 
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
    
    # Plot 2+: Individual ROI patches
    for plot_idx, roi in enumerate(rois, start=2):
        ax = plt.subplot(1, len(rois) + 1, plot_idx)
        
        # Extract patch
        patch = img_rgb[roi['y']:roi['y'] + roi_size, 
                       roi['x']:roi['x'] + roi_size]
        
        ax.imshow(patch)
        
        # Title with metadata
        title = (f"ROI {plot_idx - 1}\n"
                f"Type: {roi['type']}\n"
                f"Stability: {roi['score']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Add colored border matching ROI box
        color = np.array(BG_COLORS.get(roi['type'], (128, 128, 128))) / 255.0
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_grid(image_paths: list, output_path: Path):
    """
    Create comparison grid showing all images side-by-side.
    
    Args:
        image_paths: List of (image_path, image_id) tuples
        output_path: Path to save comparison grid
    """
    n_images = len(image_paths)
    fig, axes = plt.subplots(1, n_images, figsize=(8 * n_images, 8))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (img_path, img_id) in zip(axes, image_paths):
        # Load and display image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.set_title(f"{img_id}\n(Clean - NOT in train.csv)", 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle("Clean Background Images for ROI Extraction", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(output_dir: Path, processed_images: list, total_rois: int):
    """Print summary of results"""
    print()
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 Output directory: {output_dir}")
    print()
    print("Generated files:")
    for i, img_id in enumerate(processed_images, 1):
        print(f"  - image_{i}_{img_id}.png (detailed view with {total_rois} ROIs)")
    print(f"  - comparison_grid.png (summary view)")
    print()
    print("📊 Background Type Color Legend:")
    print("  🟢 Green  = smooth (uniform surface)")
    print("  🔵 Blue   = vertical_stripe")
    print("  🔴 Red    = horizontal_stripe")
    print("  🟠 Orange = textured")
    print("  🟡 Yellow = complex_pattern")
    print()
    print("✅ VERIFICATION SUCCESS: All processed images are clean (NOT in train.csv)")
    print()
    print("📝 Next Steps:")
    print("  1. Open outputs/clean_bg_verification/ and review visualization files")
    print("  2. Verify that images are truly clean (no visible defects)")
    print("  3. Check that ROI boxes make sense for each background type")
    print("  4. If everything looks good, run full background extraction:")
    print("     python scripts\\run_background_extraction.py --max-images 100")
    print()


def main():
    """Main execution function"""
    # Print header
    print_header()
    
    # Check paths
    if not train_csv_path.exists():
        print(f"❌ ERROR: train.csv not found at {train_csv_path}")
        sys.exit(1)
    
    if not train_images_dir.exists():
        print(f"❌ ERROR: train_images/ not found at {train_images_dir}")
        sys.exit(1)
    
    # Find clean images (CORRECT METHOD)
    clean_images = find_clean_images(train_csv_path, train_images_dir)
    
    # Select 3 random clean images for verification
    print("2. Selecting 3 random clean images for visualization...")
    random.seed(42)  # For reproducibility
    selected_images = random.sample(clean_images, min(3, len(clean_images)))
    
    for i, img_id in enumerate(selected_images, 1):
        print(f"   {i}. {img_id}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize background analyzer
    print("3. Initializing background analyzer...")
    analyzer = BackgroundAnalyzer(grid_size=64, variance_threshold=100.0, edge_threshold=0.3)
    print("   ✓ BackgroundAnalyzer initialized")
    print()
    
    # Process each image
    print("4. Processing images...")
    print("=" * 80)
    print()
    
    processed_images = []
    image_paths = []
    total_rois = 0
    
    for idx, image_id in enumerate(selected_images, 1):
        print(f"[{idx}/3] Processing {image_id}...")
        
        # Full path to image
        img_path = train_images_dir / image_id
        
        # Analyze background and extract ROIs
        img_rgb, rois, bg_stats = analyze_background_rois(img_path, analyzer)
        total_rois += len(rois)
        
        # Create visualization
        output_path = output_dir / f"image_{idx}_{image_id}.png"
        visualize_single_image(img_rgb, rois, image_id, output_path)
        
        print(f"   ✓ Saved: {output_path.name}")
        print()
        
        processed_images.append(image_id)
        image_paths.append((img_path, image_id))
    
    # Create comparison grid
    print("5. Creating comparison grid...")
    comparison_path = output_dir / "comparison_grid.png"
    create_comparison_grid(image_paths, comparison_path)
    print(f"   ✓ Saved: {comparison_path.name}")
    
    # Print summary
    avg_rois = total_rois / len(processed_images) if processed_images else 0
    print_summary(output_dir, processed_images, int(avg_rois))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
