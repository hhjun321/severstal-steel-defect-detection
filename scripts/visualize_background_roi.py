"""
Visualize Background ROI Extraction from Clean (Defect-Free) Images
결함이 없는 깨끗한 이미지에서 배경 ROI 추출 시각화
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Paths
project_root = Path(r"D:\project\severstal-steel-defect-detection")
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "background_roi_visualizations"

# Add utils to path
sys.path.insert(0, str(project_root / "src"))
from analysis.background_characterization import BackgroundCharacterizer

print("="*80)
print("BACKGROUND ROI VISUALIZATION FROM CLEAN IMAGES")
print("결함 없는 이미지에서 배경 ROI 추출 시각화")
print("="*80)

# Check paths
print("\n1. Checking paths...")
print(f"   Train CSV: {train_csv_path.exists()}")
print(f"   Train images: {train_images_dir.exists()}")

if not all([train_csv_path.exists(), train_images_dir.exists()]):
    print("   ERROR: Required files not found!")
    sys.exit(1)

# Load data
print("\n2. Loading data...")
train_df = pd.read_csv(train_csv_path)
print(f"   Loaded {len(train_df)} training samples")

# Find images with NO defects (EncodedPixels is NaN)
print("\n3. Finding clean images (no defects)...")
clean_images = train_df[train_df['EncodedPixels'].isna()]['ImageId'].unique()
print(f"   Found {len(clean_images)} clean images")

if len(clean_images) == 0:
    print("   ERROR: No clean images found!")
    print("   Note: In this dataset, all images have at least one defect.")
    print("   Switching to: Find images with minimal defects for background extraction")
    
    # Find images with only 1 defect (best for background extraction)
    defect_counts = train_df.groupby('ImageId').size()
    minimal_defect_images = defect_counts[defect_counts == 1].index.tolist()
    print(f"   Found {len(minimal_defect_images)} images with only 1 defect")
    
    # Use these for background extraction
    selected_images = minimal_defect_images[:5]
    extraction_mode = "minimal_defect"
else:
    # Select 5 random clean images
    selected_images = list(clean_images[:5])
    extraction_mode = "clean"

print(f"\n4. Selected {len(selected_images)} images for background visualization:")
for i, img_id in enumerate(selected_images, 1):
    print(f"   {i}. {img_id}")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize background characterizer
print("\n5. Initializing background characterizer...")
bg_char = BackgroundCharacterizer()

# Background type colors
bg_colors = {
    'smooth': (0, 255, 0),           # Green
    'vertical_stripe': (0, 0, 255),   # Blue
    'horizontal_stripe': (255, 0, 0), # Red
    'textured': (255, 165, 0),        # Orange
    'complex_pattern': (255, 255, 0)  # Yellow
}

print("\n6. Generating visualizations...")
print("="*80)

for idx, image_id in enumerate(selected_images, 1):
    print(f"\n[{idx}/5] Processing {image_id}...")
    
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"   Loaded image: {img.shape}")
    
    # Analyze background using grid-based method
    print(f"   Analyzing background grid (64x64)...")
    grid_size = 64
    H, W = img_gray.shape
    grid_h = H // grid_size
    grid_w = W // grid_size
    
    # Create background type map
    bg_type_map = np.zeros((grid_h, grid_w), dtype=object)
    bg_score_map = np.zeros((grid_h, grid_w))
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Extract grid cell
            cell = img_gray[i*grid_size:(i+1)*grid_size,
                           j*grid_size:(j+1)*grid_size]
            
            # Classify background
            bg_type = bg_char.classify_background(cell)
            bg_type_map[i, j] = bg_type
            
            # Compute stability score (1 - variance/max_variance)
            variance = np.var(cell)
            stability = max(0, 1.0 - (variance / (255**2)))
            bg_score_map[i, j] = stability
    
    # Count background types
    unique_types, counts = np.unique(bg_type_map, return_counts=True)
    type_dist = dict(zip(unique_types, counts))
    
    print(f"   Background distribution:")
    for bg_type, count in type_dist.items():
        pct = (count / (grid_h * grid_w)) * 100
        print(f"     - {bg_type}: {count} cells ({pct:.1f}%)")
    
    # Select 5 diverse ROI regions (512x512)
    print(f"   Selecting 5 diverse ROI regions (512x512)...")
    roi_size = 512
    rois = []
    
    # Find suitable ROI positions
    for target_type in ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']:
        # Find grid cells with this type
        matches = np.argwhere(bg_type_map == target_type)
        if len(matches) == 0:
            continue
        
        # Pick the one with highest stability
        best_idx = np.argmax([bg_score_map[m[0], m[1]] for m in matches])
        gi, gj = matches[best_idx]
        
        # Convert grid position to image position (centered)
        y = max(0, min(H - roi_size, gi * grid_size - roi_size//4))
        x = max(0, min(W - roi_size, gj * grid_size - roi_size//4))
        
        # Ensure within bounds
        if y + roi_size > H or x + roi_size > W:
            continue
        
        rois.append({
            'x': x, 'y': y,
            'type': target_type,
            'score': bg_score_map[gi, gj]
        })
        
        if len(rois) >= 5:
            break
    
    print(f"   Selected {len(rois)} ROIs")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: Full image with ROI boxes
    ax1 = plt.subplot(1, len(rois) + 1, 1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"{image_id}\n{len(rois)} Background ROIs\n({extraction_mode} mode)", fontsize=10)
    ax1.axis('off')
    
    # Draw ROI boxes
    for roi in rois:
        color = np.array(bg_colors.get(roi['type'], (128, 128, 128))) / 255.0
        
        rect = patches.Rectangle(
            (roi['x'], roi['y']), roi_size, roi_size,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Add label
        label = f"{roi['type'][:4]}"
        ax1.text(roi['x'], roi['y'] - 5, label, color=color, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
    
    # Plot 2+: Individual ROI patches
    for plot_idx, roi in enumerate(rois, start=2):
        ax = plt.subplot(1, len(rois) + 1, plot_idx)
        
        # Extract patch
        patch = img_rgb[roi['y']:roi['y']+roi_size, 
                       roi['x']:roi['x']+roi_size]
        
        ax.imshow(patch)
        
        title = (f"ROI {plot_idx-1}\n"
                f"Type: {roi['type']}\n"
                f"Stability: {roi['score']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / f"background_roi_{idx}_{image_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path.name}")
    plt.close()

# Create comparison view
print("\n7. Creating comparison view...")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for ax, image_id in zip(axes, selected_images):
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ax.imshow(img_rgb)
    ax.set_title(f"{image_id}\nClean Background", fontsize=9)
    ax.axis('off')

plt.tight_layout()
comparison_path = output_dir / "background_comparison_all_5_samples.png"
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved comparison: {comparison_path.name}")
plt.close()

print("\n" + "="*80)
print("✅ BACKGROUND VISUALIZATION COMPLETE!")
print("="*80)
print(f"\n📁 Output directory: {output_dir}")
print(f"\nGenerated files:")
for i in range(1, len(selected_images) + 1):
    print(f"  - background_roi_{i}_*.png (detailed view)")
print(f"  - background_comparison_all_5_samples.png (comparison view)")

print("\n📊 Background Type Color Legend:")
print("  🟢 Green   = smooth (uniform surface)")
print("  🔵 Blue    = vertical_stripe")
print("  🔴 Red     = horizontal_stripe")
print("  🟠 Orange  = textured")
print("  🟡 Yellow  = complex_pattern")

print("\n📝 Note:")
if extraction_mode == "minimal_defect":
    print("  - Used images with minimal defects (1 defect each)")
    print("  - These images have large clean background areas")
    print("  - Perfect for background extraction in augmentation")
else:
    print("  - Used completely clean images (no defects)")
    print("  - Pure background samples")

print("\n🎯 Use Case:")
print("  These background ROIs will be used in Stage 3 (Augmentation)")
print("  to generate synthetic defects on realistic backgrounds")
