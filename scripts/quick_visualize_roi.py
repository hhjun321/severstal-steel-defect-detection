"""
Quick ROI Visualization - Inline execution
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import ast
import sys

# Paths
project_root = Path(r"D:\project\severstal-steel-defect-detection")
roi_metadata_path = project_root / "data" / "processed" / "roi_patches" / "roi_metadata.csv"
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "roi_visualizations"

# Add utils to path
sys.path.insert(0, str(project_root / "src"))
from utils.rle_utils import rle_decode

# Check paths
print("Checking paths...")
print(f"ROI metadata: {roi_metadata_path.exists()}")
print(f"Train CSV: {train_csv_path.exists()}")
print(f"Train images: {train_images_dir.exists()}")

if not all([roi_metadata_path.exists(), train_csv_path.exists(), train_images_dir.exists()]):
    print("ERROR: Required files not found!")
    sys.exit(1)

# Load data
print("\nLoading data...")
roi_df = pd.read_csv(roi_metadata_path)
train_df = pd.read_csv(train_csv_path)

print(f"Loaded {len(roi_df)} ROI entries")
print(f"Loaded {len(train_df)} training samples")

# Select 5 diverse images
print("\nSelecting diverse samples...")

selected_images = []

# 1. Class 3 with linear_scratch + vertical_stripe
sample1 = roi_df[
    (roi_df['class_id'] == 3) & 
    (roi_df['defect_subtype'] == 'linear_scratch') &
    (roi_df['background_type'] == 'vertical_stripe')
]
if len(sample1) > 0:
    selected_images.append(sample1.iloc[0]['image_id'])
    print(f"1. {sample1.iloc[0]['image_id']} - Class 3, linear_scratch + vertical_stripe")

# 2. Class 1 with compact_blob + smooth
sample2 = roi_df[
    (roi_df['class_id'] == 1) & 
    (roi_df['defect_subtype'] == 'compact_blob') &
    (roi_df['background_type'] == 'smooth') &
    (~roi_df['image_id'].isin(selected_images))
]
if len(sample2) > 0:
    selected_images.append(sample2.iloc[0]['image_id'])
    print(f"2. {sample2.iloc[0]['image_id']} - Class 1, compact_blob + smooth")

# 3. Class 4 (rare)
sample3 = roi_df[
    (roi_df['class_id'] == 4) &
    (~roi_df['image_id'].isin(selected_images))
]
if len(sample3) > 0:
    selected_images.append(sample3.iloc[0]['image_id'])
    print(f"3. {sample3.iloc[0]['image_id']} - Class 4 (rare class)")

# 4. Image with multiple ROIs
roi_counts = roi_df.groupby('image_id').size()
multi_roi = roi_counts[roi_counts >= 5].index.tolist()
for img_id in multi_roi:
    if img_id not in selected_images:
        selected_images.append(img_id)
        print(f"4. {img_id} - Multiple ROIs ({roi_counts[img_id]})")
        break

# 5. Class 2
sample5 = roi_df[
    (roi_df['class_id'] == 2) &
    (~roi_df['image_id'].isin(selected_images))
]
if len(sample5) > 0:
    selected_images.append(sample5.iloc[0]['image_id'])
    print(f"5. {sample5.iloc[0]['image_id']} - Class 2")

# Limit to 5
selected_images = selected_images[:5]

print(f"\nSelected {len(selected_images)} images for visualization")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Color maps
rec_colors = {
    'suitable': (0, 255, 0),
    'acceptable': (255, 165, 0),
    'unsuitable': (255, 0, 0)
}

def parse_bbox(bbox_str):
    return ast.literal_eval(bbox_str)

# Generate visualizations
print("\nGenerating visualizations...")

for idx, image_id in enumerate(selected_images, 1):
    print(f"\n[{idx}/5] Processing {image_id}...")
    
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get ROIs
    img_rois = roi_df[roi_df['image_id'] == image_id]
    print(f"  Found {len(img_rois)} ROIs")
    
    # Create figure
    n_rois = min(len(img_rois), 8)  # Max 8 ROIs
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: Full image with boxes
    ax1 = plt.subplot(1, n_rois + 1, 1)
    ax1.imshow(img)
    ax1.set_title(f"{image_id}\n{len(img_rois)} ROIs", fontsize=10)
    ax1.axis('off')
    
    # Draw all ROI boxes
    for _, row in img_rois.head(n_rois).iterrows():
        roi_bbox = parse_bbox(row['roi_bbox'])
        x1, y1, x2, y2 = roi_bbox
        
        rec = row['recommendation']
        color = np.array(rec_colors.get(rec, (128, 128, 128))) / 255.0
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        label = f"R{row['region_id']}"
        ax1.text(x1, y1 - 5, label, color=color, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
    
    # Plot 2+: Individual ROI patches
    for plot_idx, (_, row) in enumerate(img_rois.head(n_rois).iterrows(), start=2):
        ax = plt.subplot(1, n_rois + 1, plot_idx)
        
        # Extract patch
        roi_bbox = parse_bbox(row['roi_bbox'])
        x1, y1, x2, y2 = roi_bbox
        patch = img[y1:y2, x1:x2]
        
        # Get mask
        mask_data = train_df[
            (train_df['ImageId'] == image_id) & 
            (train_df['ClassId'] == row['class_id'])
        ]
        
        if len(mask_data) > 0 and not pd.isna(mask_data.iloc[0]['EncodedPixels']):
            rle = mask_data.iloc[0]['EncodedPixels']
            mask = rle_decode(rle, shape=(256, 1600))
            mask_patch = mask[y1:y2, x1:x2]
            
            # Overlay
            overlay = patch.copy()
            overlay[mask_patch > 0] = [255, 0, 0]
            patch = cv2.addWeighted(patch, 0.7, overlay, 0.3, 0)
        
        ax.imshow(patch)
        
        title = (f"ROI {row['region_id']} | C{row['class_id']}\n"
                f"{row['defect_subtype']}\n"
                f"{row['background_type']}\n"
                f"Score: {row['suitability_score']:.2f}\n"
                f"({row['recommendation']})")
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / f"roi_visualization_{idx}_{image_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path.name}")
    plt.close()

# Create comparison view
print("\nCreating comparison view...")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for ax, image_id in zip(axes, selected_images):
    # Load image
    img_path = train_images_dir / image_id
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get ROIs
    img_rois = roi_df[roi_df['image_id'] == image_id]
    
    ax.imshow(img)
    ax.set_title(f"{image_id}\n{len(img_rois)} ROIs", fontsize=9)
    ax.axis('off')
    
    # Draw boxes
    for _, row in img_rois.iterrows():
        roi_bbox = parse_bbox(row['roi_bbox'])
        x1, y1, x2, y2 = roi_bbox
        
        rec = row['recommendation']
        color = np.array(rec_colors.get(rec, (128, 128, 128))) / 255.0
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        label = f"C{row['class_id']}"
        ax.text(x1, y1 - 3, label, color=color, fontsize=7,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))

plt.tight_layout()
comparison_path = output_dir / "roi_comparison_all_5_samples.png"
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"Saved comparison: {comparison_path.name}")
plt.close()

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)
print(f"\n📁 Output directory: {output_dir}")
print(f"\nGenerated files:")
for i in range(1, 6):
    print(f"  - roi_visualization_{i}_*.png (detailed view)")
print(f"  - roi_comparison_all_5_samples.png (comparison view)")
print("\nLegend:")
print("  🟢 Green boxes: suitable (score ≥ 0.8)")
print("  🟠 Orange boxes: acceptable (0.5 ≤ score < 0.8)")
print("  🔴 Red boxes: unsuitable (score < 0.5)")
print("\nRed overlay in patches = actual defect mask")
