"""
Visualize ROI Extraction Results
Displays original images with ROI bounding boxes overlaid
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import ast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.rle_utils import rle_decode


class ROIVisualizer:
    """Visualize ROI extraction results"""
    
    def __init__(self, roi_metadata_path, train_csv_path, train_images_dir):
        """
        Initialize visualizer
        
        Args:
            roi_metadata_path: Path to roi_metadata.csv
            train_csv_path: Path to train.csv
            train_images_dir: Directory containing training images
        """
        self.roi_df = pd.read_csv(roi_metadata_path)
        self.train_df = pd.read_csv(train_csv_path)
        self.images_dir = Path(train_images_dir)
        
        # Color map for different classes
        self.class_colors = {
            1: (255, 0, 0),      # Red
            2: (0, 255, 0),      # Green
            3: (0, 0, 255),      # Blue
            4: (255, 255, 0)     # Yellow
        }
        
        # Recommendation colors
        self.rec_colors = {
            'suitable': (0, 255, 0),      # Green
            'acceptable': (255, 165, 0),  # Orange
            'unsuitable': (255, 0, 0)     # Red
        }
    
    def parse_bbox(self, bbox_str):
        """Parse bbox string to tuple"""
        return ast.literal_eval(bbox_str)
    
    def load_image(self, image_id):
        """Load image from disk"""
        img_path = self.images_dir / image_id
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_defect_mask(self, image_id, class_id):
        """Get defect mask from train.csv"""
        # Filter by image and class
        mask_data = self.train_df[
            (self.train_df['ImageId'] == image_id) & 
            (self.train_df['ClassId'] == class_id)
        ]
        
        if len(mask_data) == 0:
            return None
        
        # Get RLE encoding
        rle = mask_data.iloc[0]['EncodedPixels']
        if pd.isna(rle):
            return None
        
        # Decode RLE to mask
        mask = rle_decode(rle, shape=(256, 1600))
        return mask
    
    def visualize_single_image(self, image_id, max_rois=None, save_path=None):
        """
        Visualize a single image with all its ROIs
        
        Args:
            image_id: Image filename
            max_rois: Maximum number of ROIs to display (None = all)
            save_path: Path to save figure (None = display only)
        """
        # Load image
        img = self.load_image(image_id)
        
        # Get all ROIs for this image
        img_rois = self.roi_df[self.roi_df['image_id'] == image_id]
        
        if len(img_rois) == 0:
            print(f"No ROIs found for {image_id}")
            return
        
        if max_rois is not None:
            img_rois = img_rois.head(max_rois)
        
        # Create figure with subplots
        n_rois = len(img_rois)
        n_cols = min(4, n_rois + 1)
        n_rows = (n_rois + n_cols) // n_cols
        
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        
        # Plot 1: Original image with all ROI boxes
        ax = plt.subplot(n_rows, n_cols, 1)
        ax.imshow(img)
        ax.set_title(f"{image_id}\nTotal ROIs: {n_rois}", fontsize=10)
        ax.axis('off')
        
        # Draw all ROI bounding boxes
        for idx, row in img_rois.iterrows():
            class_id = row['class_id']
            roi_bbox = self.parse_bbox(row['roi_bbox'])
            x1, y1, x2, y2 = roi_bbox
            
            # Get color based on recommendation
            rec = row['recommendation']
            color = np.array(self.rec_colors.get(rec, (128, 128, 128))) / 255.0
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"C{class_id}-R{row['region_id']}"
            ax.text(x1, y1 - 5, label, color=color, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Plot 2+: Individual ROI patches
        for plot_idx, (idx, row) in enumerate(img_rois.iterrows(), start=2):
            if plot_idx > n_cols * n_rows:
                break
            
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            
            # Extract ROI patch
            roi_bbox = self.parse_bbox(row['roi_bbox'])
            x1, y1, x2, y2 = roi_bbox
            roi_patch = img[y1:y2, x1:x2]
            
            # Get defect mask for overlay
            mask = self.get_defect_mask(image_id, row['class_id'])
            if mask is not None:
                mask_patch = mask[y1:y2, x1:x2]
                # Create colored overlay
                overlay = roi_patch.copy()
                overlay[mask_patch > 0] = [255, 0, 0]  # Red
                # Blend
                roi_patch = cv2.addWeighted(roi_patch, 0.7, overlay, 0.3, 0)
            
            ax.imshow(roi_patch)
            
            # Create title with key info
            title = (f"ROI {row['region_id']} | Class {row['class_id']}\n"
                    f"{row['defect_subtype']}\n"
                    f"{row['background_type']}\n"
                    f"Suit: {row['suitability_score']:.2f} ({row['recommendation']})")
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_comparison(self, image_ids, save_path=None):
        """
        Visualize multiple images side by side
        
        Args:
            image_ids: List of image filenames
            save_path: Path to save figure
        """
        n_images = len(image_ids)
        fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
        
        if n_images == 1:
            axes = [axes]
        
        for ax, image_id in zip(axes, image_ids):
            # Load image
            img = self.load_image(image_id)
            
            # Get ROIs
            img_rois = self.roi_df[self.roi_df['image_id'] == image_id]
            
            ax.imshow(img)
            ax.set_title(f"{image_id}\n{len(img_rois)} ROIs", fontsize=10)
            ax.axis('off')
            
            # Draw ROI boxes
            for idx, row in img_rois.iterrows():
                roi_bbox = self.parse_bbox(row['roi_bbox'])
                x1, y1, x2, y2 = roi_bbox
                
                rec = row['recommendation']
                color = np.array(self.rec_colors.get(rec, (128, 128, 128))) / 255.0
                
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Label
                label = f"C{row['class_id']}"
                ax.text(x1, y1 - 3, label, color=color, fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_summary_visualization(self, image_ids, save_dir):
        """
        Create comprehensive summary for multiple images
        
        Args:
            image_ids: List of 5 image IDs to visualize
            save_dir: Directory to save outputs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("Creating ROI visualizations...")
        print("=" * 80)
        
        # Create individual visualizations
        for i, image_id in enumerate(image_ids, 1):
            print(f"\n[{i}/{len(image_ids)}] Processing {image_id}...")
            
            # Get ROI info
            img_rois = self.roi_df[self.roi_df['image_id'] == image_id]
            print(f"  - Found {len(img_rois)} ROIs")
            
            # Print summary
            for _, row in img_rois.iterrows():
                print(f"    ROI {row['region_id']}: "
                      f"Class {row['class_id']} | "
                      f"{row['defect_subtype']} + {row['background_type']} | "
                      f"Score: {row['suitability_score']:.3f} ({row['recommendation']})")
            
            # Create detailed visualization
            save_path = save_dir / f"roi_detail_{i}_{image_id}"
            self.visualize_single_image(image_id, max_rois=8, 
                                       save_path=save_path)
        
        # Create comparison visualization
        print(f"\nCreating comparison visualization...")
        comparison_path = save_dir / "roi_comparison_all.png"
        self.visualize_comparison(image_ids, save_path=comparison_path)
        
        print("\n" + "=" * 80)
        print(f"✅ All visualizations saved to: {save_dir}")
        print("\nLegend:")
        print("  🟢 Green boxes: suitable (score ≥ 0.8)")
        print("  🟠 Orange boxes: acceptable (0.5 ≤ score < 0.8)")
        print("  🔴 Red boxes: unsuitable (score < 0.5)")


def main():
    """Main visualization script"""
    # Paths
    project_root = Path(__file__).parent.parent
    roi_metadata_path = project_root / "data" / "processed" / "roi_patches" / "roi_metadata.csv"
    train_csv_path = project_root / "data" / "train.csv"
    train_images_dir = project_root / "data" / "train_images"
    output_dir = project_root / "outputs" / "roi_visualizations"
    
    # Check files exist
    if not roi_metadata_path.exists():
        print(f"❌ Error: ROI metadata not found at {roi_metadata_path}")
        print("Please run Stage 1 (ROI extraction) first.")
        return
    
    if not train_csv_path.exists():
        print(f"❌ Error: train.csv not found at {train_csv_path}")
        return
    
    if not train_images_dir.exists():
        print(f"❌ Error: train_images directory not found at {train_images_dir}")
        return
    
    # Initialize visualizer
    visualizer = ROIVisualizer(roi_metadata_path, train_csv_path, train_images_dir)
    
    # Select diverse samples (different classes and characteristics)
    roi_df = visualizer.roi_df
    
    # Strategy: Pick images with different classes and good variety
    selected_images = []
    
    # 1. Class 3 with linear_scratch + vertical_stripe (perfect match)
    sample1 = roi_df[
        (roi_df['class_id'] == 3) & 
        (roi_df['defect_subtype'] == 'linear_scratch') &
        (roi_df['background_type'] == 'vertical_stripe')
    ]
    if len(sample1) > 0:
        selected_images.append(sample1.iloc[0]['image_id'])
    
    # 2. Class 1 with compact_blob + smooth (perfect match)
    sample2 = roi_df[
        (roi_df['class_id'] == 1) & 
        (roi_df['defect_subtype'] == 'compact_blob') &
        (roi_df['background_type'] == 'smooth') &
        (~roi_df['image_id'].isin(selected_images))
    ]
    if len(sample2) > 0:
        selected_images.append(sample2.iloc[0]['image_id'])
    
    # 3. Class 4 (rare class)
    sample3 = roi_df[
        (roi_df['class_id'] == 4) &
        (~roi_df['image_id'].isin(selected_images))
    ]
    if len(sample3) > 0:
        selected_images.append(sample3.iloc[0]['image_id'])
    
    # 4. Image with multiple ROIs
    roi_counts = roi_df.groupby('image_id').size()
    multi_roi_images = roi_counts[roi_counts >= 5].index.tolist()
    for img_id in multi_roi_images:
        if img_id not in selected_images:
            selected_images.append(img_id)
            break
    
    # 5. Class 2
    sample5 = roi_df[
        (roi_df['class_id'] == 2) &
        (~roi_df['image_id'].isin(selected_images))
    ]
    if len(sample5) > 0:
        selected_images.append(sample5.iloc[0]['image_id'])
    
    # Ensure we have exactly 5
    if len(selected_images) < 5:
        # Fill with any remaining images
        remaining = roi_df[~roi_df['image_id'].isin(selected_images)]['image_id'].unique()
        selected_images.extend(remaining[:5 - len(selected_images)])
    
    selected_images = selected_images[:5]
    
    print("\n" + "=" * 80)
    print("ROI MATCHING VISUALIZATION")
    print("=" * 80)
    print(f"\nSelected {len(selected_images)} diverse samples:")
    for i, img_id in enumerate(selected_images, 1):
        img_rois = roi_df[roi_df['image_id'] == img_id]
        classes = img_rois['class_id'].unique()
        print(f"  {i}. {img_id} - {len(img_rois)} ROIs, Classes: {list(classes)}")
    
    # Create visualizations
    visualizer.create_summary_visualization(selected_images, output_dir)
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Check outputs in: {output_dir}")


if __name__ == "__main__":
    main()
