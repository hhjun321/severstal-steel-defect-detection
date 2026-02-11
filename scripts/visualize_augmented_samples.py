"""
Visualization script for inspecting augmented samples.

This script creates visual comparisons of:
- Original images vs augmented images
- Ground truth masks vs augmented masks
- Quality scores and validation results
- Background-defect matching examples

Usage:
    # Visualize random samples
    python scripts/visualize_augmented_samples.py \\
        --augmented_dir data/augmented \\
        --output_dir visualizations \\
        --num_samples 20

    # Visualize specific samples
    python scripts/visualize_augmented_samples.py \\
        --augmented_dir data/augmented \\
        --output_dir visualizations \\
        --sample_ids aug_00000 aug_00123 aug_00456

    # Visualize by quality (show best/worst)
    python scripts/visualize_augmented_samples.py \\
        --augmented_dir data/augmented \\
        --output_dir visualizations \\
        --show_best 10 \\
        --show_worst 10

Author: CASDA Pipeline Team
Date: 2026-02-09
"""

import argparse
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


class AugmentedDataVisualizer:
    """Visualize augmented samples with metadata and quality scores."""
    
    def __init__(self, augmented_dir: str, output_dir: str):
        self.augmented_dir = augmented_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metadata
        self.metadata = self.load_metadata()
        self.quality_scores = self.load_quality_scores()
        
        # Class names
        self.class_names = {
            1: "Class 1",
            2: "Class 2",
            3: "Class 3",
            4: "Class 4"
        }
        
        # Background type colors
        self.bg_colors = {
            'smooth': '#3498db',
            'vertical_stripe': '#2ecc71',
            'horizontal_stripe': '#f39c12',
            'textured': '#e74c3c',
            'complex_pattern': '#9b59b6'
        }
    
    def load_metadata(self) -> List[Dict]:
        """Load augmented metadata."""
        metadata_path = os.path.join(self.augmented_dir, 'augmented_metadata.json')
        
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata not found at {metadata_path}")
            return []
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_quality_scores(self) -> Dict:
        """Load quality validation scores."""
        quality_path = os.path.join(self.augmented_dir, 'validation', 'quality_scores.json')
        
        if not os.path.exists(quality_path):
            print(f"Warning: Quality scores not found at {quality_path}")
            return {}
        
        with open(quality_path, 'r') as f:
            return json.load(f)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask as binary numpy array."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        return (mask > 127).astype(np.uint8)
    
    def visualize_single_sample(self, aug_id: str, save_path: str = None):
        """Visualize a single augmented sample with all details."""
        # Find metadata
        sample_meta = None
        for meta in self.metadata:
            if meta['aug_id'] == aug_id:
                sample_meta = meta
                break
        
        if sample_meta is None:
            print(f"Warning: Metadata not found for {aug_id}")
            return
        
        # Load image and mask
        img_path = os.path.join(self.augmented_dir, 'images', sample_meta['image_file'])
        mask_path = os.path.join(self.augmented_dir, 'masks', sample_meta['mask_file'])
        
        try:
            image = self.load_image(img_path)
            mask = self.load_mask(mask_path)
        except FileNotFoundError as e:
            print(f"Error loading {aug_id}: {e}")
            return
        
        # Get quality scores
        quality = self.quality_scores.get(aug_id, {})
        
        # Create visualization
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Augmented Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Mask
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Defect Mask', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        ax3.imshow(blended)
        ax3.set_title('Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. Metadata
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        
        meta_text = f"""
METADATA
--------
Aug ID: {aug_id}
Class ID: {sample_meta['class_id']} ({self.class_names[sample_meta['class_id']]})
Defect Subtype: {sample_meta['defect_subtype']}
Background Type: {sample_meta['background_type']}
Scale Factor: {sample_meta['scale_factor']:.2f}
Position: ({sample_meta['defect_position'][0]}, {sample_meta['defect_position'][1]})
Template ID: {sample_meta['template_id']}
Background ID: {sample_meta['background_id']}
        """
        ax4.text(0.1, 0.5, meta_text.strip(), fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Quality scores
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        if quality:
            quality_text = f"""
QUALITY SCORES
--------------
Overall: {quality.get('overall_score', 0):.3f}
Status: {quality.get('passed', False) and 'PASSED ✓' or 'REJECTED ✗'}

Blur: {quality.get('blur_score', 0):.3f}
Artifacts: {quality.get('artifact_score', 0):.3f}
Color: {quality.get('color_score', 0):.3f}
Consistency: {quality.get('consistency_score', 0):.3f}
Presence: {quality.get('presence_score', 0):.3f}
            """
            color = 'lightgreen' if quality.get('passed', False) else 'lightcoral'
        else:
            quality_text = "\nQUALITY SCORES\n--------------\nNot available"
            color = 'lightgray'
        
        ax5.text(0.1, 0.5, quality_text.strip(), fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        
        # 6. Statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        defect_area = np.sum(mask)
        total_area = mask.shape[0] * mask.shape[1]
        defect_ratio = defect_area / total_area
        
        stats_text = f"""
STATISTICS
----------
Image Size: {image.shape[1]}×{image.shape[0]}
Defect Pixels: {defect_area:,}
Defect Ratio: {defect_ratio*100:.2f}%
Mask Range: [{mask.min()}, {mask.max()}]
Image Mean: {image.mean():.1f}
Image Std: {image.std():.1f}
        """
        ax6.text(0.1, 0.5, stats_text.strip(), fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Overall title
        fig.suptitle(f'Augmented Sample: {aug_id}', fontsize=16, fontweight='bold')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_grid(self, sample_ids: List[str], save_path: str = None):
        """Visualize multiple samples in a grid."""
        n_samples = len(sample_ids)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, aug_id in enumerate(sample_ids):
            ax = axes[idx]
            
            # Find metadata
            sample_meta = None
            for meta in self.metadata:
                if meta['aug_id'] == aug_id:
                    sample_meta = meta
                    break
            
            if sample_meta is None:
                ax.axis('off')
                ax.text(0.5, 0.5, f'{aug_id}\nNot found', ha='center', va='center')
                continue
            
            # Load image
            img_path = os.path.join(self.augmented_dir, 'images', sample_meta['image_file'])
            mask_path = os.path.join(self.augmented_dir, 'masks', sample_meta['mask_file'])
            
            try:
                image = self.load_image(img_path)
                mask = self.load_mask(mask_path)
                
                # Create overlay
                overlay = image.copy()
                overlay[mask > 0] = [255, 0, 0]
                blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                
                ax.imshow(blended)
                
                # Add quality badge
                quality = self.quality_scores.get(aug_id, {})
                if quality:
                    score = quality.get('overall_score', 0)
                    passed = quality.get('passed', False)
                    color = 'green' if passed else 'red'
                    badge = f"{'✓' if passed else '✗'} {score:.2f}"
                    ax.text(0.05, 0.95, badge, transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                           color='white', fontweight='bold')
                
                # Title with metadata
                title = f"{aug_id}\nClass {sample_meta['class_id']} | {sample_meta['defect_subtype']}\n{sample_meta['background_type']}"
                ax.set_title(title, fontsize=8)
                ax.axis('off')
                
            except Exception as e:
                ax.axis('off')
                ax.text(0.5, 0.5, f'{aug_id}\nError: {str(e)[:20]}', ha='center', va='center', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Augmented Samples Grid ({n_samples} samples)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_quality_distribution(self, save_path: str = None):
        """Visualize quality score distribution."""
        if not self.quality_scores:
            print("No quality scores available")
            return
        
        scores = [q['overall_score'] for q in self.quality_scores.values()]
        passed = [q['passed'] for q in self.quality_scores.values()]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Threshold (0.7)')
        ax1.set_xlabel('Quality Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Pass/Fail pie chart
        ax2 = axes[1]
        pass_count = sum(passed)
        fail_count = len(passed) - pass_count
        
        colors = ['lightgreen', 'lightcoral']
        labels = [f'Passed: {pass_count} ({pass_count/len(passed)*100:.1f}%)',
                  f'Rejected: {fail_count} ({fail_count/len(passed)*100:.1f}%)']
        
        ax2.pie([pass_count, fail_count], labels=labels, colors=colors, autopct='',
                startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Validation Results', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_class_distribution(self, save_path: str = None):
        """Visualize class and background type distributions."""
        if not self.metadata:
            print("No metadata available")
            return
        
        # Count by class
        class_counts = {}
        for meta in self.metadata:
            class_id = meta['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Count by background type
        bg_counts = {}
        for meta in self.metadata:
            bg_type = meta['background_type']
            bg_counts[bg_type] = bg_counts.get(bg_type, 0) + 1
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class distribution
        ax1 = axes[0]
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        colors_class = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax1.bar(classes, counts, color=colors_class[:len(classes)], edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Class ID', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Augmented Samples by Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(classes)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Background type distribution
        ax2 = axes[1]
        bg_types = sorted(bg_counts.keys())
        bg_values = [bg_counts[bg] for bg in bg_types]
        colors_bg = [self.bg_colors.get(bg, '#95a5a6') for bg in bg_types]
        
        bars = ax2.barh(bg_types, bg_values, color=colors_bg, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Count', fontsize=12)
        ax2.set_ylabel('Background Type', fontsize=12)
        ax2.set_title('Augmented Samples by Background Type', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Visualize augmented samples')
    
    parser.add_argument('--augmented_dir', type=str, required=True,
                        help='Path to augmented data directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    
    # Sample selection
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of random samples to visualize')
    parser.add_argument('--sample_ids', type=str, nargs='+',
                        help='Specific sample IDs to visualize')
    parser.add_argument('--show_best', type=int, default=0,
                        help='Show N best quality samples')
    parser.add_argument('--show_worst', type=int, default=0,
                        help='Show N worst quality samples')
    
    # Visualization options
    parser.add_argument('--detailed', action='store_true',
                        help='Create detailed single-sample visualizations')
    parser.add_argument('--grid_only', action='store_true',
                        help='Only create grid visualization')
    parser.add_argument('--distributions', action='store_true',
                        help='Show quality and class distributions')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = AugmentedDataVisualizer(args.augmented_dir, args.output_dir)
    
    print(f"Loaded {len(visualizer.metadata)} samples")
    print(f"Quality scores available: {len(visualizer.quality_scores)}")
    
    # Select samples
    if args.sample_ids:
        sample_ids = args.sample_ids
    elif args.show_best > 0 or args.show_worst > 0:
        # Sort by quality
        sorted_samples = sorted(
            visualizer.quality_scores.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        sample_ids = []
        if args.show_best > 0:
            sample_ids.extend([s[0] for s in sorted_samples[:args.show_best]])
        if args.show_worst > 0:
            sample_ids.extend([s[0] for s in sorted_samples[-args.show_worst:]])
    else:
        # Random samples
        all_ids = [meta['aug_id'] for meta in visualizer.metadata]
        sample_ids = random.sample(all_ids, min(args.num_samples, len(all_ids)))
    
    print(f"Visualizing {len(sample_ids)} samples...")
    
    # Generate visualizations
    if args.distributions:
        print("Creating distribution plots...")
        visualizer.visualize_quality_distribution(
            os.path.join(args.output_dir, 'quality_distribution.png')
        )
        visualizer.visualize_class_distribution(
            os.path.join(args.output_dir, 'class_distribution.png')
        )
    
    if not args.grid_only and args.detailed:
        print("Creating detailed visualizations...")
        for aug_id in sample_ids:
            save_path = os.path.join(args.output_dir, f'{aug_id}_detailed.png')
            visualizer.visualize_single_sample(aug_id, save_path)
            print(f"  Saved: {save_path}")
    
    if not args.detailed or len(sample_ids) > 1:
        print("Creating grid visualization...")
        save_path = os.path.join(args.output_dir, 'samples_grid.png')
        visualizer.visualize_grid(sample_ids, save_path)
        print(f"  Saved: {save_path}")
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
