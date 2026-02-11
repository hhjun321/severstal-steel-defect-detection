"""
Dataset Sanity Check Module

This module performs quality checks on the extracted ROI dataset before training.
According to PROJECT(prepare_control).md:

1. Distribution Check: Ensure data is not skewed toward specific sub-classes or backgrounds
2. Visual Check: Sample inspection for background continuity and defect positioning

These checks ensure the dataset is physically plausible for training.
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter


class DatasetValidator:
    """
    Validates ROI dataset quality before ControlNet training.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize dataset validator.
        
        Args:
            output_dir: Directory to save validation reports and visualizations
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def check_distribution(self, roi_metadata_df: pd.DataFrame) -> Dict:
        """
        Check data distribution across classes, subtypes, and backgrounds.
        
        Args:
            roi_metadata_df: DataFrame with ROI metadata
            
        Returns:
            Dictionary with distribution statistics and warnings
        """
        total_rois = len(roi_metadata_df)
        
        # Class distribution
        class_dist = roi_metadata_df['class_id'].value_counts().to_dict()
        class_percentages = {k: v/total_rois*100 for k, v in class_dist.items()}
        
        # Subtype distribution
        subtype_dist = roi_metadata_df['defect_subtype'].value_counts().to_dict()
        subtype_percentages = {k: v/total_rois*100 for k, v in subtype_dist.items()}
        
        # Background distribution
        bg_dist = roi_metadata_df['background_type'].value_counts().to_dict()
        bg_percentages = {k: v/total_rois*100 for k, v in bg_dist.items()}
        
        # Check for imbalance
        warnings = []
        
        # Class imbalance (>60% in one class is concerning)
        max_class_pct = max(class_percentages.values())
        if max_class_pct > 60:
            warnings.append(f"⚠️  Class imbalance detected: Class {max(class_percentages, key=class_percentages.get)} has {max_class_pct:.1f}%")
        
        # Subtype imbalance
        max_subtype_pct = max(subtype_percentages.values())
        if max_subtype_pct > 50:
            warnings.append(f"⚠️  Subtype imbalance: '{max(subtype_percentages, key=subtype_percentages.get)}' has {max_subtype_pct:.1f}%")
        
        # Background imbalance
        max_bg_pct = max(bg_percentages.values())
        if max_bg_pct > 50:
            warnings.append(f"⚠️  Background imbalance: '{max(bg_percentages, key=bg_percentages.get)}' has {max_bg_pct:.1f}%")
        
        # Check for missing combinations
        combinations = roi_metadata_df.groupby(['defect_subtype', 'background_type']).size()
        
        # Ideal combinations that should exist
        ideal_combinations = [
            ('linear_scratch', 'vertical_stripe'),
            ('linear_scratch', 'horizontal_stripe'),
            ('compact_blob', 'smooth'),
            ('irregular', 'complex_pattern')
        ]
        
        missing_combinations = []
        for subtype, bg_type in ideal_combinations:
            if (subtype, bg_type) not in combinations.index:
                missing_combinations.append((subtype, bg_type))
        
        if missing_combinations:
            warnings.append(f"⚠️  Missing ideal combinations: {missing_combinations}")
        
        result = {
            'total_rois': total_rois,
            'class_distribution': class_dist,
            'class_percentages': class_percentages,
            'subtype_distribution': subtype_dist,
            'subtype_percentages': subtype_percentages,
            'background_distribution': bg_dist,
            'background_percentages': bg_percentages,
            'warnings': warnings,
            'is_balanced': len(warnings) == 0
        }
        
        return result
    
    def visualize_distributions(self, distribution_stats: Dict, save_path: Optional[Path] = None):
        """
        Create visualization of data distributions.
        
        Args:
            distribution_stats: Results from check_distribution()
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Class distribution
        ax1 = axes[0, 0]
        classes = sorted(distribution_stats['class_distribution'].keys())
        counts = [distribution_stats['class_distribution'][c] for c in classes]
        ax1.bar(classes, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (c, count) in enumerate(zip(classes, counts)):
            pct = distribution_stats['class_percentages'][c]
            ax1.text(c, count, f'{pct:.1f}%', ha='center', va='bottom')
        
        # Subtype distribution
        ax2 = axes[0, 1]
        subtypes = list(distribution_stats['subtype_distribution'].keys())
        counts = list(distribution_stats['subtype_distribution'].values())
        ax2.barh(subtypes, counts, color='coral', alpha=0.7)
        ax2.set_xlabel('Count')
        ax2.set_title('Defect Subtype Distribution')
        ax2.grid(axis='x', alpha=0.3)
        
        # Background distribution
        ax3 = axes[1, 0]
        bg_types = list(distribution_stats['background_distribution'].keys())
        counts = list(distribution_stats['background_distribution'].values())
        ax3.barh(bg_types, counts, color='seagreen', alpha=0.7)
        ax3.set_xlabel('Count')
        ax3.set_title('Background Type Distribution')
        ax3.grid(axis='x', alpha=0.3)
        
        # Warnings summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if distribution_stats['is_balanced']:
            status_text = "✅ Dataset is well-balanced"
            status_color = 'green'
        else:
            status_text = "⚠️  Imbalance detected"
            status_color = 'orange'
        
        summary_text = f"{status_text}\n\n"
        summary_text += f"Total ROIs: {distribution_stats['total_rois']}\n\n"
        
        if distribution_stats['warnings']:
            summary_text += "Warnings:\n"
            for warning in distribution_stats['warnings']:
                summary_text += f"  {warning}\n"
        else:
            summary_text += "No warnings - dataset looks good!"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved distribution visualization to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visual_check_sample(self, roi_metadata_df: pd.DataFrame,
                           num_samples: int = 16,
                           random_seed: int = 42) -> List[Dict]:
        """
        Perform visual inspection on random samples.
        
        Checks:
        - Background pattern continuity
        - Defect positioning (not too close to edges)
        - ROI quality
        
        Args:
            roi_metadata_df: DataFrame with ROI metadata
            num_samples: Number of samples to check
            random_seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries with inspection results
        """
        np.random.seed(random_seed)
        
        # Sample ROIs
        sample_indices = np.random.choice(
            len(roi_metadata_df),
            size=min(num_samples, len(roi_metadata_df)),
            replace=False
        )
        
        samples = roi_metadata_df.iloc[sample_indices]
        
        inspection_results = []
        
        for idx, row in samples.iterrows():
            result = {
                'image_id': row['image_id'],
                'class_id': row['class_id'],
                'region_id': row['region_id'],
                'defect_subtype': row['defect_subtype'],
                'background_type': row['background_type'],
                'suitability_score': row['suitability_score'],
                'issues': []
            }
            
            # Check if image/mask files exist
            if 'roi_image_path' in row and pd.notna(row['roi_image_path']):
                image_path = Path(row['roi_image_path'])
                if not image_path.exists():
                    result['issues'].append(f"Image file missing: {image_path}")
            
            if 'roi_mask_path' in row and pd.notna(row['roi_mask_path']):
                mask_path = Path(row['roi_mask_path'])
                if not mask_path.exists():
                    result['issues'].append(f"Mask file missing: {mask_path}")
            
            # Check defect positioning
            roi_bbox = row['roi_bbox']
            defect_bbox = row['defect_bbox']
            
            # Convert string tuples if needed
            if isinstance(roi_bbox, str):
                roi_bbox = eval(roi_bbox)
            if isinstance(defect_bbox, str):
                defect_bbox = eval(defect_bbox)
            
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
            def_x1, def_y1, def_x2, def_y2 = defect_bbox
            
            roi_width = roi_x2 - roi_x1
            roi_height = roi_y2 - roi_y1
            
            # Check if defect is too close to edges (within 10% margin)
            margin = 0.1
            if (def_x1 - roi_x1) < roi_width * margin:
                result['issues'].append("Defect too close to left edge")
            if (roi_x2 - def_x2) < roi_width * margin:
                result['issues'].append("Defect too close to right edge")
            if (def_y1 - roi_y1) < roi_height * margin:
                result['issues'].append("Defect too close to top edge")
            if (roi_y2 - def_y2) < roi_height * margin:
                result['issues'].append("Defect too close to bottom edge")
            
            # Check suitability score
            if row['suitability_score'] < 0.5:
                result['issues'].append(f"Low suitability score: {row['suitability_score']:.3f}")
            
            # Check continuity score
            if row['continuity_score'] < 0.5:
                result['issues'].append(f"Low continuity score: {row['continuity_score']:.3f}")
            
            result['has_issues'] = len(result['issues']) > 0
            inspection_results.append(result)
        
        return inspection_results
    
    def create_visual_inspection_report(self, roi_metadata_df: pd.DataFrame,
                                       inspection_results: List[Dict],
                                       save_path: Optional[Path] = None):
        """
        Create visual report showing sampled ROIs with their issues.
        
        Args:
            roi_metadata_df: DataFrame with ROI metadata
            inspection_results: Results from visual_check_sample()
            save_path: Path to save the report
        """
        num_samples = len(inspection_results)
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, result in enumerate(inspection_results):
            ax = axes[idx]
            
            # Try to load and display image
            row = roi_metadata_df[
                (roi_metadata_df['image_id'] == result['image_id']) &
                (roi_metadata_df['class_id'] == result['class_id']) &
                (roi_metadata_df['region_id'] == result['region_id'])
            ].iloc[0]
            
            if 'roi_image_path' in row and pd.notna(row['roi_image_path']):
                image_path = Path(row['roi_image_path'])
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                else:
                    ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, "No image path", ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Title with status
            status = "❌" if result['has_issues'] else "✅"
            title = f"{status} {result['defect_subtype']}\n{result['background_type']}"
            ax.set_title(title, fontsize=8)
            
            # Show issues as text
            if result['issues']:
                issues_text = "\n".join(result['issues'][:2])  # Show first 2 issues
                ax.text(0.05, 0.05, issues_text, transform=ax.transAxes,
                       fontsize=6, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                       color='white')
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Visual Inspection Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visual inspection report to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_full_report(self, roi_metadata_df: pd.DataFrame,
                            num_visual_samples: int = 16) -> Dict:
        """
        Generate complete validation report.
        
        Args:
            roi_metadata_df: DataFrame with ROI metadata
            num_visual_samples: Number of samples for visual inspection
            
        Returns:
            Dictionary with all validation results
        """
        print("="*80)
        print("Dataset Validation Report")
        print("="*80)
        
        # Distribution check
        print("\n[1/2] Checking data distribution...")
        dist_stats = self.check_distribution(roi_metadata_df)
        
        print(f"\nTotal ROIs: {dist_stats['total_rois']}")
        print(f"\nClass distribution:")
        for class_id in sorted(dist_stats['class_distribution'].keys()):
            count = dist_stats['class_distribution'][class_id]
            pct = dist_stats['class_percentages'][class_id]
            print(f"  Class {class_id}: {count} ({pct:.1f}%)")
        
        print(f"\nDefect subtype distribution:")
        for subtype, count in dist_stats['subtype_distribution'].items():
            pct = dist_stats['subtype_percentages'][subtype]
            print(f"  {subtype}: {count} ({pct:.1f}%)")
        
        print(f"\nBackground type distribution:")
        for bg_type, count in dist_stats['background_distribution'].items():
            pct = dist_stats['background_percentages'][bg_type]
            print(f"  {bg_type}: {count} ({pct:.1f}%)")
        
        if dist_stats['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in dist_stats['warnings']:
                print(f"  {warning}")
        else:
            print(f"\n✅ Distribution looks balanced!")
        
        # Visual inspection
        print(f"\n[2/2] Performing visual inspection on {num_visual_samples} samples...")
        inspection_results = self.visual_check_sample(roi_metadata_df, num_visual_samples)
        
        issues_found = sum(1 for r in inspection_results if r['has_issues'])
        print(f"\nInspection complete: {issues_found}/{len(inspection_results)} samples have issues")
        
        if issues_found > 0:
            print("\nSamples with issues:")
            for result in inspection_results:
                if result['has_issues']:
                    print(f"  {result['image_id']} (Class {result['class_id']}): {', '.join(result['issues'][:2])}")
        
        # Generate visualizations
        if self.output_dir:
            print("\n[3/3] Generating visualizations...")
            self.visualize_distributions(dist_stats, self.output_dir / 'distribution_analysis.png')
            self.create_visual_inspection_report(roi_metadata_df, inspection_results,
                                                 self.output_dir / 'visual_inspection.png')
        
        print("\n" + "="*80)
        print("Validation complete!")
        print("="*80)
        
        return {
            'distribution_stats': dist_stats,
            'inspection_results': inspection_results,
            'overall_status': 'PASS' if dist_stats['is_balanced'] and issues_found < len(inspection_results) * 0.2 else 'WARNING'
        }
