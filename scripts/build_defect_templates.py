"""
Defect Template Library Builder

This script builds a defect template library from existing ROI metadata.
Templates are indexed by class and subtype for efficient sampling during augmentation.

Usage:
    python scripts/build_defect_templates.py --roi_metadata data/processed/roi_patches/roi_metadata.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import json
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.roi_suitability import ROISuitabilityEvaluator


class DefectTemplateBuilder:
    """
    Builds a searchable library of defect templates.
    """
    
    # Matching rules from ROISuitabilityEvaluator
    MATCHING_RULES = {
        'linear_scratch': {
            'vertical_stripe': 1.0,
            'horizontal_stripe': 1.0,
            'smooth': 0.7,
            'textured': 0.5,
            'complex_pattern': 0.3,
        },
        'elongated': {
            'vertical_stripe': 0.9,
            'horizontal_stripe': 0.9,
            'smooth': 0.8,
            'textured': 0.6,
            'complex_pattern': 0.4,
        },
        'compact_blob': {
            'smooth': 1.0,
            'textured': 0.7,
            'vertical_stripe': 0.5,
            'horizontal_stripe': 0.5,
            'complex_pattern': 0.6,
        },
        'irregular': {
            'complex_pattern': 1.0,
            'textured': 0.8,
            'smooth': 0.6,
            'vertical_stripe': 0.5,
            'horizontal_stripe': 0.5,
        },
        'general': {
            'smooth': 0.7,
            'textured': 0.7,
            'vertical_stripe': 0.7,
            'horizontal_stripe': 0.7,
            'complex_pattern': 0.7,
        }
    }
    
    def __init__(self, min_suitability=0.7):
        """
        Initialize template builder.
        
        Args:
            min_suitability: Minimum suitability score to include template
        """
        self.min_suitability = min_suitability
    
    def load_roi_metadata(self, roi_metadata_path):
        """
        Load ROI metadata CSV.
        
        Args:
            roi_metadata_path: Path to roi_metadata.csv
            
        Returns:
            DataFrame with ROI data
        """
        df = pd.read_csv(roi_metadata_path)
        
        # Filter by suitability
        df = df[df['suitability_score'] >= self.min_suitability]
        
        print(f"Loaded {len(df)} ROIs with suitability >= {self.min_suitability}")
        
        return df
    
    def build_template_library(self, roi_df):
        """
        Build template library organized by class and subtype.
        
        Args:
            roi_df: DataFrame with ROI metadata
            
        Returns:
            Dictionary with template library
        """
        templates = {
            'by_class': {},
            'by_subtype': {},
            'by_background': {},
            'all_templates': []
        }
        
        # Process each ROI
        for idx, row in roi_df.iterrows():
            template = {
                'template_id': idx,
                'image_id': row['image_id'],
                'class_id': int(row['class_id']),
                'region_id': int(row['region_id']),
                'roi_image_path': row['roi_image_path'],
                'roi_mask_path': row['roi_mask_path'],
                
                # Defect metrics
                'linearity': float(row['linearity']),
                'solidity': float(row['solidity']),
                'extent': float(row['extent']),
                'aspect_ratio': float(row['aspect_ratio']),
                'area': int(row['area']),
                
                # Classifications
                'defect_subtype': row['defect_subtype'],
                'background_type': row['background_type'],
                
                # Suitability scores
                'suitability_score': float(row['suitability_score']),
                'matching_score': float(row['matching_score']),
                'continuity_score': float(row['continuity_score']),
                'stability_score': float(row['stability_score']),
                
                # Compatible backgrounds (for augmentation)
                'compatible_backgrounds': self.get_compatible_backgrounds(
                    row['defect_subtype']
                )
            }
            
            # Add to all templates
            templates['all_templates'].append(template)
            
            # Index by class
            class_id = template['class_id']
            if class_id not in templates['by_class']:
                templates['by_class'][class_id] = []
            templates['by_class'][class_id].append(template)
            
            # Index by subtype
            subtype = template['defect_subtype']
            if subtype not in templates['by_subtype']:
                templates['by_subtype'][subtype] = []
            templates['by_subtype'][subtype].append(template)
            
            # Index by background
            bg_type = template['background_type']
            if bg_type not in templates['by_background']:
                templates['by_background'][bg_type] = []
            templates['by_background'][bg_type].append(template)
        
        return templates
    
    def get_compatible_backgrounds(self, defect_subtype, min_score=0.7):
        """
        Get list of compatible background types for a defect subtype.
        
        Args:
            defect_subtype: Defect sub-classification
            min_score: Minimum matching score
            
        Returns:
            List of compatible background types
        """
        if defect_subtype not in self.MATCHING_RULES:
            return ['smooth', 'textured', 'vertical_stripe', 'horizontal_stripe', 'complex_pattern']
        
        rules = self.MATCHING_RULES[defect_subtype]
        compatible = [bg_type for bg_type, score in rules.items() if score >= min_score]
        
        return compatible
    
    def compute_statistics(self, templates):
        """
        Compute statistics about template library.
        
        Args:
            templates: Template library dictionary
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_templates': len(templates['all_templates']),
            'templates_per_class': {},
            'templates_per_subtype': {},
            'templates_per_background': {},
            'avg_metrics': {
                'linearity': 0.0,
                'solidity': 0.0,
                'extent': 0.0,
                'aspect_ratio': 0.0,
                'suitability_score': 0.0
            }
        }
        
        # Count by class
        for class_id, class_templates in templates['by_class'].items():
            stats['templates_per_class'][int(class_id)] = len(class_templates)
        
        # Count by subtype
        for subtype, subtype_templates in templates['by_subtype'].items():
            stats['templates_per_subtype'][subtype] = len(subtype_templates)
        
        # Count by background
        for bg_type, bg_templates in templates['by_background'].items():
            stats['templates_per_background'][bg_type] = len(bg_templates)
        
        # Compute average metrics
        all_templates = templates['all_templates']
        if len(all_templates) > 0:
            stats['avg_metrics']['linearity'] = np.mean([t['linearity'] for t in all_templates])
            stats['avg_metrics']['solidity'] = np.mean([t['solidity'] for t in all_templates])
            stats['avg_metrics']['extent'] = np.mean([t['extent'] for t in all_templates])
            stats['avg_metrics']['aspect_ratio'] = np.mean([t['aspect_ratio'] for t in all_templates])
            stats['avg_metrics']['suitability_score'] = np.mean([t['suitability_score'] for t in all_templates])
        
        return stats
    
    def save_template_library(self, templates, output_dir):
        """
        Save template library to disk.
        
        Args:
            templates: Template library dictionary
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main template library
        templates_path = output_dir / 'templates_metadata.json'
        with open(templates_path, 'w') as f:
            json.dump(templates, f, indent=2)
        
        print(f"Saved template library to: {templates_path}")
        
        # Save statistics
        stats = self.compute_statistics(templates)
        stats_path = output_dir / 'template_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to: {stats_path}")
        
        # Save matching rules
        rules_path = output_dir / 'matching_rules.json'
        with open(rules_path, 'w') as f:
            json.dump(self.MATCHING_RULES, f, indent=2)
        
        print(f"Saved matching rules to: {rules_path}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Build defect template library from ROI metadata'
    )
    parser.add_argument(
        '--roi_metadata',
        type=str,
        required=True,
        help='Path to roi_metadata.csv from ROI extraction'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/defect_templates',
        help='Output directory for template library'
    )
    parser.add_argument(
        '--min_suitability',
        type=float,
        default=0.7,
        help='Minimum suitability score to include template'
    )
    
    args = parser.parse_args()
    
    # Validate input
    roi_metadata_path = Path(args.roi_metadata)
    if not roi_metadata_path.exists():
        print(f"Error: ROI metadata not found: {roi_metadata_path}")
        print("Please run scripts/extract_rois.py first.")
        return
    
    print("="*80)
    print("Defect Template Library Builder")
    print("="*80)
    print(f"ROI metadata: {args.roi_metadata}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min suitability: {args.min_suitability}")
    print("="*80)
    
    # Build template library
    builder = DefectTemplateBuilder(min_suitability=args.min_suitability)
    
    # Load ROI metadata
    print("\n[1/3] Loading ROI metadata...")
    roi_df = builder.load_roi_metadata(args.roi_metadata)
    
    # Build library
    print("\n[2/3] Building template library...")
    templates = builder.build_template_library(roi_df)
    
    # Save library
    print("\n[3/3] Saving template library...")
    stats = builder.save_template_library(templates, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("Template Library Built Successfully!")
    print("="*80)
    print(f"\nTotal templates: {stats['total_templates']}")
    
    print(f"\nTemplates per class:")
    for class_id in sorted(stats['templates_per_class'].keys()):
        count = stats['templates_per_class'][class_id]
        print(f"  Class {class_id}: {count}")
    
    print(f"\nTemplates per defect subtype:")
    for subtype, count in sorted(stats['templates_per_subtype'].items()):
        print(f"  {subtype}: {count}")
    
    print(f"\nTemplates per background type:")
    for bg_type, count in sorted(stats['templates_per_background'].items()):
        print(f"  {bg_type}: {count}")
    
    print(f"\nAverage metrics:")
    for metric, value in stats['avg_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
