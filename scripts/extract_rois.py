"""
Main script for ROI extraction pipeline.

This script implements the complete pipeline from PROJECT(roi).md:
1. Analyze background characteristics (grid-based)
2. Analyze defect characteristics (4 indicators)
3. Evaluate ROI suitability (defect-background matching)
4. Extract and save ROI patches
5. Generate metadata CSV

Usage:
    python scripts/extract_rois.py --max_images 100  # Process first 100 images
    python scripts/extract_rois.py  # Process all images
"""
import argparse
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.roi_extraction import ROIExtractor
from src.analysis.defect_characterization import DefectCharacterizer
from src.analysis.background_characterization import BackgroundAnalyzer
from src.analysis.roi_suitability import ROISuitabilityEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Extract ROIs from Severstal steel defect dataset'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='train_images',
        help='Directory containing training images'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='train.csv',
        help='Path to train.csv with annotations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/roi_patches',
        help='Output directory for ROI data'
    )
    parser.add_argument(
        '--roi_size',
        type=int,
        default=512,
        help='Size of ROI patches'
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=64,
        help='Grid size for background analysis'
    )
    parser.add_argument(
        '--min_suitability',
        type=float,
        default=0.5,
        help='Minimum suitability score to accept ROI'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to process (for testing)'
    )
    parser.add_argument(
        '--no_save_patches',
        action='store_true',
        help='Do not save image/mask patches (only generate metadata)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    image_dir = Path(args.image_dir)
    train_csv = Path(args.train_csv)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    if not train_csv.exists():
        print(f"Error: Training CSV not found: {train_csv}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ROI Extraction Pipeline (PROJECT(roi).md)")
    print("="*80)
    print(f"Image directory: {image_dir}")
    print(f"Training CSV: {train_csv}")
    print(f"Output directory: {output_dir}")
    print(f"ROI size: {args.roi_size}")
    print(f"Grid size: {args.grid_size}")
    print(f"Min suitability: {args.min_suitability}")
    print(f"Max images: {args.max_images or 'all'}")
    print(f"Save patches: {not args.no_save_patches}")
    print("="*80)
    
    # Initialize analyzers
    print("\n[1/5] Initializing analyzers...")
    defect_analyzer = DefectCharacterizer()
    background_analyzer = BackgroundAnalyzer(
        grid_size=args.grid_size,
        variance_threshold=100.0,
        edge_threshold=0.3
    )
    roi_evaluator = ROISuitabilityEvaluator(defect_analyzer, background_analyzer)
    
    # Initialize extractor
    extractor = ROIExtractor(
        defect_analyzer=defect_analyzer,
        background_analyzer=background_analyzer,
        roi_evaluator=roi_evaluator,
        roi_size=args.roi_size,
        min_suitability=args.min_suitability
    )
    
    # Process dataset
    print("\n[2/5] Processing dataset...")
    roi_df = extractor.process_dataset(
        image_dir=image_dir,
        train_csv=train_csv,
        output_dir=output_dir,
        save_patches=not args.no_save_patches,
        max_images=args.max_images
    )
    
    # Save metadata to CSV
    print("\n[3/5] Saving metadata...")
    metadata_csv = output_dir / 'roi_metadata.csv'
    roi_df.to_csv(metadata_csv, index=False)
    print(f"Saved metadata to: {metadata_csv}")
    
    # Generate statistics
    print("\n[4/5] Generating statistics...")
    stats = {
        'total_rois': len(roi_df),
        'rois_per_class': roi_df['class_id'].value_counts().to_dict(),
        'rois_per_subtype': roi_df['defect_subtype'].value_counts().to_dict(),
        'rois_per_background': roi_df['background_type'].value_counts().to_dict(),
        'avg_suitability': roi_df['suitability_score'].mean(),
        'avg_matching': roi_df['matching_score'].mean(),
        'avg_continuity': roi_df['continuity_score'].mean(),
        'recommendation_counts': roi_df['recommendation'].value_counts().to_dict()
    }
    
    print(f"\nTotal ROIs extracted: {stats['total_rois']}")
    print(f"\nROIs per class:")
    for class_id, count in sorted(stats['rois_per_class'].items()):
        print(f"  Class {class_id}: {count}")
    
    print(f"\nROIs per defect subtype:")
    for subtype, count in stats['rois_per_subtype'].items():
        print(f"  {subtype}: {count}")
    
    print(f"\nROIs per background type:")
    for bg_type, count in stats['rois_per_background'].items():
        print(f"  {bg_type}: {count}")
    
    print(f"\nAverage scores:")
    print(f"  Suitability: {stats['avg_suitability']:.3f}")
    print(f"  Matching: {stats['avg_matching']:.3f}")
    print(f"  Continuity: {stats['avg_continuity']:.3f}")
    
    print(f"\nRecommendation distribution:")
    for rec, count in stats['recommendation_counts'].items():
        print(f"  {rec}: {count}")
    
    # Save statistics
    print("\n[5/5] Saving statistics...")
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("ROI Extraction Statistics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total ROIs extracted: {stats['total_rois']}\n\n")
        
        f.write("ROIs per class:\n")
        for class_id, count in sorted(stats['rois_per_class'].items()):
            f.write(f"  Class {class_id}: {count}\n")
        
        f.write("\nROIs per defect subtype:\n")
        for subtype, count in stats['rois_per_subtype'].items():
            f.write(f"  {subtype}: {count}\n")
        
        f.write("\nROIs per background type:\n")
        for bg_type, count in stats['rois_per_background'].items():
            f.write(f"  {bg_type}: {count}\n")
        
        f.write(f"\nAverage scores:\n")
        f.write(f"  Suitability: {stats['avg_suitability']:.3f}\n")
        f.write(f"  Matching: {stats['avg_matching']:.3f}\n")
        f.write(f"  Continuity: {stats['avg_continuity']:.3f}\n")
        
        f.write("\nRecommendation distribution:\n")
        for rec, count in stats['recommendation_counts'].items():
            f.write(f"  {rec}: {count}\n")
    
    print(f"Saved statistics to: {stats_path}")
    
    print("\n" + "="*80)
    print("ROI extraction complete!")
    print("="*80)


if __name__ == '__main__':
    main()
