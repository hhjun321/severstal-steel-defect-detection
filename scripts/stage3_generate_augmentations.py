"""
Stage 3: Augmentation Generation with Background-Defect Matching
3단계: 배경-결함 매칭을 통한 증강 생성

This script implements the production augmentation generation:
1. Loads background library (from background extraction)
2. Loads defect templates (from ROI extraction)
3. Matches compatible backgrounds with defects
4. Generates augmentation specifications
5. Creates ControlNet input files (images, masks, hints)

Usage:
    python scripts/stage3_generate_augmentations.py --n-samples 1000
"""

import argparse
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing.background_library import BackgroundLibrary
from preprocessing.augmentation_generator import AugmentationGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented samples with background-defect matching"
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of augmented samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--class-distribution',
        type=str,
        default='0.25,0.25,0.35,0.15',
        help='Class distribution as comma-separated ratios (default: 0.25,0.25,0.35,0.15)'
    )
    parser.add_argument(
        '--min-compatibility',
        type=float,
        default=0.5,
        help='Minimum compatibility score for template matching (default: 0.5)'
    )
    parser.add_argument(
        '--background-metadata',
        type=str,
        default=None,
        help='Path to background_metadata.csv (default: data/processed/background_patches/background_metadata.csv)'
    )
    parser.add_argument(
        '--defect-metadata',
        type=str,
        default=None,
        help='Path to roi_metadata.csv (default: data/processed/roi_patches/roi_metadata.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/augmented)'
    )
    
    args = parser.parse_args()
    
    # Parse class distribution
    try:
        ratios = [float(x) for x in args.class_distribution.split(',')]
        if len(ratios) != 4:
            raise ValueError("Must provide 4 ratios (one per class)")
        class_dist = {i+1: r for i, r in enumerate(ratios)}
    except Exception as e:
        print(f"ERROR: Invalid class distribution: {e}")
        print("Example: --class-distribution 0.25,0.25,0.35,0.15")
        sys.exit(1)
    
    # Default paths
    if args.background_metadata is None:
        bg_metadata_path = project_root / "data" / "processed" / "background_patches" / "background_metadata.csv"
    else:
        bg_metadata_path = Path(args.background_metadata)
    
    if args.defect_metadata is None:
        defect_metadata_path = project_root / "data" / "processed" / "roi_patches" / "roi_metadata.csv"
    else:
        defect_metadata_path = Path(args.defect_metadata)
    
    if args.output_dir is None:
        output_dir = project_root / "data" / "augmented"
    else:
        output_dir = Path(args.output_dir)
    
    # Validate paths
    if not bg_metadata_path.exists():
        print(f"ERROR: Background metadata not found: {bg_metadata_path}")
        print("\nPlease run background extraction first:")
        print("  python scripts/run_background_extraction.py")
        sys.exit(1)
    
    if not defect_metadata_path.exists():
        print(f"ERROR: Defect metadata not found: {defect_metadata_path}")
        print("\nPlease run ROI extraction first")
        sys.exit(1)
    
    # Print configuration
    print("="*80)
    print("STAGE 3: AUGMENTATION GENERATION")
    print("="*80)
    print(f"\nInput:")
    print(f"  Background metadata: {bg_metadata_path}")
    print(f"  Defect metadata:     {defect_metadata_path}")
    print(f"\nOutput:")
    print(f"  Augmented samples:   {output_dir}")
    print(f"\nConfiguration:")
    print(f"  Total samples:       {args.n_samples}")
    print(f"  Class distribution:")
    for class_id, ratio in class_dist.items():
        n_class = int(args.n_samples * ratio)
        print(f"    Class {class_id}: {ratio*100:5.1f}% ({n_class:4d} samples)")
    print(f"  Min compatibility:   {args.min_compatibility}")
    
    # Load background library
    print("\n" + "-"*80)
    print("Loading background library...")
    print("-"*80)
    bg_library = BackgroundLibrary(bg_metadata_path)
    bg_library.print_statistics()
    
    # Initialize augmentation generator
    print("\n" + "-"*80)
    print("Initializing augmentation generator...")
    print("-"*80)
    generator = AugmentationGenerator(
        background_library=bg_library,
        defect_metadata_path=defect_metadata_path,
        output_dir=output_dir,
        min_compatibility=args.min_compatibility
    )
    
    # Generate augmentations
    print("\n" + "-"*80)
    print("Generating augmented samples...")
    print("-"*80)
    results_df = generator.generate_batch(
        n_samples=args.n_samples,
        class_distribution=class_dist
    )
    
    # Summary
    print("\n" + "="*80)
    print("AUGMENTATION GENERATION COMPLETE")
    print("="*80)
    
    print(f"\nGenerated Files:")
    print(f"  Images:   {output_dir / 'images'} ({len(results_df)} files)")
    print(f"  Masks:    {output_dir / 'masks'} ({len(results_df)} files)")
    print(f"  Hints:    {output_dir / 'hints'} ({len(results_df)} files)")
    print(f"  Metadata: {output_dir / 'augmentation_metadata.csv'}")
    
    print(f"\nQuality Summary:")
    high_compat = (results_df['compatibility_score'] >= 0.8).sum()
    med_compat = ((results_df['compatibility_score'] >= 0.5) & 
                  (results_df['compatibility_score'] < 0.8)).sum()
    
    print(f"  High compatibility (≥0.8):  {high_compat} ({100*high_compat/len(results_df):.1f}%)")
    print(f"  Medium compatibility (0.5-0.8): {med_compat} ({100*med_compat/len(results_df):.1f}%)")
    
    print(f"\n  Mean compatibility:  {results_df['compatibility_score'].mean():.3f}")
    print(f"  Mean suitability:    {results_df['defect_suitability'].mean():.3f}")
    print(f"  Mean stability:      {results_df['background_stability'].mean():.3f}")
    
    print(f"\nNext Steps:")
    print(f"  1. Review generated samples in {output_dir}")
    print(f"  2. Train ControlNet using generated hints")
    print(f"  3. Generate final synthetic defects")
    print(f"  4. Validate quality and merge with original dataset")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
