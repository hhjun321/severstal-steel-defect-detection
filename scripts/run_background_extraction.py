"""
Run Background Extraction Pipeline
배경 추출 파이프라인 실행

RESEARCH PROTOCOL:
This script extracts backgrounds from CLEAN images (NOT in train.csv).
- train.csv contains images WITH defects (used for ROI extraction)
- Clean images provide defect-free backgrounds for augmentation

연구 원칙:
이 스크립트는 깨끗한 이미지(train.csv에 없음)에서 배경을 추출합니다.
- train.csv는 결함이 있는 이미지 포함 (ROI 추출용)
- 깨끗한 이미지는 증강용 결함 없는 배경 제공

Usage:
    python scripts/run_background_extraction.py [--max-images N]
    
Arguments:
    --max-images: Maximum number of CLEAN images to process (default: all)
    --roi-size: Size of ROI patches (default: 512)
    --min-stability: Minimum stability score (default: 0.6)
"""

import argparse
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing.background_extraction import BackgroundExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract background ROIs from CLEAN images (not in train.csv)"
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of CLEAN images to process (default: all clean images)'
    )
    parser.add_argument(
        '--roi-size',
        type=int,
        default=512,
        help='Size of ROI patches in pixels (default: 512)'
    )
    parser.add_argument(
        '--min-stability',
        type=float,
        default=0.6,
        help='Minimum stability score to accept ROI (default: 0.6)'
    )
    parser.add_argument(
        '--rois-per-image',
        type=int,
        default=5,
        help='Number of diverse ROIs per image (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Paths
    train_csv = project_root / "train.csv"
    train_images = project_root / "train_images"
    output_dir = project_root / "data" / "processed" / "background_patches"
    
    # Validate paths
    if not train_csv.exists():
        print(f"ERROR: train.csv not found at {train_csv}")
        sys.exit(1)
    
    if not train_images.exists():
        print(f"ERROR: train_images directory not found at {train_images}")
        sys.exit(1)
    
    # Print configuration
    print("="*80)
    print("BACKGROUND EXTRACTION FROM CLEAN IMAGES (STAGE 3)")
    print("깨끗한 이미지에서 배경 추출 (3단계)")
    print("="*80)
    print("\nRESEARCH PROTOCOL:")
    print("  ✓ Extract backgrounds from images NOT in train.csv")
    print("  ✓ train.csv images have defects → used for ROI extraction")
    print("  ✓ Clean images provide defect-free backgrounds for augmentation")
    print("\n연구 원칙:")
    print("  ✓ train.csv에 없는 이미지에서 배경 추출")
    print("  ✓ train.csv 이미지는 결함 있음 → ROI 추출용")
    print("  ✓ 깨끗한 이미지는 증강용 결함 없는 배경 제공")
    print("\nConfiguration:")
    print(f"  Train CSV:        {train_csv}")
    print(f"  Train images:     {train_images}")
    print(f"  Output directory: {output_dir}")
    print(f"  ROI size:         {args.roi_size}x{args.roi_size}")
    print(f"  Min stability:    {args.min_stability}")
    print(f"  ROIs per image:   {args.rois_per_image}")
    print(f"  Max clean images: {args.max_images if args.max_images else 'all'}")
    
    # Initialize extractor
    print(f"\nInitializing BackgroundExtractor...")
    extractor = BackgroundExtractor(
        roi_size=args.roi_size,
        grid_size=64,
        min_stability=args.min_stability,
        rois_per_image=args.rois_per_image
    )
    
    # Run extraction on CLEAN images only
    print(f"\nStarting background extraction from clean images...")
    metadata_df = extractor.process_dataset(
        train_csv_path=train_csv,
        train_images_dir=train_images,
        output_dir=output_dir,
        max_images=args.max_images
    )
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    
    if len(metadata_df) > 0:
        print(f"\nResults:")
        print(f"  Total backgrounds extracted: {len(metadata_df)}")
        print(f"  Unique clean images used:    {metadata_df['image_id'].nunique()}")
        print(f"  Output directory:            {output_dir}")
        print(f"  Metadata file:               {output_dir / 'background_metadata.csv'}")
        
        print(f"\nQuality Metrics:")
        high_quality = (metadata_df['stability_score'] >= 0.8).sum()
        medium_quality = ((metadata_df['stability_score'] >= 0.6) & 
                         (metadata_df['stability_score'] < 0.8)).sum()
        low_quality = (metadata_df['stability_score'] < 0.6).sum()
        
        print(f"  High quality (≥0.8):     {high_quality} ({100*high_quality/len(metadata_df):.1f}%)")
        print(f"  Medium quality (0.6-0.8): {medium_quality} ({100*medium_quality/len(metadata_df):.1f}%)")
        print(f"  Low quality (<0.6):      {low_quality} ({100*low_quality/len(metadata_df):.1f}%)")
        
        print(f"\nNext Steps:")
        print(f"  1. Review background patches in: {output_dir}")
        print(f"  2. Use backgrounds for augmentation generation:")
        print(f"     python scripts/stage3_generate_augmentations.py --n-samples 1000")
        print(f"  3. Match with defect templates from roi_patches/")
    else:
        print("\nERROR: No backgrounds extracted!")
        print("Possible reasons:")
        print("  - All images are in train.csv (no clean images)")
        print("  - Stability threshold too high (try lowering --min-stability)")
        print("  - Image directory path incorrect")
        print("\nThis should not happen - you have 5,902 clean images available!")


if __name__ == '__main__':
    main()
