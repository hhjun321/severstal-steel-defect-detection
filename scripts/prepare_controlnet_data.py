"""
Main script for preparing ControlNet training dataset.

This script implements the complete pipeline from PROJECT(prepare_control).md:
1. Multi-channel hint image generation
2. Hybrid prompt generation
3. Dataset sanity check (distribution & visual inspection)
4. Final packaging for ControlNet training (train.jsonl)

Usage:
    python scripts/prepare_controlnet_data.py --roi_metadata data/processed/roi_patches/roi_metadata.csv
    python scripts/prepare_controlnet_data.py --roi_metadata data/processed/roi_patches/roi_metadata.csv --skip_validation
"""
import argparse
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.controlnet_packager import ControlNetDatasetPackager
from src.preprocessing.hint_generator import HintImageGenerator
from src.preprocessing.prompt_generator import PromptGenerator
from src.utils.dataset_validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ControlNet training dataset from extracted ROIs'
    )
    parser.add_argument(
        '--roi_metadata',
        type=str,
        required=True,
        help='Path to ROI metadata CSV from ROI extraction'
    )
    parser.add_argument(
        '--train_images',
        type=str,
        default='train_images',
        help='Directory containing original training images'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='train.csv',
        help='Path to train.csv with RLE annotations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/controlnet_dataset',
        help='Output directory for ControlNet dataset'
    )
    parser.add_argument(
        '--prompt_style',
        type=str,
        choices=['simple', 'detailed', 'technical'],
        default='detailed',
        help='Prompt generation style'
    )
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Skip dataset validation step'
    )
    parser.add_argument(
        '--skip_hints',
        action='store_true',
        help='Skip hint image generation (only create prompts and jsonl)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    parser.add_argument(
        '--validation_samples',
        type=int,
        default=16,
        help='Number of samples for visual validation'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    roi_metadata_path = Path(args.roi_metadata)
    train_images_dir = Path(args.train_images)
    train_csv = Path(args.train_csv)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not roi_metadata_path.exists():
        print(f"Error: ROI metadata not found: {roi_metadata_path}")
        print("Please run extract_rois.py first to generate ROI metadata.")
        return
    
    if not train_csv.exists():
        print(f"Error: Training CSV not found: {train_csv}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ControlNet Training Data Preparation (PROJECT(prepare_control).md)")
    print("="*80)
    print(f"ROI metadata: {roi_metadata_path}")
    print(f"Train images: {train_images_dir}")
    print(f"Train CSV: {train_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Generate hints: {not args.skip_hints}")
    print(f"Run validation: {not args.skip_validation}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("="*80)
    
    # Load ROI metadata
    print("\n[Step 1/4] Loading ROI metadata...")
    roi_df = pd.read_csv(roi_metadata_path)
    print(f"Loaded {len(roi_df)} ROIs")
    
    # Dataset validation
    if not args.skip_validation:
        print("\n[Step 2/4] Validating dataset...")
        validator = DatasetValidator(output_dir=output_dir / 'validation')
        validation_report = validator.generate_full_report(
            roi_df,
            num_visual_samples=args.validation_samples
        )
        
        if validation_report['overall_status'] == 'WARNING':
            print("\n⚠️  Validation warnings detected. Review the reports before proceeding.")
            print(f"   Reports saved to: {output_dir / 'validation'}")
            
            response = input("\nDo you want to continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborting. Please address the issues and try again.")
                return
        else:
            print("\n✅ Dataset validation passed!")
    else:
        print("\n[Step 2/4] Skipping validation...")
    
    # Initialize components
    print("\n[Step 3/4] Initializing components...")
    hint_generator = HintImageGenerator(
        enhance_linearity=True,
        enhance_background=True
    )
    prompt_generator = PromptGenerator(
        style=args.prompt_style,
        include_class_id=True
    )
    packager = ControlNetDatasetPackager(
        hint_generator=hint_generator,
        prompt_generator=prompt_generator,
        prompt_style=args.prompt_style
    )
    
    # Package dataset
    print("\n[Step 4/4] Packaging dataset for ControlNet training...")
    packaged_dir = packager.package_dataset(
        roi_metadata_df=roi_df,
        train_images_dir=train_images_dir,
        train_csv=train_csv,
        output_dir=output_dir,
        create_hints=not args.skip_hints,
        max_samples=args.max_samples
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("✅ ControlNet Training Data Preparation Complete!")
    print("="*80)
    print(f"\nOutput directory: {packaged_dir}")
    print(f"\nGenerated files:")
    print(f"  - train.jsonl: Training data index")
    print(f"  - metadata.json: Complete dataset metadata")
    print(f"  - packaged_roi_metadata.csv: Updated ROI metadata with prompts")
    
    if not args.skip_hints:
        print(f"  - hints/: Multi-channel hint images")
    
    if not args.skip_validation:
        print(f"  - validation/: Validation reports and visualizations")
    
    print(f"\nNext steps:")
    print(f"  1. Review validation reports (if generated)")
    print(f"  2. Inspect sample hint images in {output_dir / 'hints'}")
    print(f"  3. Use train.jsonl for ControlNet training")
    print(f"  4. See PROJECT(control_net).md for training configuration")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
