"""
Dataset Merger Script

This script merges original and augmented datasets into a single training dataset.
It converts masks to RLE format and creates a unified train_augmented.csv file.

Usage:
    python scripts/merge_datasets.py \
        --original_csv train.csv \
        --original_images train_images \
        --augmented_dir data/augmented \
        --output_csv data/final_dataset/train_augmented.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import json
import shutil
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rle_utils import mask_to_rle


class DatasetMerger:
    """
    Merges original and augmented datasets.
    """
    
    def __init__(self, copy_images=False):
        """
        Initialize merger.
        
        Args:
            copy_images: Whether to copy images to output directory
        """
        self.copy_images = copy_images
    
    def load_augmented_metadata(self, augmented_dir):
        """
        Load augmented data metadata.
        
        Args:
            augmented_dir: Directory with augmented data
            
        Returns:
            List of augmented sample metadata
        """
        metadata_path = Path(augmented_dir) / 'augmented_metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Augmented metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def load_validation_results(self, augmented_dir):
        """
        Load validation results if available.
        
        Args:
            augmented_dir: Directory with augmented data
            
        Returns:
            Dict mapping filename to validation result, or None
        """
        validation_dir = Path(augmented_dir) / 'validation'
        results_path = validation_dir / 'quality_scores.json'
        
        if not results_path.exists():
            print("⚠️  Warning: No validation results found. Using all augmented samples.")
            return None
        
        with open(results_path, 'r') as f:
            validation_results = json.load(f)
        
        # Create lookup dict
        results_dict = {r['image_filename']: r for r in validation_results}
        
        return results_dict
    
    def convert_mask_to_rle(self, mask):
        """
        Convert binary mask to RLE format.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            RLE string
        """
        if np.sum(mask) == 0:
            return ""  # Empty mask
        
        rle = mask_to_rle(mask)
        return rle
    
    def create_augmented_entries(self, augmented_dir, metadata, validation_results, 
                                 use_only_passed=True):
        """
        Create CSV entries for augmented samples.
        
        Args:
            augmented_dir: Directory with augmented data
            metadata: List of augmented sample metadata
            validation_results: Dict with validation results
            use_only_passed: Whether to use only validated samples
            
        Returns:
            List of (image_id, class_id, encoded_pixels) tuples
        """
        augmented_dir = Path(augmented_dir)
        entries = []
        
        print(f"\nProcessing {len(metadata)} augmented samples...")
        
        for sample_meta in tqdm(metadata, desc="Converting masks to RLE"):
            image_filename = sample_meta['image_filename']
            
            # Check validation
            if validation_results and use_only_passed:
                if image_filename not in validation_results:
                    continue
                if not validation_results[image_filename]['passed']:
                    continue
            
            # Load mask
            mask_path = augmented_dir / 'masks' / sample_meta['mask_filename']
            
            if not mask_path.exists():
                print(f"Warning: Mask not found: {mask_path}")
                continue
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
            
            # Convert to RLE
            rle = self.convert_mask_to_rle(mask)
            
            if len(rle) == 0:
                print(f"Warning: Empty mask for {image_filename}")
                continue
            
            # Create entry (ImageId, ClassId, EncodedPixels)
            image_id = image_filename  # Use augmented filename as ImageId
            class_id = sample_meta['class_id']
            
            entries.append((image_id, class_id, rle))
        
        return entries
    
    def merge_datasets(self, original_csv, augmented_dir, output_csv,
                      use_only_passed=True):
        """
        Merge original and augmented datasets.
        
        Args:
            original_csv: Path to original train.csv
            augmented_dir: Directory with augmented data
            output_csv: Path to output merged CSV
            use_only_passed: Use only validated augmented samples
            
        Returns:
            Merged DataFrame and statistics
        """
        # Load original CSV
        print(f"\nLoading original dataset from {original_csv}...")
        original_df = pd.read_csv(original_csv)
        print(f"Original dataset: {len(original_df)} entries")
        
        # Count original samples
        original_images = original_df['ImageId'].nunique()
        original_with_defects = original_df[original_df['EncodedPixels'].notna()].groupby('ImageId').size().shape[0]
        
        print(f"  Total images: {original_images}")
        print(f"  Images with defects: {original_with_defects}")
        
        # Load augmented metadata
        print(f"\nLoading augmented metadata...")
        augmented_metadata = self.load_augmented_metadata(augmented_dir)
        
        # Load validation results
        validation_results = self.load_validation_results(augmented_dir)
        
        # Create augmented entries
        augmented_entries = self.create_augmented_entries(
            augmented_dir, augmented_metadata, validation_results, use_only_passed
        )
        
        print(f"\nAugmented samples to add: {len(augmented_entries)}")
        
        # Convert augmented entries to DataFrame
        augmented_df = pd.DataFrame(
            augmented_entries,
            columns=['ImageId', 'ClassId', 'EncodedPixels']
        )
        
        # Merge datasets
        print("\nMerging datasets...")
        merged_df = pd.concat([original_df, augmented_df], ignore_index=True)
        
        # Sort by ImageId and ClassId
        merged_df = merged_df.sort_values(['ImageId', 'ClassId']).reset_index(drop=True)
        
        # Save merged CSV
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        merged_df.to_csv(output_csv, index=False)
        print(f"\nSaved merged dataset to: {output_csv}")
        
        # Compute statistics
        stats = self.compute_statistics(original_df, augmented_df, merged_df)
        
        return merged_df, stats
    
    def compute_statistics(self, original_df, augmented_df, merged_df):
        """
        Compute dataset statistics.
        
        Args:
            original_df: Original dataset DataFrame
            augmented_df: Augmented dataset DataFrame
            merged_df: Merged dataset DataFrame
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'original': {
                'total_entries': len(original_df),
                'unique_images': original_df['ImageId'].nunique(),
                'images_with_defects': original_df[original_df['EncodedPixels'].notna()].groupby('ImageId').size().shape[0],
                'defects_per_class': {}
            },
            'augmented': {
                'total_entries': len(augmented_df),
                'unique_images': augmented_df['ImageId'].nunique(),
                'defects_per_class': {}
            },
            'merged': {
                'total_entries': len(merged_df),
                'unique_images': merged_df['ImageId'].nunique(),
                'total_images_with_defects': merged_df[merged_df['EncodedPixels'].notna()].groupby('ImageId').size().shape[0],
                'defects_per_class': {}
            },
            'augmentation_ratio': 0.0
        }
        
        # Count defects per class
        for class_id in [1, 2, 3, 4]:
            stats['original']['defects_per_class'][class_id] = len(
                original_df[(original_df['ClassId'] == class_id) & 
                           (original_df['EncodedPixels'].notna())]
            )
            
            stats['augmented']['defects_per_class'][class_id] = len(
                augmented_df[augmented_df['ClassId'] == class_id]
            )
            
            stats['merged']['defects_per_class'][class_id] = len(
                merged_df[(merged_df['ClassId'] == class_id) & 
                         (merged_df['EncodedPixels'].notna())]
            )
        
        # Augmentation ratio
        if stats['original']['unique_images'] > 0:
            stats['augmentation_ratio'] = stats['augmented']['unique_images'] / stats['original']['unique_images']
        
        return stats
    
    def save_statistics(self, stats, output_dir):
        """
        Save statistics to files.
        
        Args:
            stats: Statistics dictionary
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        stats_json_path = output_dir / 'dataset_statistics.json'
        with open(stats_json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics (JSON) to: {stats_json_path}")
        
        # Save text report
        stats_txt_path = output_dir / 'dataset_statistics.txt'
        with open(stats_txt_path, 'w') as f:
            f.write("Dataset Merger Statistics\n")
            f.write("="*80 + "\n\n")
            
            f.write("Original Dataset:\n")
            f.write(f"  Total entries: {stats['original']['total_entries']}\n")
            f.write(f"  Unique images: {stats['original']['unique_images']}\n")
            f.write(f"  Images with defects: {stats['original']['images_with_defects']}\n")
            f.write("  Defects per class:\n")
            for class_id, count in sorted(stats['original']['defects_per_class'].items()):
                f.write(f"    Class {class_id}: {count}\n")
            
            f.write("\nAugmented Dataset:\n")
            f.write(f"  Total entries: {stats['augmented']['total_entries']}\n")
            f.write(f"  Unique images: {stats['augmented']['unique_images']}\n")
            f.write("  Defects per class:\n")
            for class_id, count in sorted(stats['augmented']['defects_per_class'].items()):
                f.write(f"    Class {class_id}: {count}\n")
            
            f.write("\nMerged Dataset:\n")
            f.write(f"  Total entries: {stats['merged']['total_entries']}\n")
            f.write(f"  Unique images: {stats['merged']['unique_images']}\n")
            f.write(f"  Images with defects: {stats['merged']['total_images_with_defects']}\n")
            f.write("  Defects per class:\n")
            for class_id, count in sorted(stats['merged']['defects_per_class'].items()):
                f.write(f"    Class {class_id}: {count}\n")
            
            f.write(f"\nAugmentation ratio: {stats['augmentation_ratio']:.2%}\n")
        
        print(f"Saved statistics (TXT) to: {stats_txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge original and augmented datasets'
    )
    parser.add_argument(
        '--original_csv',
        type=str,
        required=True,
        help='Path to original train.csv'
    )
    parser.add_argument(
        '--original_images',
        type=str,
        default='train_images',
        help='Directory with original training images'
    )
    parser.add_argument(
        '--augmented_dir',
        type=str,
        required=True,
        help='Directory with augmented data'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='data/final_dataset/train_augmented.csv',
        help='Path to output merged CSV'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/final_dataset',
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--use_only_passed',
        action='store_true',
        default=True,
        help='Use only validated augmented samples (default: True)'
    )
    parser.add_argument(
        '--copy_images',
        action='store_true',
        help='Copy augmented images to output directory'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Dataset Merger")
    print("="*80)
    print(f"Original CSV: {args.original_csv}")
    print(f"Original images: {args.original_images}")
    print(f"Augmented data: {args.augmented_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use only passed samples: {args.use_only_passed}")
    print("="*80)
    
    # Create merger
    merger = DatasetMerger(copy_images=args.copy_images)
    
    # Merge datasets
    merged_df, stats = merger.merge_datasets(
        original_csv=args.original_csv,
        augmented_dir=args.augmented_dir,
        output_csv=args.output_csv,
        use_only_passed=args.use_only_passed
    )
    
    # Save statistics
    merger.save_statistics(stats, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("Dataset Merge Complete!")
    print("="*80)
    
    print(f"\nOriginal dataset:")
    print(f"  Entries: {stats['original']['total_entries']}")
    print(f"  Images: {stats['original']['unique_images']}")
    print(f"  Images with defects: {stats['original']['images_with_defects']}")
    
    print(f"\nAugmented dataset:")
    print(f"  Entries: {stats['augmented']['total_entries']}")
    print(f"  Images: {stats['augmented']['unique_images']}")
    
    print(f"\nMerged dataset:")
    print(f"  Entries: {stats['merged']['total_entries']}")
    print(f"  Images: {stats['merged']['unique_images']}")
    print(f"  Images with defects: {stats['merged']['total_images_with_defects']}")
    
    print(f"\nDefects per class:")
    print(f"{'Class':<8} {'Original':<12} {'Augmented':<12} {'Merged':<12} {'Increase':<12}")
    print("-" * 60)
    for class_id in [1, 2, 3, 4]:
        orig = stats['original']['defects_per_class'][class_id]
        aug = stats['augmented']['defects_per_class'][class_id]
        merged = stats['merged']['defects_per_class'][class_id]
        increase = ((merged - orig) / orig * 100) if orig > 0 else 0
        print(f"{class_id:<8} {orig:<12} {aug:<12} {merged:<12} {increase:>10.1f}%")
    
    print(f"\nAugmentation ratio: {stats['augmentation_ratio']:.2%}")
    print(f"\nOutput CSV: {args.output_csv}")
    print("="*80)
    
    # Note about image locations
    print("\n📝 Note: Augmented images remain in their original location:")
    print(f"   {Path(args.augmented_dir) / 'images'}")
    print("   Make sure to include this directory when training your model.")


if __name__ == '__main__':
    main()
