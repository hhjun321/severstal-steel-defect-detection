"""
Clean Background Extraction Script

This script extracts defect-free background regions from training images
for use in data augmentation. It identifies clean areas, classifies background
types, and saves them organized by type.

Usage:
    python scripts/extract_clean_backgrounds.py --train_csv train.csv --image_dir train_images
    python scripts/extract_clean_backgrounds.py --train_csv train.csv --max_images 100
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import json
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.background_characterization import BackgroundAnalyzer
from src.utils.rle_utils import get_all_masks_for_image


class BackgroundExtractor:
    """
    Extracts clean background regions from training images.
    """
    
    def __init__(self, patch_size=512, min_quality=0.7, 
                 overlap=0.5, blur_threshold=100.0):
        """
        Initialize background extractor.
        
        Args:
            patch_size: Size of background patches
            min_quality: Minimum quality score (0-1)
            overlap: Overlap ratio between patches (0-1)
            blur_threshold: Laplacian variance threshold for blur detection
        """
        self.patch_size = patch_size
        self.min_quality = min_quality
        self.overlap = overlap
        self.blur_threshold = blur_threshold
        
        # Initialize background analyzer
        self.bg_analyzer = BackgroundAnalyzer(
            grid_size=64,
            variance_threshold=100.0,
            edge_threshold=0.3
        )
    
    def detect_blur(self, image):
        """
        Detect if image is blurry using Laplacian variance.
        
        Args:
            image: Grayscale image
            
        Returns:
            True if image is sharp, False if blurry
        """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var >= self.blur_threshold
    
    def is_patch_clean(self, image, mask_image, x, y, size):
        """
        Check if a patch contains no defects.
        
        Args:
            image: Full image
            mask_image: Combined mask of all defects (H, W)
            x, y: Top-left corner of patch
            size: Patch size
            
        Returns:
            True if patch is clean (no defects)
        """
        h, w = image.shape[:2]
        
        # Check bounds
        if x + size > w or y + size > h:
            return False
        
        # Extract patch from mask
        mask_patch = mask_image[y:y+size, x:x+size]
        
        # Check if any defect pixels exist
        defect_ratio = np.sum(mask_patch > 0) / (size * size)
        
        return defect_ratio == 0.0  # Completely clean
    
    def compute_quality_score(self, patch):
        """
        Compute quality score for a background patch.
        
        Args:
            patch: Image patch (H, W, 3)
            
        Returns:
            Quality score (0-1)
        """
        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = patch
        
        # 1. Blur check (30%)
        is_sharp = self.detect_blur(gray)
        blur_score = 1.0 if is_sharp else 0.3
        
        # 2. Contrast check (30%)
        contrast = np.std(gray) / 128.0  # Normalize to 0-1
        contrast_score = min(contrast, 1.0)
        
        # 3. Brightness check (20%)
        mean_brightness = np.mean(gray) / 255.0
        # Prefer medium brightness (0.3-0.7)
        if 0.3 <= mean_brightness <= 0.7:
            brightness_score = 1.0
        else:
            brightness_score = 0.7
        
        # 4. Noise check (20%)
        # Use local variance as noise indicator
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        noise_level = np.mean(local_var) / 100.0  # Normalize
        noise_score = 1.0 - min(noise_level, 1.0)
        
        # Weighted combination
        quality = (
            0.30 * blur_score +
            0.30 * contrast_score +
            0.20 * brightness_score +
            0.20 * noise_score
        )
        
        return float(quality)
    
    def extract_patches_from_image(self, image_path, train_df, image_id):
        """
        Extract clean background patches from a single image.
        
        Args:
            image_path: Path to image
            train_df: Training dataframe with masks
            image_id: Image identifier
            
        Returns:
            List of (patch_image, patch_info) tuples
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Get all defect masks for this image
        masks = get_all_masks_for_image(image_id, train_df, shape=(h, w))
        
        # Combine all masks into single mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for class_id, mask in masks.items():
            combined_mask = np.maximum(combined_mask, mask)
        
        # Calculate step size with overlap
        step = int(self.patch_size * (1 - self.overlap))
        
        patches = []
        
        # Sliding window
        for y in range(0, h - self.patch_size + 1, step):
            for x in range(0, w - self.patch_size + 1, step):
                # Check if patch is clean
                if not self.is_patch_clean(image_rgb, combined_mask, x, y, self.patch_size):
                    continue
                
                # Extract patch
                patch = image_rgb[y:y+self.patch_size, x:x+self.patch_size]
                
                # Compute quality score
                quality = self.compute_quality_score(patch)
                
                if quality < self.min_quality:
                    continue
                
                # Analyze background type
                bg_analysis = self.bg_analyzer.analyze_image(patch)
                
                # Get dominant background type
                bg_map = bg_analysis['background_map']
                unique, counts = np.unique(bg_map, return_counts=True)
                dominant_bg_type = unique[np.argmax(counts)]
                
                # Average stability
                avg_stability = np.mean(bg_analysis['stability_map'])
                
                patch_info = {
                    'image_id': image_id,
                    'position': (x, y),
                    'size': self.patch_size,
                    'background_type': dominant_bg_type,
                    'quality_score': quality,
                    'stability_score': float(avg_stability)
                }
                
                patches.append((patch, patch_info))
        
        return patches
    
    def extract_backgrounds(self, image_dir, train_csv, output_dir,
                          patches_per_image=5, max_images=None):
        """
        Extract clean backgrounds from entire dataset.
        
        Args:
            image_dir: Directory with training images
            train_csv: Path to train.csv
            output_dir: Output directory
            patches_per_image: Target patches per image
            max_images: Maximum images to process (None = all)
            
        Returns:
            Dictionary with extraction statistics
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        train_df = pd.read_csv(train_csv)
        
        # Get unique image IDs
        all_image_ids = set(train_df['ImageId'].unique())
        
        # Also check for images without defects
        all_images = list(image_dir.glob('*.jpg'))
        for img_path in all_images:
            all_image_ids.add(img_path.name)
        
        all_image_ids = sorted(list(all_image_ids))
        
        if max_images:
            all_image_ids = all_image_ids[:max_images]
        
        print(f"\nProcessing {len(all_image_ids)} images...")
        
        # Statistics
        stats = {
            'total_images': len(all_image_ids),
            'total_patches': 0,
            'patches_by_type': {},
            'patches_by_quality': {
                'high': 0,      # quality >= 0.9
                'medium': 0,    # 0.7 <= quality < 0.9
                'low': 0        # quality < 0.7 (should be 0)
            }
        }
        
        inventory = []
        
        # Process each image
        for image_id in tqdm(all_image_ids, desc="Extracting backgrounds"):
            image_path = image_dir / image_id
            
            if not image_path.exists():
                continue
            
            # Extract patches
            patches = self.extract_patches_from_image(image_path, train_df, image_id)
            
            # Limit patches per image
            if len(patches) > patches_per_image:
                # Sort by quality and take top N
                patches = sorted(patches, key=lambda x: x[1]['quality_score'], reverse=True)
                patches = patches[:patches_per_image]
            
            # Save patches
            for idx, (patch, info) in enumerate(patches):
                bg_type = info['background_type']
                quality = info['quality_score']
                
                # Create type directory
                type_dir = output_dir / bg_type
                type_dir.mkdir(exist_ok=True)
                
                # Save patch
                patch_filename = f"{image_id.replace('.jpg', '')}_patch{idx}.png"
                patch_path = type_dir / patch_filename
                
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(patch_path), patch_bgr)
                
                # Update info
                info['patch_path'] = str(patch_path.relative_to(output_dir))
                inventory.append(info)
                
                # Update statistics
                stats['total_patches'] += 1
                stats['patches_by_type'][bg_type] = stats['patches_by_type'].get(bg_type, 0) + 1
                
                if quality >= 0.9:
                    stats['patches_by_quality']['high'] += 1
                elif quality >= 0.7:
                    stats['patches_by_quality']['medium'] += 1
                else:
                    stats['patches_by_quality']['low'] += 1
        
        # Save inventory
        inventory_path = output_dir / 'background_inventory.json'
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"\nSaved inventory to: {inventory_path}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract clean background regions from training images'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        required=True,
        help='Path to train.csv'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='train_images',
        help='Directory with training images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/backgrounds',
        help='Output directory for background patches'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=512,
        help='Size of background patches'
    )
    parser.add_argument(
        '--patches_per_image',
        type=int,
        default=5,
        help='Target number of patches per image'
    )
    parser.add_argument(
        '--min_quality',
        type=float,
        default=0.7,
        help='Minimum quality score threshold'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum images to process (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Clean Background Extraction")
    print("="*80)
    print(f"Train CSV: {args.train_csv}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Patches per image: {args.patches_per_image}")
    print(f"Min quality: {args.min_quality}")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print("="*80)
    
    # Create extractor
    extractor = BackgroundExtractor(
        patch_size=args.patch_size,
        min_quality=args.min_quality,
        overlap=0.5,
        blur_threshold=100.0
    )
    
    # Extract backgrounds
    stats = extractor.extract_backgrounds(
        image_dir=args.image_dir,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        patches_per_image=args.patches_per_image,
        max_images=args.max_images
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("Extraction Complete!")
    print("="*80)
    print(f"\nTotal images processed: {stats['total_images']}")
    print(f"Total patches extracted: {stats['total_patches']}")
    
    print(f"\nPatches by background type:")
    for bg_type, count in sorted(stats['patches_by_type'].items()):
        print(f"  {bg_type}: {count}")
    
    print(f"\nPatches by quality:")
    for quality_level, count in stats['patches_by_quality'].items():
        print(f"  {quality_level}: {count}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
