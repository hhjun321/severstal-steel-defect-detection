"""
Augmented Data Generation Script

This script generates augmented defect images using a trained ControlNet model.
It combines clean backgrounds with defect templates to create physically plausible
synthetic defects.

Usage:
    python scripts/generate_augmented_data.py \
        --model_path outputs/controlnet_training/best.pth \
        --backgrounds_dir data/backgrounds \
        --templates_dir data/defect_templates \
        --num_samples 2500

Requirements:
    - Trained ControlNet model
    - Clean background patches (from extract_clean_backgrounds.py)
    - Defect templates (from build_defect_templates.py)
"""
import argparse
from pathlib import Path
import json
import numpy as np
import cv2
import random
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.hint_generator import HintImageGenerator
from src.preprocessing.prompt_generator import PromptGenerator
from scripts.train_controlnet import SimpleControlNet


class AugmentationGenerator:
    """
    Generates augmented defect images using ControlNet.
    """
    
    def __init__(self, model_path, backgrounds_dir, templates_dir,
                 device='cuda', scale_range=(0.8, 1.0)):
        """
        Initialize augmentation generator.
        
        Args:
            model_path: Path to trained ControlNet model
            backgrounds_dir: Directory with clean background patches
            templates_dir: Directory with defect templates
            device: Device for inference (cuda/cpu)
            scale_range: Scale range for defect size (min, max)
        """
        self.device = device
        self.scale_range = scale_range
        
        # Load model
        print(f"Loading ControlNet model from {model_path}...")
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Load backgrounds
        print(f"Loading backgrounds from {backgrounds_dir}...")
        self.backgrounds = self.load_backgrounds(backgrounds_dir)
        
        # Load templates
        print(f"Loading defect templates from {templates_dir}...")
        self.templates = self.load_templates(templates_dir)
        
        # Initialize generators
        self.hint_generator = HintImageGenerator(
            enhance_linearity=True,
            enhance_background=True
        )
        self.prompt_generator = PromptGenerator(
            style='detailed',
            include_class_id=True
        )
        
        print(f"\nInitialization complete!")
        print(f"  Backgrounds: {len(self.backgrounds)} patches")
        print(f"  Templates: {len(self.templates['all_templates'])} defects")
    
    def load_model(self, model_path):
        """Load trained ControlNet model."""
        model = SimpleControlNet(in_channels=3, out_channels=3, hint_channels=3)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    def load_backgrounds(self, backgrounds_dir):
        """Load background inventory."""
        backgrounds_dir = Path(backgrounds_dir)
        inventory_path = backgrounds_dir / 'background_inventory.json'
        
        if not inventory_path.exists():
            raise FileNotFoundError(f"Background inventory not found: {inventory_path}")
        
        with open(inventory_path, 'r') as f:
            inventory = json.load(f)
        
        # Convert paths to absolute
        for bg in inventory:
            bg['patch_path'] = str(backgrounds_dir / bg['patch_path'])
        
        return inventory
    
    def load_templates(self, templates_dir):
        """Load defect template library."""
        templates_dir = Path(templates_dir)
        templates_path = templates_dir / 'templates_metadata.json'
        
        if not templates_path.exists():
            raise FileNotFoundError(f"Template metadata not found: {templates_path}")
        
        with open(templates_path, 'r') as f:
            templates = json.load(f)
        
        return templates
    
    def is_compatible(self, background, template):
        """
        Check if background and defect template are compatible.
        
        Args:
            background: Background info dict
            template: Template info dict
            
        Returns:
            True if compatible
        """
        bg_type = background['background_type']
        compatible_bgs = template['compatible_backgrounds']
        
        return bg_type in compatible_bgs
    
    def sample_background_template_pair(self, class_id=None):
        """
        Sample a compatible background-template pair.
        
        Args:
            class_id: Target class ID (1-4), None for any class
            
        Returns:
            Tuple of (background, template) or (None, None) if no match
        """
        # Get templates for target class
        if class_id is not None:
            candidate_templates = self.templates['by_class'].get(str(class_id), [])
            if len(candidate_templates) == 0:
                return None, None
        else:
            candidate_templates = self.templates['all_templates']
        
        # Try to find compatible pair (with retries)
        max_retries = 50
        for _ in range(max_retries):
            template = random.choice(candidate_templates)
            
            # Get compatible backgrounds
            compatible_bg_types = template['compatible_backgrounds']
            
            # Filter backgrounds by compatible types
            compatible_backgrounds = [
                bg for bg in self.backgrounds
                if bg['background_type'] in compatible_bg_types
            ]
            
            if len(compatible_backgrounds) == 0:
                continue
            
            # Sample background (prefer high quality)
            # Weight by quality score
            qualities = np.array([bg['quality_score'] for bg in compatible_backgrounds])
            weights = qualities / qualities.sum()
            
            background_idx = np.random.choice(len(compatible_backgrounds), p=weights)
            background = compatible_backgrounds[background_idx]
            
            return background, template
        
        return None, None
    
    def create_synthetic_defect_mask(self, template, target_size=512):
        """
        Create synthetic defect mask based on template metrics.
        
        Args:
            template: Template info dict
            target_size: Size of output image
            
        Returns:
            Binary defect mask (H, W)
        """
        # Get template metrics
        linearity = template['linearity']
        solidity = template['solidity']
        aspect_ratio = template['aspect_ratio']
        
        # Determine defect size (with scale variation)
        scale_factor = random.uniform(*self.scale_range)
        base_size = int(template['area'] ** 0.5 * scale_factor)
        base_size = max(20, min(base_size, target_size // 3))  # Clamp size
        
        # Create mask based on subtype
        subtype = template['defect_subtype']
        
        if subtype == 'linear_scratch':
            mask = self.create_linear_mask(base_size, aspect_ratio, linearity)
        elif subtype == 'compact_blob':
            mask = self.create_blob_mask(base_size, solidity)
        elif subtype == 'irregular':
            mask = self.create_irregular_mask(base_size, solidity)
        elif subtype == 'elongated':
            mask = self.create_elongated_mask(base_size, aspect_ratio)
        else:  # general
            mask = self.create_general_mask(base_size)
        
        return mask
    
    def create_linear_mask(self, size, aspect_ratio, linearity):
        """Create linear scratch mask."""
        # Create elongated ellipse
        width = int(size * aspect_ratio)
        height = size
        
        mask = np.zeros((height * 2, width * 2), dtype=np.uint8)
        center = (width, height)
        axes = (width // 2, height // 2)
        
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Make more linear if needed
        if linearity > 0.8:
            # Apply thinning
            from skimage.morphology import skeletonize
            mask_bool = mask > 0
            skeleton = skeletonize(mask_bool)
            mask = (skeleton * 255).astype(np.uint8)
            # Dilate slightly
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def create_blob_mask(self, size, solidity):
        """Create compact blob mask."""
        mask = np.zeros((size * 2, size * 2), dtype=np.uint8)
        center = (size, size)
        radius = size // 2
        
        cv2.circle(mask, center, radius, 255, -1)
        
        # Add slight irregularity if solidity < 1.0
        if solidity < 0.95:
            noise = np.random.randint(0, 10, mask.shape, dtype=np.uint8)
            mask = cv2.bitwise_and(mask, mask, mask=255 - noise)
        
        return mask
    
    def create_irregular_mask(self, size, solidity):
        """Create irregular mask."""
        mask = np.zeros((size * 2, size * 2), dtype=np.uint8)
        
        # Create random polygon
        num_points = random.randint(5, 10)
        center = np.array([size, size])
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radii = np.random.uniform(size * 0.3, size * 0.6, num_points)
        
        points = []
        for angle, radius in zip(angles, radii):
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def create_elongated_mask(self, size, aspect_ratio):
        """Create elongated mask."""
        width = int(size * aspect_ratio * 0.7)
        height = size
        
        mask = np.zeros((height * 2, width * 2), dtype=np.uint8)
        center = (width, height)
        axes = (width // 2, height // 2)
        
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def create_general_mask(self, size):
        """Create general mask."""
        mask = np.zeros((size * 2, size * 2), dtype=np.uint8)
        center = (size, size)
        axes = (size // 2, int(size * 0.6))
        
        cv2.ellipse(mask, center, axes, random.randint(0, 180), 0, 360, 255, -1)
        
        return mask
    
    def place_defect_on_background(self, background_image, defect_mask):
        """
        Place defect mask on background at random position.
        
        Args:
            background_image: Background image (512, 512, 3)
            defect_mask: Defect mask (variable size)
            
        Returns:
            Tuple of (combined_mask, position)
        """
        bg_h, bg_w = background_image.shape[:2]
        mask_h, mask_w = defect_mask.shape
        
        # Ensure defect fits
        if mask_h >= bg_h or mask_w >= bg_w:
            # Resize defect to fit
            scale = min(bg_h / mask_h, bg_w / mask_w) * 0.8
            new_h = int(mask_h * scale)
            new_w = int(mask_w * scale)
            defect_mask = cv2.resize(defect_mask, (new_w, new_h))
            mask_h, mask_w = defect_mask.shape
        
        # Random position
        max_y = bg_h - mask_h
        max_x = bg_w - mask_w
        
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        
        # Create full-size mask
        full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        full_mask[y:y+mask_h, x:x+mask_w] = defect_mask
        
        return full_mask, (x, y, x + mask_w, y + mask_h)
    
    def create_hint_image(self, background_image, defect_mask, template):
        """
        Create multi-channel hint image.
        
        Args:
            background_image: Background RGB image (H, W, 3)
            defect_mask: Binary defect mask (H, W)
            template: Template info dict
            
        Returns:
            Hint image (H, W, 3)
        """
        defect_metrics = {
            'linearity': template['linearity'],
            'solidity': template['solidity'],
            'extent': template['extent'],
            'aspect_ratio': template['aspect_ratio']
        }
        
        hint_image = self.hint_generator.generate_hint_image(
            roi_image=background_image,
            roi_mask=defect_mask,
            defect_metrics=defect_metrics,
            background_type=template['background_type'],
            stability_score=template['stability_score']
        )
        
        return hint_image
    
    def create_prompt(self, template, background):
        """
        Create text prompt for generation.
        
        Args:
            template: Template info dict
            background: Background info dict
            
        Returns:
            Text prompt string
        """
        prompt = self.prompt_generator.generate_prompt(
            defect_subtype=template['defect_subtype'],
            background_type=background['background_type'],
            class_id=template['class_id'],
            stability_score=background['stability_score'],
            defect_metrics=template,
            suitability_score=template['suitability_score']
        )
        
        return prompt
    
    @torch.no_grad()
    def generate_sample(self, background, template):
        """
        Generate one augmented sample.
        
        Args:
            background: Background info dict
            template: Template info dict
            
        Returns:
            Dictionary with generated data, or None if failed
        """
        try:
            # Load background image
            bg_image = cv2.imread(background['patch_path'])
            if bg_image is None:
                return None
            bg_image_rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            
            # Create synthetic defect mask
            defect_mask = self.create_synthetic_defect_mask(template, target_size=512)
            
            # Place on background
            full_mask, bbox = self.place_defect_on_background(bg_image_rgb, defect_mask)
            
            # Create hint image
            hint_image = self.create_hint_image(bg_image_rgb, full_mask, template)
            
            # Create prompt
            prompt = self.create_prompt(template, background)
            
            # Prepare tensors for model
            bg_tensor = torch.from_numpy(bg_image_rgb).permute(2, 0, 1).float() / 255.0
            bg_tensor = (bg_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            bg_tensor = bg_tensor.unsqueeze(0).to(self.device)
            
            hint_tensor = torch.from_numpy(hint_image).permute(2, 0, 1).float() / 255.0
            hint_tensor = hint_tensor.unsqueeze(0).to(self.device)
            
            # Generate with ControlNet
            output_tensor = self.model(bg_tensor, hint_tensor)
            
            # Convert back to image
            output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)  # Denormalize
            generated_image = output_tensor[0].permute(1, 2, 0).cpu().numpy()
            generated_image = (generated_image * 255).astype(np.uint8)
            
            result = {
                'generated_image': generated_image,
                'defect_mask': full_mask,
                'hint_image': hint_image,
                'background_image': bg_image_rgb,
                'prompt': prompt,
                'class_id': template['class_id'],
                'defect_subtype': template['defect_subtype'],
                'background_type': background['background_type'],
                'template_id': template['template_id'],
                'bbox': bbox
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None
    
    def generate_augmented_dataset(self, output_dir, num_samples=2500,
                                  samples_per_class=None, save_hints=False):
        """
        Generate complete augmented dataset.
        
        Args:
            output_dir: Output directory
            num_samples: Total number of samples to generate
            samples_per_class: Dict with target per class, or None for equal
            save_hints: Whether to save hint images (for debugging)
            
        Returns:
            List of generated sample metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        images_dir = output_dir / 'images'
        masks_dir = output_dir / 'masks'
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        if save_hints:
            hints_dir = output_dir / 'hints'
            hints_dir.mkdir(exist_ok=True)
        
        # Determine samples per class
        if samples_per_class is None:
            samples_per_class = {1: num_samples // 4, 2: num_samples // 4,
                                3: num_samples // 4, 4: num_samples // 4}
        
        print(f"\nGenerating {num_samples} augmented samples...")
        print(f"Target per class: {samples_per_class}")
        
        generated_metadata = []
        class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        pbar = tqdm(total=num_samples, desc="Generating augmented data")
        
        sample_id = 0
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loop
        
        while sample_id < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Determine which class to generate
            needed_classes = [c for c in [1, 2, 3, 4]
                            if class_counts[c] < samples_per_class[c]]
            
            if len(needed_classes) == 0:
                break
            
            target_class = random.choice(needed_classes)
            
            # Sample compatible pair
            background, template = self.sample_background_template_pair(class_id=target_class)
            
            if background is None or template is None:
                continue
            
            # Generate sample
            result = self.generate_sample(background, template)
            
            if result is None:
                continue
            
            # Save generated image
            image_filename = f"aug_{sample_id:05d}.jpg"
            image_path = images_dir / image_filename
            generated_bgr = cv2.cvtColor(result['generated_image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), generated_bgr)
            
            # Save mask
            mask_filename = f"aug_{sample_id:05d}.png"
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), result['defect_mask'])
            
            # Save hint (optional)
            if save_hints:
                hint_filename = f"aug_{sample_id:05d}_hint.png"
                hint_path = hints_dir / hint_filename
                hint_bgr = cv2.cvtColor(result['hint_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(hint_path), hint_bgr)
            
            # Store metadata
            metadata = {
                'sample_id': sample_id,
                'image_filename': image_filename,
                'mask_filename': mask_filename,
                'class_id': result['class_id'],
                'defect_subtype': result['defect_subtype'],
                'background_type': result['background_type'],
                'template_id': result['template_id'],
                'prompt': result['prompt'],
                'bbox': result['bbox']
            }
            
            generated_metadata.append(metadata)
            
            # Update counts
            class_counts[target_class] += 1
            sample_id += 1
            pbar.update(1)
        
        pbar.close()
        
        # Save metadata
        metadata_path = output_dir / 'augmented_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(generated_metadata, f, indent=2)
        
        print(f"\nSaved metadata to: {metadata_path}")
        
        # Save generation log
        log_path = output_dir / 'generation_log.txt'
        with open(log_path, 'w') as f:
            f.write("Augmented Data Generation Log\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples generated: {sample_id}\n")
            f.write(f"Total attempts: {attempts}\n")
            f.write(f"Success rate: {sample_id / attempts * 100:.2f}%\n\n")
            f.write("Samples per class:\n")
            for class_id, count in sorted(class_counts.items()):
                f.write(f"  Class {class_id}: {count}\n")
        
        print(f"Saved generation log to: {log_path}")
        
        return generated_metadata, class_counts


def main():
    parser = argparse.ArgumentParser(
        description='Generate augmented defect data using ControlNet'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained ControlNet model (.pth)'
    )
    parser.add_argument(
        '--backgrounds_dir',
        type=str,
        required=True,
        help='Directory with clean background patches'
    )
    parser.add_argument(
        '--templates_dir',
        type=str,
        required=True,
        help='Directory with defect templates'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/augmented',
        help='Output directory for augmented data'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=2500,
        help='Total number of augmented samples to generate'
    )
    parser.add_argument(
        '--samples_per_class',
        type=str,
        default=None,
        help='Samples per class as JSON (e.g., \'{"1":625,"2":625,"3":625,"4":625}\')'
    )
    parser.add_argument(
        '--scale_min',
        type=float,
        default=0.8,
        help='Minimum scale factor for defect size'
    )
    parser.add_argument(
        '--scale_max',
        type=float,
        default=1.0,
        help='Maximum scale factor for defect size'
    )
    parser.add_argument(
        '--save_hints',
        action='store_true',
        help='Save hint images for debugging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse samples per class
    samples_per_class = None
    if args.samples_per_class:
        samples_per_class = json.loads(args.samples_per_class)
        samples_per_class = {int(k): v for k, v in samples_per_class.items()}
    
    print("="*80)
    print("Augmented Data Generation with ControlNet")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Backgrounds: {args.backgrounds_dir}")
    print(f"Templates: {args.templates_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Total samples: {args.num_samples}")
    if samples_per_class:
        print(f"Samples per class: {samples_per_class}")
    print(f"Scale range: {args.scale_min}-{args.scale_max}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Create generator
    generator = AugmentationGenerator(
        model_path=args.model_path,
        backgrounds_dir=args.backgrounds_dir,
        templates_dir=args.templates_dir,
        device=args.device,
        scale_range=(args.scale_min, args.scale_max)
    )
    
    # Generate augmented dataset
    metadata, class_counts = generator.generate_augmented_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        samples_per_class=samples_per_class,
        save_hints=args.save_hints
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"\nTotal samples generated: {len(metadata)}")
    print(f"\nSamples per class:")
    for class_id in sorted(class_counts.keys()):
        print(f"  Class {class_id}: {class_counts[class_id]}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
