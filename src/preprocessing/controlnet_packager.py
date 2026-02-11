"""
ControlNet Dataset Packaging Module

This module packages the processed ROI data into ControlNet training format.
Creates:
- Multi-channel hint images
- train.jsonl with image paths and prompts
- Organized directory structure for training

Output format matches standard ControlNet training requirements.
"""
import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .hint_generator import HintImageGenerator
from .prompt_generator import PromptGenerator
from ..utils.rle_utils import get_all_masks_for_image


class ControlNetDatasetPackager:
    """
    Packages ROI data for ControlNet training.
    """
    
    def __init__(self, hint_generator: Optional[HintImageGenerator] = None,
                 prompt_generator: Optional[PromptGenerator] = None,
                 prompt_style: str = 'detailed'):
        """
        Initialize dataset packager.
        
        Args:
            hint_generator: HintImageGenerator instance
            prompt_generator: PromptGenerator instance
            prompt_style: Style for prompts ('simple', 'detailed', 'technical')
        """
        self.hint_generator = hint_generator or HintImageGenerator()
        self.prompt_generator = prompt_generator or PromptGenerator(style=prompt_style)
    
    def package_single_roi(self, roi_data: Dict, 
                          roi_image: np.ndarray,
                          roi_mask: np.ndarray,
                          output_dir: Path) -> Dict:
        """
        Package a single ROI with hint image and metadata.
        
        Args:
            roi_data: ROI metadata dictionary
            roi_image: ROI image array (H, W, 3)
            roi_mask: ROI mask array (H, W)
            output_dir: Output directory
            
        Returns:
            Updated roi_data with hint_path and prompt
        """
        # Generate hint image
        hint_image = self.hint_generator.generate_hint_image(
            roi_image=roi_image,
            roi_mask=roi_mask,
            defect_metrics=roi_data,
            background_type=roi_data.get('background_type', 'smooth'),
            stability_score=roi_data.get('stability_score', 0.5)
        )
        
        # Save hint image
        hint_dir = output_dir / 'hints'
        hint_dir.mkdir(parents=True, exist_ok=True)
        
        image_id = roi_data['image_id']
        class_id = roi_data['class_id']
        region_id = roi_data['region_id']
        hint_filename = f"{image_id}_class{class_id}_region{region_id}_hint.png"
        hint_path = hint_dir / hint_filename
        
        self.hint_generator.save_hint_image(hint_image, hint_path)
        
        # Generate prompt
        prompt = self.prompt_generator.generate_prompt(
            defect_subtype=roi_data.get('defect_subtype', 'general'),
            background_type=roi_data.get('background_type', 'smooth'),
            class_id=class_id,
            stability_score=roi_data.get('stability_score', 0.5),
            defect_metrics=roi_data,
            suitability_score=roi_data.get('suitability_score', 0.5)
        )
        
        negative_prompt = self.prompt_generator.generate_negative_prompt()
        
        # Update roi_data
        roi_data['hint_path'] = str(hint_path)
        roi_data['prompt'] = prompt
        roi_data['negative_prompt'] = negative_prompt
        
        return roi_data
    
    def create_train_jsonl(self, roi_metadata: List[Dict], 
                          output_path: Path,
                          relative_paths: bool = True,
                          base_dir: Optional[Path] = None):
        """
        Create train.jsonl file for ControlNet training.
        
        Format per line:
        {
            "source": "path/to/roi_image.png",
            "target": "path/to/roi_image.png",  # Same as source for this task
            "prompt": "a linear scratch on vertical striped metal surface...",
            "hint": "path/to/hint_image.png",
            "negative_prompt": "blurry, low quality..."
        }
        
        Args:
            roi_metadata: List of ROI metadata dictionaries
            output_path: Path to save train.jsonl
            relative_paths: Use relative paths instead of absolute
            base_dir: Base directory for relative paths
        """
        jsonl_lines = []
        
        for roi_data in roi_metadata:
            # Get paths
            source_path = roi_data.get('roi_image_path', '')
            hint_path = roi_data.get('hint_path', '')
            
            if relative_paths and base_dir:
                try:
                    source_path = Path(source_path).relative_to(base_dir)
                    hint_path = Path(hint_path).relative_to(base_dir)
                except ValueError:
                    pass  # Keep absolute if relative conversion fails
            
            entry = {
                "source": str(source_path),
                "target": str(source_path),  # For defect generation, target = source
                "prompt": roi_data.get('prompt', ''),
                "hint": str(hint_path),
                "negative_prompt": roi_data.get('negative_prompt', '')
            }
            
            jsonl_lines.append(entry)
        
        # Write JSONL file
        with open(output_path, 'w') as f:
            for entry in jsonl_lines:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Created train.jsonl with {len(jsonl_lines)} entries at: {output_path}")
    
    def create_metadata_json(self, roi_metadata: List[Dict], 
                            output_path: Path):
        """
        Create comprehensive metadata JSON file.
        
        Args:
            roi_metadata: List of ROI metadata dictionaries
            output_path: Path to save metadata.json
        """
        metadata = {
            'dataset_name': 'Severstal Steel Defect Detection - ControlNet Training Set',
            'total_samples': len(roi_metadata),
            'format': 'ControlNet training format with multi-channel hints',
            'channels': {
                'red': 'Defect mask with 4-indicator enhancement',
                'green': 'Background structure lines (edge information)',
                'blue': 'Background fine texture'
            },
            'prompt_structure': '[Defect characteristics] + [Background type] + [Surface condition]',
            'samples': roi_metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created metadata.json at: {output_path}")
    
    def package_dataset(self, roi_metadata_df: pd.DataFrame,
                       train_images_dir: Path,
                       train_csv: Path,
                       output_dir: Path,
                       create_hints: bool = True,
                       max_samples: Optional[int] = None) -> Path:
        """
        Package complete dataset for ControlNet training.
        
        Args:
            roi_metadata_df: DataFrame with ROI metadata from ROI extraction
            train_images_dir: Directory with original training images
            train_csv: Path to train.csv with RLE annotations
            output_dir: Output directory for packaged dataset
            create_hints: Whether to generate hint images
            max_samples: Maximum number of samples to package (for testing)
            
        Returns:
            Path to output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("ControlNet Dataset Packaging")
        print("="*80)
        print(f"Input: {len(roi_metadata_df)} ROIs")
        print(f"Output: {output_dir}")
        print(f"Create hints: {create_hints}")
        print("="*80)
        
        # Load train.csv for mask decoding
        train_df = pd.read_csv(train_csv)
        
        # Limit samples if specified
        if max_samples:
            roi_metadata_df = roi_metadata_df.head(max_samples)
        
        packaged_data = []
        
        # Process each ROI
        for idx, row in tqdm(roi_metadata_df.iterrows(), 
                            total=len(roi_metadata_df),
                            desc="Packaging ROIs"):
            
            roi_data = row.to_dict()
            
            # Load ROI image
            roi_image_path = Path(row['roi_image_path'])
            if not roi_image_path.exists():
                print(f"Warning: Image not found: {roi_image_path}")
                continue
            
            roi_image = cv2.imread(str(roi_image_path))
            roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            
            # Load ROI mask
            roi_mask_path = Path(row['roi_mask_path'])
            if not roi_mask_path.exists():
                print(f"Warning: Mask not found: {roi_mask_path}")
                continue
            
            roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
            roi_mask = (roi_mask > 0).astype(np.uint8)
            
            # Package this ROI
            if create_hints:
                roi_data = self.package_single_roi(
                    roi_data=roi_data,
                    roi_image=roi_image_rgb,
                    roi_mask=roi_mask,
                    output_dir=output_dir
                )
            else:
                # Just generate prompts
                prompt = self.prompt_generator.generate_prompt(
                    defect_subtype=roi_data.get('defect_subtype', 'general'),
                    background_type=roi_data.get('background_type', 'smooth'),
                    class_id=roi_data['class_id'],
                    stability_score=roi_data.get('stability_score', 0.5),
                    defect_metrics=roi_data,
                    suitability_score=roi_data.get('suitability_score', 0.5)
                )
                roi_data['prompt'] = prompt
                roi_data['negative_prompt'] = self.prompt_generator.generate_negative_prompt()
            
            packaged_data.append(roi_data)
        
        print(f"\nSuccessfully packaged {len(packaged_data)} ROIs")
        
        # Create train.jsonl
        print("\nCreating train.jsonl...")
        train_jsonl_path = output_dir / 'train.jsonl'
        self.create_train_jsonl(
            packaged_data, 
            train_jsonl_path,
            relative_paths=True,
            base_dir=output_dir.parent
        )
        
        # Create metadata.json
        print("\nCreating metadata.json...")
        metadata_path = output_dir / 'metadata.json'
        self.create_metadata_json(packaged_data, metadata_path)
        
        # Save updated ROI metadata
        print("\nSaving updated ROI metadata...")
        packaged_df = pd.DataFrame(packaged_data)
        packaged_csv = output_dir / 'packaged_roi_metadata.csv'
        packaged_df.to_csv(packaged_csv, index=False)
        print(f"Saved to: {packaged_csv}")
        
        # Create summary
        summary = {
            'total_packaged': len(packaged_data),
            'hints_created': create_hints,
            'output_directory': str(output_dir),
            'train_jsonl': str(train_jsonl_path),
            'metadata_json': str(metadata_path)
        }
        
        summary_path = output_dir / 'packaging_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("ControlNet Dataset Packaging Summary\n")
            f.write("="*80 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nSummary saved to: {summary_path}")
        print("\n" + "="*80)
        print("Packaging complete!")
        print("="*80)
        
        return output_dir
