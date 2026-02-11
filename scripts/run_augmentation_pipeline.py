"""
Automated execution script for the full CASDA augmentation pipeline.

This script orchestrates all 5 phases of the augmentation pipeline:
1. Extract clean backgrounds
2. Build defect template library
3. Generate augmented data
4. Validate quality
5. Merge datasets

Usage:
    python scripts/run_augmentation_pipeline.py \
        --train_csv train.csv \
        --image_dir train_images \
        --model_path outputs/controlnet_training/best.pth \
        --roi_metadata data/processed/roi_patches/roi_metadata.csv \
        --output_base data \
        --num_samples 2500

Author: CASDA Pipeline Team
Date: 2026-02-09
"""

import argparse
import os
import sys
import subprocess
import time
import json
from pathlib import Path


class PipelineExecutor:
    """Orchestrates the full augmentation pipeline with progress tracking."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.phase_times = {}
        
        # Setup output directories
        self.backgrounds_dir = os.path.join(args.output_base, 'backgrounds')
        self.templates_dir = os.path.join(args.output_base, 'defect_templates')
        self.augmented_dir = os.path.join(args.output_base, 'augmented')
        self.final_dir = os.path.join(args.output_base, 'final_dataset')
        
    def log(self, message, level='INFO'):
        """Print formatted log message."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_command(self, cmd, phase_name):
        """Run a command and track execution time."""
        self.log(f"Starting {phase_name}...", 'INFO')
        self.log(f"Command: {' '.join(cmd)}", 'DEBUG')
        
        phase_start = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not self.args.verbose,
                text=True
            )
            
            phase_duration = time.time() - phase_start
            self.phase_times[phase_name] = phase_duration
            
            self.log(f"Completed {phase_name} in {phase_duration:.1f}s", 'SUCCESS')
            
            if self.args.verbose and result.stdout:
                print(result.stdout)
            
            return True
            
        except subprocess.CalledProcessError as e:
            phase_duration = time.time() - phase_start
            self.phase_times[phase_name] = phase_duration
            
            self.log(f"Failed {phase_name} after {phase_duration:.1f}s", 'ERROR')
            self.log(f"Error: {e}", 'ERROR')
            
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            
            return False
    
    def verify_prerequisites(self):
        """Verify all required files exist before starting."""
        self.log("Verifying prerequisites...", 'INFO')
        
        required_files = [
            (self.args.train_csv, "Original train.csv"),
            (self.args.image_dir, "Image directory"),
            (self.args.roi_metadata, "ROI metadata"),
            (self.args.model_path, "Trained ControlNet model"),
        ]
        
        missing = []
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                missing.append(f"  - {description}: {file_path}")
        
        if missing:
            self.log("Missing required files:", 'ERROR')
            for item in missing:
                print(item)
            return False
        
        self.log("All prerequisites verified ✓", 'SUCCESS')
        return True
    
    def phase1_extract_backgrounds(self):
        """Phase 1: Extract clean backgrounds."""
        cmd = [
            sys.executable,
            'scripts/extract_clean_backgrounds.py',
            '--train_csv', self.args.train_csv,
            '--image_dir', self.args.image_dir,
            '--output_dir', self.backgrounds_dir,
            '--patch_size', str(self.args.patch_size),
            '--patches_per_image', str(self.args.patches_per_image),
            '--min_quality', str(self.args.min_quality),
            '--stride', str(self.args.stride),
        ]
        
        return self.run_command(cmd, "Phase 1: Background Extraction")
    
    def phase2_build_templates(self):
        """Phase 2: Build defect template library."""
        cmd = [
            sys.executable,
            'scripts/build_defect_templates.py',
            '--roi_metadata', self.args.roi_metadata,
            '--output_dir', self.templates_dir,
            '--min_suitability', str(self.args.min_suitability),
        ]
        
        return self.run_command(cmd, "Phase 2: Template Library")
    
    def phase3_generate_augmented(self):
        """Phase 3: Generate augmented data."""
        cmd = [
            sys.executable,
            'scripts/generate_augmented_data.py',
            '--model_path', self.args.model_path,
            '--backgrounds_dir', self.backgrounds_dir,
            '--templates_dir', self.templates_dir,
            '--output_dir', self.augmented_dir,
            '--num_samples', str(self.args.num_samples),
            '--scale_min', str(self.args.scale_min),
            '--scale_max', str(self.args.scale_max),
            '--device', self.args.device,
            '--batch_size', str(self.args.batch_size),
            '--seed', str(self.args.seed),
        ]
        
        if self.args.samples_per_class:
            cmd.extend(['--samples_per_class', self.args.samples_per_class])
        
        if self.args.save_hints:
            cmd.append('--save_hints')
        
        return self.run_command(cmd, "Phase 3: Data Generation (CORE)")
    
    def phase4_validate_quality(self):
        """Phase 4: Validate augmented quality."""
        validation_dir = os.path.join(self.augmented_dir, 'validation')
        
        cmd = [
            sys.executable,
            'scripts/validate_augmented_quality.py',
            '--augmented_dir', self.augmented_dir,
            '--output_dir', validation_dir,
            '--min_quality_score', str(self.args.min_quality_score),
        ]
        
        if not self.args.skip_quality_checks:
            cmd.extend([
                '--check_blur',
                '--check_artifacts',
                '--check_color',
                '--check_defect_consistency',
                '--check_defect_presence',
            ])
        
        return self.run_command(cmd, "Phase 4: Quality Validation")
    
    def phase5_merge_datasets(self):
        """Phase 5: Merge original and augmented datasets."""
        output_csv = os.path.join(self.final_dir, 'train_augmented.csv')
        
        cmd = [
            sys.executable,
            'scripts/merge_datasets.py',
            '--original_csv', self.args.train_csv,
            '--original_images', self.args.image_dir,
            '--augmented_dir', self.augmented_dir,
            '--output_csv', output_csv,
            '--output_dir', self.final_dir,
        ]
        
        if self.args.use_only_passed:
            cmd.append('--use_only_passed')
        
        return self.run_command(cmd, "Phase 5: Dataset Merger")
    
    def print_summary(self):
        """Print execution summary."""
        total_duration = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        
        for phase, duration in self.phase_times.items():
            status = "✓" if duration > 0 else "✗"
            print(f"{status} {phase:<40} {duration:>8.1f}s")
        
        print("-"*70)
        print(f"{'TOTAL EXECUTION TIME':<40} {total_duration:>8.1f}s")
        print("="*70)
        
        # Print key outputs
        print("\nKEY OUTPUT FILES:")
        print(f"  - Backgrounds: {self.backgrounds_dir}")
        print(f"  - Templates: {self.templates_dir}")
        print(f"  - Augmented images: {self.augmented_dir}/images/")
        print(f"  - Final dataset: {self.final_dir}/train_augmented.csv")
        print("\nNext steps:")
        print("  1. Review quality report: data/augmented/validation/quality_report.txt")
        print("  2. Check dataset statistics: data/final_dataset/dataset_statistics.txt")
        print("  3. Use train_augmented.csv for model training")
        print("="*70 + "\n")
    
    def run(self):
        """Execute the full pipeline."""
        self.log("="*70, 'INFO')
        self.log("CASDA Augmentation Pipeline - Automated Execution", 'INFO')
        self.log("="*70, 'INFO')
        
        # Verify prerequisites
        if not self.verify_prerequisites():
            self.log("Pipeline aborted due to missing prerequisites", 'ERROR')
            return False
        
        # Execute phases sequentially
        phases = [
            (self.phase1_extract_backgrounds, "Phase 1"),
            (self.phase2_build_templates, "Phase 2"),
            (self.phase3_generate_augmented, "Phase 3"),
            (self.phase4_validate_quality, "Phase 4"),
            (self.phase5_merge_datasets, "Phase 5"),
        ]
        
        for i, (phase_func, phase_name) in enumerate(phases, 1):
            self.log(f"\n{'='*70}", 'INFO')
            self.log(f"{phase_name} ({i}/{len(phases)})", 'INFO')
            self.log(f"{'='*70}", 'INFO')
            
            success = phase_func()
            
            if not success:
                self.log(f"\nPipeline failed at {phase_name}", 'ERROR')
                self.log("Please check error messages above and refer to AUGMENTATION_PIPELINE_GUIDE.md", 'ERROR')
                self.print_summary()
                return False
        
        # Print summary
        self.print_summary()
        self.log("Pipeline completed successfully! ✓", 'SUCCESS')
        return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Automated CASDA augmentation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  python scripts/run_augmentation_pipeline.py \\
      --train_csv train.csv \\
      --image_dir train_images \\
      --model_path outputs/controlnet_training/best.pth \\
      --roi_metadata data/processed/roi_patches/roi_metadata.csv

  # Custom augmentation scale
  python scripts/run_augmentation_pipeline.py \\
      --train_csv train.csv \\
      --image_dir train_images \\
      --model_path outputs/controlnet_training/best.pth \\
      --roi_metadata data/processed/roi_patches/roi_metadata.csv \\
      --num_samples 5000 \\
      --batch_size 8

  # CPU-only execution
  python scripts/run_augmentation_pipeline.py \\
      --train_csv train.csv \\
      --image_dir train_images \\
      --model_path outputs/controlnet_training/best.pth \\
      --roi_metadata data/processed/roi_patches/roi_metadata.csv \\
      --device cpu
        """
    )
    
    # Required arguments
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to original train.csv')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to train_images/ directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained ControlNet model (.pth)')
    parser.add_argument('--roi_metadata', type=str, required=True,
                        help='Path to ROI metadata CSV (from extract_rois.py)')
    
    # Output configuration
    parser.add_argument('--output_base', type=str, default='data',
                        help='Base directory for all outputs (default: data)')
    
    # Phase 1: Background extraction
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Background patch size (default: 512)')
    parser.add_argument('--patches_per_image', type=int, default=5,
                        help='Max patches per image (default: 5)')
    parser.add_argument('--stride', type=int, default=256,
                        help='Sliding window stride (default: 256)')
    parser.add_argument('--min_quality', type=float, default=0.7,
                        help='Min background quality (default: 0.7)')
    
    # Phase 2: Template library
    parser.add_argument('--min_suitability', type=float, default=0.7,
                        help='Min defect-background suitability (default: 0.7)')
    
    # Phase 3: Data generation
    parser.add_argument('--num_samples', type=int, default=2500,
                        help='Total samples to generate (default: 2500)')
    parser.add_argument('--samples_per_class', type=str, default=None,
                        help='Per-class samples as JSON dict, e.g., \'{"1":625,"2":625,"3":625,"4":625}\'')
    parser.add_argument('--scale_min', type=float, default=0.8,
                        help='Min defect size scale (default: 0.8)')
    parser.add_argument('--scale_max', type=float, default=1.0,
                        help='Max defect size scale (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for ControlNet inference (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_hints', action='store_true',
                        help='Save multi-channel hints (for debugging)')
    
    # Phase 4: Quality validation
    parser.add_argument('--min_quality_score', type=float, default=0.7,
                        help='Min quality score for validation (default: 0.7)')
    parser.add_argument('--skip_quality_checks', action='store_true',
                        help='Skip all quality checks (not recommended)')
    
    # Phase 5: Dataset merger
    parser.add_argument('--use_only_passed', action='store_true',
                        help='Use only validation-passed samples')
    
    # General options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create executor and run pipeline
    executor = PipelineExecutor(args)
    success = executor.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
