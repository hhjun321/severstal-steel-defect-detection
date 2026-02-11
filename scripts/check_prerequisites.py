"""
Prerequisites checker for CASDA augmentation pipeline.

Verifies all required files, dependencies, and system requirements
before running the augmentation pipeline.

Usage:
    python scripts/check_prerequisites.py

    # With detailed output
    python scripts/check_prerequisites.py --verbose

Author: CASDA Pipeline Team
Date: 2026-02-09
"""

import argparse
import os
import sys
from pathlib import Path


class PrerequisitesChecker:
    """Check all prerequisites for running the augmentation pipeline."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def log(self, message, level='INFO'):
        """Print formatted log message."""
        symbols = {'INFO': 'ℹ', 'SUCCESS': '✓', 'ERROR': '✗', 'WARNING': '⚠'}
        colors = {
            'INFO': '\033[94m',
            'SUCCESS': '\033[92m',
            'ERROR': '\033[91m',
            'WARNING': '\033[93m',
            'RESET': '\033[0m'
        }
        
        symbol = symbols.get(level, '•')
        color = colors.get(level, '')
        reset = colors['RESET']
        
        print(f"{color}{symbol} {message}{reset}")
    
    def check_file(self, path, description):
        """Check if a file exists."""
        if os.path.exists(path):
            self.log(f"{description}: Found at {path}", 'SUCCESS')
            self.checks_passed += 1
            return True
        else:
            self.log(f"{description}: NOT FOUND at {path}", 'ERROR')
            self.checks_failed += 1
            return False
    
    def check_directory(self, path, description, check_count=False, min_files=0):
        """Check if a directory exists and optionally verify file count."""
        if not os.path.exists(path):
            self.log(f"{description}: NOT FOUND at {path}", 'ERROR')
            self.checks_failed += 1
            return False
        
        if not os.path.isdir(path):
            self.log(f"{description}: Path exists but is not a directory: {path}", 'ERROR')
            self.checks_failed += 1
            return False
        
        if check_count:
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            if file_count < min_files:
                self.log(f"{description}: Found but only {file_count} files (expected ≥{min_files})", 'WARNING')
                self.warnings.append(f"{description} has fewer files than expected")
                self.checks_passed += 1
                return True
            else:
                self.log(f"{description}: Found with {file_count} files", 'SUCCESS')
                self.checks_passed += 1
                return True
        else:
            self.log(f"{description}: Found at {path}", 'SUCCESS')
            self.checks_passed += 1
            return True
    
    def check_python_package(self, package_name, import_name=None):
        """Check if a Python package is installed."""
        if import_name is None:
            import_name = package_name
        
        try:
            __import__(import_name)
            
            if self.verbose:
                try:
                    pkg = __import__(import_name)
                    version = getattr(pkg, '__version__', 'unknown')
                    self.log(f"Package '{package_name}': Installed (v{version})", 'SUCCESS')
                except:
                    self.log(f"Package '{package_name}': Installed", 'SUCCESS')
            else:
                self.log(f"Package '{package_name}': Installed", 'SUCCESS')
            
            self.checks_passed += 1
            return True
        except ImportError:
            self.log(f"Package '{package_name}': NOT INSTALLED", 'ERROR')
            self.checks_failed += 1
            return False
    
    def check_cuda(self):
        """Check CUDA availability."""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else 'Unknown'
                
                self.log(f"CUDA: Available ({device_count} device(s), {device_name})", 'SUCCESS')
                
                if self.verbose:
                    for i in range(device_count):
                        props = torch.cuda.get_device_properties(i)
                        vram_gb = props.total_memory / (1024**3)
                        print(f"    GPU {i}: {props.name}, {vram_gb:.1f} GB VRAM")
                
                self.checks_passed += 1
                return True
            else:
                self.log("CUDA: Not available (will use CPU - slower)", 'WARNING')
                self.warnings.append("CUDA not available, generation will be slow")
                self.checks_passed += 1
                return True
        except ImportError:
            self.log("CUDA: Cannot check (PyTorch not installed)", 'ERROR')
            self.checks_failed += 1
            return False
    
    def check_disk_space(self, path='.', required_gb=10):
        """Check available disk space."""
        try:
            import shutil
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024**3)
            
            if available_gb >= required_gb:
                self.log(f"Disk space: {available_gb:.1f} GB available (≥{required_gb} GB required)", 'SUCCESS')
                self.checks_passed += 1
                return True
            else:
                self.log(f"Disk space: Only {available_gb:.1f} GB available (<{required_gb} GB required)", 'WARNING')
                self.warnings.append(f"Low disk space: {available_gb:.1f} GB")
                self.checks_passed += 1
                return True
        except Exception as e:
            self.log(f"Disk space: Cannot check ({str(e)})", 'WARNING')
            self.checks_passed += 1
            return True
    
    def run_checks(self):
        """Run all prerequisite checks."""
        print("\n" + "="*70)
        print("CASDA PIPELINE - PREREQUISITES CHECK")
        print("="*70 + "\n")
        
        # Section 1: Required Files
        print("1. REQUIRED FILES")
        print("-" * 70)
        self.check_file('train.csv', 'Original train.csv')
        self.check_directory('train_images', 'Training images directory', check_count=True, min_files=1000)
        self.check_file('data/processed/roi_patches/roi_metadata.csv', 'ROI metadata')
        self.check_file('outputs/controlnet_training/best.pth', 'Trained ControlNet model')
        print()
        
        # Section 2: Python Dependencies
        print("2. PYTHON DEPENDENCIES")
        print("-" * 70)
        
        packages = [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('opencv-python', 'cv2'),
            ('scikit-image', 'skimage'),
            ('torch', 'torch'),
            ('torchvision', 'torchvision'),
            ('tqdm', 'tqdm'),
            ('pillow', 'PIL'),
            ('matplotlib', 'matplotlib'),
        ]
        
        for pkg_name, import_name in packages:
            self.check_python_package(pkg_name, import_name)
        print()
        
        # Section 3: System Requirements
        print("3. SYSTEM REQUIREMENTS")
        print("-" * 70)
        self.check_cuda()
        self.check_disk_space(required_gb=10)
        print()
        
        # Section 4: Pipeline Scripts
        print("4. PIPELINE SCRIPTS")
        print("-" * 70)
        scripts = [
            ('scripts/extract_clean_backgrounds.py', 'Phase 1: Background extraction'),
            ('scripts/build_defect_templates.py', 'Phase 2: Template library'),
            ('scripts/generate_augmented_data.py', 'Phase 3: Data generation'),
            ('scripts/validate_augmented_quality.py', 'Phase 4: Quality validation'),
            ('scripts/merge_datasets.py', 'Phase 5: Dataset merger'),
            ('scripts/run_augmentation_pipeline.py', 'Automated execution'),
            ('scripts/visualize_augmented_samples.py', 'Visualization tool'),
        ]
        
        for script_path, description in scripts:
            self.check_file(script_path, description)
        print()
        
        # Section 5: Support Modules
        print("5. SUPPORT MODULES")
        print("-" * 70)
        modules = [
            ('src/analysis/defect_characterization.py', 'Defect analysis'),
            ('src/analysis/background_characterization.py', 'Background analysis'),
            ('src/analysis/roi_suitability.py', 'ROI suitability'),
            ('src/preprocessing/hint_generator.py', 'Hint generation'),
            ('src/preprocessing/prompt_generator.py', 'Prompt generation'),
            ('src/utils/rle.py', 'RLE utilities'),
        ]
        
        for module_path, description in modules:
            self.check_file(module_path, description)
        print()
        
        # Summary
        print("="*70)
        print("SUMMARY")
        print("="*70)
        
        total_checks = self.checks_passed + self.checks_failed
        
        self.log(f"Total checks: {total_checks}", 'INFO')
        self.log(f"Passed: {self.checks_passed}", 'SUCCESS')
        
        if self.checks_failed > 0:
            self.log(f"Failed: {self.checks_failed}", 'ERROR')
        
        if self.warnings:
            self.log(f"Warnings: {len(self.warnings)}", 'WARNING')
            for warning in self.warnings:
                print(f"    • {warning}")
        
        print("="*70 + "\n")
        
        # Recommendations
        if self.checks_failed > 0:
            print("❌ PREREQUISITES NOT MET")
            print("\nPlease fix the following issues:\n")
            
            if not os.path.exists('train.csv'):
                print("  1. Download train.csv from Kaggle:")
                print("     https://www.kaggle.com/c/severstal-steel-defect-detection/data")
            
            if not os.path.exists('data/processed/roi_patches/roi_metadata.csv'):
                print("  2. Run ROI extraction first:")
                print("     python scripts/extract_rois.py --train_csv train.csv --image_dir train_images")
            
            if not os.path.exists('outputs/controlnet_training/best.pth'):
                print("  3. Train ControlNet model:")
                print("     python scripts/train_controlnet.py --data_dir data/processed/controlnet_dataset")
            
            # Check for missing packages
            missing_packages = []
            for pkg_name, _ in packages:
                try:
                    __import__(pkg_name.replace('-', '_'))
                except ImportError:
                    missing_packages.append(pkg_name)
            
            if missing_packages:
                print("  4. Install missing Python packages:")
                print(f"     pip install {' '.join(missing_packages)}")
            
            print()
            return False
        
        elif self.warnings:
            print("⚠️  PREREQUISITES MET WITH WARNINGS")
            print("\nYou can proceed, but be aware of the warnings above.")
            print("The pipeline may run slower or produce fewer samples than expected.\n")
            return True
        
        else:
            print("✅ ALL PREREQUISITES MET")
            print("\nYou're ready to run the augmentation pipeline!")
            print("\nQuick start:")
            print("  python scripts/run_augmentation_pipeline.py \\")
            print("      --train_csv train.csv \\")
            print("      --image_dir train_images \\")
            print("      --model_path outputs/controlnet_training/best.pth \\")
            print("      --roi_metadata data/processed/roi_patches/roi_metadata.csv\n")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check prerequisites for CASDA pipeline')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information')
    
    args = parser.parse_args()
    
    checker = PrerequisitesChecker(verbose=args.verbose)
    success = checker.run_checks()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
