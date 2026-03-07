"""
Augmented Data Quality Validation Script

This script validates the quality of generated augmented data.
It checks physical plausibility, background preservation, and defect consistency.

compose_casda_images.py 출력 형식 (metadata.json, image_path/mask_path 필드)과
레거시 package_casda_data.py 출력 형식 (augmented_metadata.json, image_filename/
mask_filename 필드) 양쪽 모두 호환.

Usage:
    python scripts/validate_augmented_quality.py \
        --augmented_dir data/augmented \
        --min_quality_score 0.7
"""
import argparse
from pathlib import Path
import json
import numpy as np
import cv2
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.defect_characterization import DefectCharacterizer


# ======================================================================
# 메타데이터 형식 호환 헬퍼 (compose / 레거시 package 양쪽 지원)
# ======================================================================

def _resolve_image_path(augmented_dir, metadata):
    """compose(image_path) 또는 레거시(image_filename) 필드에서 실제 파일 경로 반환."""
    if 'image_path' in metadata:
        return Path(augmented_dir) / metadata['image_path']          # "images/xxx.png"
    return Path(augmented_dir) / 'images' / metadata['image_filename']  # "xxx.png"


def _resolve_mask_path(augmented_dir, metadata):
    """compose(mask_path) 또는 레거시(mask_filename) 필드에서 실제 파일 경로 반환."""
    if 'mask_path' in metadata:
        return Path(augmented_dir) / metadata['mask_path']
    return Path(augmented_dir) / 'masks' / metadata['mask_filename']


def _get_image_name(metadata):
    """report 출력용 이미지 이름 반환 (compose/레거시 양쪽 대응)."""
    if 'image_path' in metadata:
        return Path(metadata['image_path']).name
    return metadata.get('image_filename', 'unknown')


class QualityValidator:
    """
    Validates quality of augmented defect images.
    """
    
    def __init__(self, min_quality_score=0.7):
        """
        Initialize validator.
        
        Args:
            min_quality_score: Minimum quality score threshold (0-1)
        """
        self.min_quality_score = min_quality_score
        self.defect_analyzer = DefectCharacterizer()
    
    def compute_blur_score(self, image):
        """
        Compute blur score using Laplacian variance.
        
        Args:
            image: Grayscale image
            
        Returns:
            Blur score (0-1), 1.0 = sharp
        """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # Normalize: assume 100+ is sharp
        score = min(laplacian_var / 100.0, 1.0)
        return float(score)
    
    def compute_artifact_score(self, image):
        """
        Detect artifacts using gradient analysis.
        
        Args:
            image: Grayscale image
            
        Returns:
            Artifact score (0-1), 1.0 = no artifacts
        """
        # Compute gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Check for abnormally high gradients (potential artifacts)
        high_gradient_ratio = np.sum(gradient_magnitude > 200) / gradient_magnitude.size
        
        # Low ratio = fewer artifacts = higher score
        score = 1.0 - min(high_gradient_ratio * 10, 1.0)
        return float(score)
    
    def compute_color_consistency_score(self, image):
        """
        Check color consistency (no unusual color shifts).
        
        Args:
            image: RGB image
            
        Returns:
            Color consistency score (0-1)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Check if colors are within reasonable range
        l_mean, a_mean, b_mean = np.mean(lab, axis=(0, 1))
        l_std, a_std, b_std = np.std(lab, axis=(0, 1))
        
        # Reasonable ranges (empirical)
        l_score = 1.0 if 30 <= l_mean <= 200 else 0.5
        ab_score = 1.0 if abs(a_mean) < 50 and abs(b_mean) < 50 else 0.5
        std_score = 1.0 if l_std > 10 else 0.5  # Not too flat
        
        score = (l_score + ab_score + std_score) / 3.0
        return float(score)
    
    def compute_defect_metrics_consistency(self, mask, expected_subtype):
        """
        Check if generated defect matches expected metrics.
        
        Args:
            mask: Binary defect mask
            expected_subtype: Expected defect subtype
            
        Returns:
            Consistency score (0-1)
        """
        # Analyze generated defect
        result = self.defect_analyzer.analyze_defect_region(mask)
        
        if result is None:
            return 0.0
        
        # Classify subtype
        actual_subtype = self.defect_analyzer.classify_defect_subtype(result)
        
        # Check if subtype matches
        if actual_subtype == expected_subtype:
            consistency = 1.0
        elif actual_subtype == 'general' or expected_subtype == 'general':
            consistency = 0.7  # General is acceptable fallback
        else:
            consistency = 0.3  # Mismatch
        
        return float(consistency)
    
    def validate_sample(self, generated_image, defect_mask, metadata):
        """
        Validate a single augmented sample.
        
        Args:
            generated_image: Generated RGB image
            defect_mask: Binary defect mask
            metadata: Sample metadata dict
            
        Returns:
            Dictionary with validation results
        """
        # Convert to grayscale for some checks
        gray = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
        
        # 1. Blur check (20%)
        blur_score = self.compute_blur_score(gray)
        
        # 2. Artifact check (20%)
        artifact_score = self.compute_artifact_score(gray)
        
        # 3. Color consistency (15%)
        color_score = self.compute_color_consistency_score(generated_image)
        
        # 4. Defect metrics consistency (25%)
        # compose 출력은 defect_subtype을 조건부로만 포함 (unknown이면 누락)
        defect_consistency = self.compute_defect_metrics_consistency(
            defect_mask, metadata.get('defect_subtype', 'general')
        )
        
        # 5. Defect presence check (20%)
        defect_ratio = np.sum(defect_mask > 0) / defect_mask.size
        if 0.001 < defect_ratio < 0.3:  # Reasonable defect size
            presence_score = 1.0
        else:
            presence_score = 0.5
        
        # Weighted overall quality score
        quality_score = (
            0.20 * blur_score +
            0.20 * artifact_score +
            0.15 * color_score +
            0.25 * defect_consistency +
            0.20 * presence_score
        )
        
        validation_result = {
            'quality_score': float(quality_score),
            'blur_score': blur_score,
            'artifact_score': artifact_score,
            'color_score': color_score,
            'defect_consistency': defect_consistency,
            'presence_score': presence_score,
            'passed': quality_score >= self.min_quality_score
        }
        
        return validation_result
    
    def validate_dataset(self, augmented_dir, output_dir, num_workers=0):
        """
        Validate entire augmented dataset.
        
        Args:
            augmented_dir: Directory with augmented data
            output_dir: Output directory for validation results
            num_workers: 병렬 워커 수 (0 = 순차 처리, >=2 = 병렬 처리)
            
        Returns:
            Validation statistics
        """
        augmented_dir = Path(augmented_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata — compose 출력(metadata.json) 우선, 레거시 fallback
        metadata_path = augmented_dir / 'metadata.json'
        if not metadata_path.exists():
            metadata_path = augmented_dir / 'augmented_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: tried metadata.json and "
                f"augmented_metadata.json in {augmented_dir}"
            )
        
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        print(f"\nValidating {len(all_metadata)} augmented samples...")
        
        validation_results = []
        passed_samples = []
        rejected_samples = []
        use_parallel = num_workers > 1 and len(all_metadata) > 10
        
        if use_parallel:
            # ── 병렬 경로: ProcessPoolExecutor ──
            print(f"  병렬 모드: {num_workers} workers")
            worker_args = []
            for metadata in all_metadata:
                image_path = str(_resolve_image_path(augmented_dir, metadata))
                mask_path = str(_resolve_mask_path(augmented_dir, metadata))
                worker_args.append((image_path, mask_path, metadata,
                                    self.min_quality_score))

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_validate_single_sample, arg)
                           for arg in worker_args]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Validating samples"):
                    result = future.result()
                    if result is None:
                        continue
                    validation_results.append(result)
                    if result['passed']:
                        passed_samples.append(result)
                    else:
                        rejected_samples.append(result)
        else:
            # ── 순차 경로 (기존 로직) ──
            for metadata in tqdm(all_metadata, desc="Validating samples"):
                # Load image and mask
                image_path = _resolve_image_path(augmented_dir, metadata)
                mask_path = _resolve_mask_path(augmented_dir, metadata)
                
                if not image_path.exists() or not mask_path.exists():
                    continue
                
                generated_image = cv2.imread(str(image_path))
                generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
                
                defect_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # Validate
                validation_result = self.validate_sample(
                    generated_image, defect_mask, metadata
                )
                
                # Combine with metadata
                result = {**metadata, **validation_result}
                validation_results.append(result)
                
                if validation_result['passed']:
                    passed_samples.append(result)
                else:
                    rejected_samples.append(result)
        
        # Compute statistics
        stats = self.compute_validation_statistics(validation_results)
        
        # Save results
        self.save_validation_results(
            validation_results, passed_samples, rejected_samples, stats, output_dir
        )
        
        return stats
    
    def compute_validation_statistics(self, validation_results):
        """Compute validation statistics."""
        stats = {
            'total_samples': len(validation_results),
            'passed_samples': sum(1 for r in validation_results if r['passed']),
            'rejected_samples': sum(1 for r in validation_results if not r['passed']),
            'pass_rate': 0.0,
            'avg_quality_score': 0.0,
            'avg_blur_score': 0.0,
            'avg_artifact_score': 0.0,
            'avg_color_score': 0.0,
            'avg_defect_consistency': 0.0,
            'avg_presence_score': 0.0,
            'class_pass_rates': {}
        }
        
        if len(validation_results) > 0:
            stats['pass_rate'] = stats['passed_samples'] / stats['total_samples']
            stats['avg_quality_score'] = np.mean([r['quality_score'] for r in validation_results])
            stats['avg_blur_score'] = np.mean([r['blur_score'] for r in validation_results])
            stats['avg_artifact_score'] = np.mean([r['artifact_score'] for r in validation_results])
            stats['avg_color_score'] = np.mean([r['color_score'] for r in validation_results])
            stats['avg_defect_consistency'] = np.mean([r['defect_consistency'] for r in validation_results])
            stats['avg_presence_score'] = np.mean([r['presence_score'] for r in validation_results])
            
            # Per-class pass rates (metadata는 0-indexed, 출력은 +1)
            for class_id in [0, 1, 2, 3]:
                class_results = [r for r in validation_results if r['class_id'] == class_id]
                if len(class_results) > 0:
                    passed = sum(1 for r in class_results if r['passed'])
                    stats['class_pass_rates'][class_id] = passed / len(class_results)
        
        return stats
    
    def save_validation_results(self, validation_results, passed_samples,
                                rejected_samples, stats, output_dir):
        """Save validation results to disk."""
        # Save all results
        results_path = output_dir / 'quality_scores.json'
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\nSaved quality scores to: {results_path}")
        
        # Save passed samples list
        passed_path = output_dir / 'passed_samples.txt'
        with open(passed_path, 'w') as f:
            for sample in passed_samples:
                f.write(f"{_get_image_name(sample)}\n")
        print(f"Saved passed samples list to: {passed_path}")
        
        # Save rejected samples list
        rejected_path = output_dir / 'rejected_samples.txt'
        with open(rejected_path, 'w') as f:
            for sample in rejected_samples:
                f.write(f"{_get_image_name(sample)} (score: {sample['quality_score']:.3f})\n")
        print(f"Saved rejected samples list to: {rejected_path}")
        
        # Save statistics
        stats_path = output_dir / 'validation_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to: {stats_path}")
        
        # Save detailed report
        report_path = output_dir / 'quality_report.txt'
        with open(report_path, 'w') as f:
            f.write("Augmented Data Quality Validation Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples: {stats['total_samples']}\n")
            f.write(f"Passed samples: {stats['passed_samples']}\n")
            f.write(f"Rejected samples: {stats['rejected_samples']}\n")
            f.write(f"Pass rate: {stats['pass_rate']*100:.2f}%\n\n")
            
            f.write("Average Scores:\n")
            f.write(f"  Overall quality: {stats['avg_quality_score']:.3f}\n")
            f.write(f"  Blur score: {stats['avg_blur_score']:.3f}\n")
            f.write(f"  Artifact score: {stats['avg_artifact_score']:.3f}\n")
            f.write(f"  Color score: {stats['avg_color_score']:.3f}\n")
            f.write(f"  Defect consistency: {stats['avg_defect_consistency']:.3f}\n")
            f.write(f"  Presence score: {stats['avg_presence_score']:.3f}\n\n")
            
            f.write("Pass Rates by Class:\n")
            for class_id, rate in sorted(stats['class_pass_rates'].items()):
                f.write(f"  Class {class_id + 1}: {rate*100:.2f}%\n")
        
        print(f"Saved quality report to: {report_path}")


# ======================================================================
# 병렬 워커 함수 (ProcessPoolExecutor용 — 모듈 레벨 정의 필수)
# ======================================================================

def _validate_single_sample(args_tuple):
    """
    단일 샘플의 품질을 검증하는 워커 함수.

    ProcessPoolExecutor에서 pickle 직렬화가 가능하도록 모듈 레벨에 정의.
    각 워커 프로세스에서 QualityValidator를 생성하여 사용한다.

    Args:
        args_tuple: (image_path_str, mask_path_str, metadata_dict, min_quality_score)

    Returns:
        성공 시: {**metadata, **validation_result} dict
        실패 시: None
    """
    import cv2 as _cv2

    image_path_str, mask_path_str, metadata, min_quality_score = args_tuple

    if not Path(image_path_str).exists() or not Path(mask_path_str).exists():
        return None

    generated_image = _cv2.imread(image_path_str)
    if generated_image is None:
        return None
    generated_image = _cv2.cvtColor(generated_image, _cv2.COLOR_BGR2RGB)

    defect_mask = _cv2.imread(mask_path_str, _cv2.IMREAD_GRAYSCALE)
    if defect_mask is None:
        return None

    # 워커별 validator 인스턴스 (DefectCharacterizer 포함)
    validator = QualityValidator(min_quality_score=min_quality_score)
    validation_result = validator.validate_sample(
        generated_image, defect_mask, metadata
    )

    return {**metadata, **validation_result}


def main():
    parser = argparse.ArgumentParser(
        description='Validate quality of augmented defect data'
    )
    parser.add_argument(
        '--augmented_dir',
        type=str,
        required=True,
        help='Directory with augmented data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for validation results (default: augmented_dir/validation)'
    )
    parser.add_argument(
        '--min_quality_score',
        type=float,
        default=0.7,
        help='Minimum quality score threshold'
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='병렬 워커 수 (기본 0 = 순차 처리, -1 = CPU 코어 수 자동 감지, '
             'N >= 2 = N개 프로세스 병렬 처리)',
    )
    
    args = parser.parse_args()
    
    # 워커 수 결정
    num_workers = args.workers
    if num_workers < 0:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count - 1)
        print(f"워커 수 자동 설정: {num_workers} (CPU: {cpu_count})")
    
    # Default output dir
    if args.output_dir is None:
        args.output_dir = str(Path(args.augmented_dir) / 'validation')
    
    print("="*80)
    print("Augmented Data Quality Validation")
    print("="*80)
    print(f"Augmented data: {args.augmented_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min quality score: {args.min_quality_score}")
    print("="*80)
    
    # Create validator
    validator = QualityValidator(min_quality_score=args.min_quality_score)
    
    # Validate dataset
    stats = validator.validate_dataset(
        augmented_dir=args.augmented_dir,
        output_dir=args.output_dir,
        num_workers=num_workers,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Passed samples: {stats['passed_samples']}")
    print(f"Rejected samples: {stats['rejected_samples']}")
    print(f"Pass rate: {stats['pass_rate']*100:.2f}%")
    
    print(f"\nAverage scores:")
    print(f"  Overall quality: {stats['avg_quality_score']:.3f}")
    print(f"  Blur: {stats['avg_blur_score']:.3f}")
    print(f"  Artifacts: {stats['avg_artifact_score']:.3f}")
    print(f"  Color: {stats['avg_color_score']:.3f}")
    print(f"  Defect consistency: {stats['avg_defect_consistency']:.3f}")
    print(f"  Presence: {stats['avg_presence_score']:.3f}")
    
    print(f"\nPass rates by class:")
    for class_id, rate in sorted(stats['class_pass_rates'].items()):
        print(f"  Class {class_id + 1}: {rate*100:.2f}%")
    
    print(f"\nValidation results saved to: {args.output_dir}")
    print("="*80)
    
    # Recommendation
    if stats['pass_rate'] < 0.7:
        print("\n⚠️  Warning: Pass rate below 70%. Consider:")
        print("  - Adjusting min_quality_score threshold")
        print("  - Improving ControlNet model quality")
        print("  - Reviewing rejected samples manually")


if __name__ == '__main__':
    main()
