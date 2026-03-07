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
import os
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.controlnet_packager import ControlNetDatasetPackager
from src.preprocessing.hint_generator import HintImageGenerator
from src.preprocessing.prompt_generator import PromptGenerator
from src.utils.dataset_validator import DatasetValidator

from typing import Dict, Optional


def parse_class_edge_overrides(raw: Optional[str]) -> Optional[Dict[int, Dict[str, float]]]:
    """
    CLI 문자열을 class_margin_overrides dict로 변환.

    형식: "CLASS:X,Y;CLASS:X,Y"
      - CLASS: 1-based class_id (int)
      - X: x-margin 비율 (float)
      - Y: y-margin 비율 (float)

    예: "4:0.05,0.0"         → {4: {'x': 0.05, 'y': 0.0}}
        "3:0.08,0.03;4:0.05,0.0" → {3: {'x': 0.08, 'y': 0.03}, 4: {'x': 0.05, 'y': 0.0}}

    Returns:
        None if raw is None or empty, otherwise parsed dict.
    Raises:
        ValueError: 형식 오류 시.
    """
    if not raw:
        return None

    overrides: Dict[int, Dict[str, float]] = {}
    for entry in raw.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError(
                f"잘못된 형식 '{entry}'. 올바른 형식: CLASS:X,Y (예: 4:0.05,0.0)"
            )
        cls_str, margins_str = entry.split(':', 1)
        try:
            cls_id = int(cls_str.strip())
        except ValueError:
            raise ValueError(f"CLASS는 정수여야 합니다: '{cls_str}'")
        # class_id 범위 검증 (Severstal 데이터셋: 1~4)
        if cls_id < 1 or cls_id > 4:
            raise ValueError(
                f"CLASS는 1~4 범위여야 합니다: {cls_id}. "
                f"Severstal 데이터셋의 ClassId는 1-based (1,2,3,4)입니다."
            )
        # 중복 class_id 검출
        if cls_id in overrides:
            raise ValueError(
                f"Class {cls_id}가 중복 지정되었습니다. "
                f"이전 값: {overrides[cls_id]}, 새 값을 파싱 중: '{margins_str}'"
            )
        parts = margins_str.strip().split(',')
        if len(parts) != 2:
            raise ValueError(
                f"Class {cls_id}: X,Y 두 값이 필요합니다 ('{margins_str}')"
            )
        try:
            mx = float(parts[0].strip())
            my = float(parts[1].strip())
        except ValueError:
            raise ValueError(
                f"Class {cls_id}: X/Y는 실수여야 합니다 ('{margins_str}')"
            )
        if not (0.0 <= mx <= 1.0) or not (0.0 <= my <= 1.0):
            raise ValueError(
                f"Class {cls_id}: X/Y는 0.0~1.0 범위여야 합니다 (x={mx}, y={my})"
            )
        overrides[cls_id] = {'x': mx, 'y': my}

    return overrides if overrides else None


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
        help='Maximum number of samples to process (균등 분배 모드, v3~v5.1 호환). '
             '--per_class_cap이 지정되면 무시됨.'
    )
    parser.add_argument(
        '--per_class_cap',
        type=int,
        default=None,
        help='v5.2 class-aware capping: 풍부한 클래스의 최대 샘플 수. '
             '희소 클래스(--rare_class_threshold 이하)는 전수 포함. '
             '예: --per_class_cap 1200'
    )
    parser.add_argument(
        '--rare_class_threshold',
        type=int,
        default=200,
        help='이 수 이하인 클래스를 희소로 간주하여 전수 포함 (기본 200). '
             '--per_class_cap과 함께 사용.'
    )
    parser.add_argument(
        '--class_edge_override',
        type=str,
        default=None,
        help='B2 클래스별 엣지 마진 오버라이드. 형식: "CLASS:X,Y;CLASS:X,Y". '
             'CLASS는 1-based class_id, X/Y는 마진 비율. '
             '예: "4:0.05,0.0" → Class 4는 좌우 5%%, 상하 0%% 마진. '
             '"3:0.08,0.03;4:0.05,0.0" → Class 3/4 각각 개별 설정. '
             'Class 4 결함은 이미지 전체 높이(256px)를 차지하므로 Y=0 권장.'
    )
    parser.add_argument(
        '--validation_samples',
        type=int,
        default=16,
        help='Number of samples for visual validation'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='병렬 워커 수 (0=순차 처리, -1=자동 감지, N=N개 워커)'
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
        sys.exit(1)
    
    if not train_csv.exists():
        print(f"Error: Training CSV not found: {train_csv}")
        sys.exit(1)
    
    # B2: 클래스별 엣지 마진 오버라이드 파싱
    class_margin_overrides = None
    if args.class_edge_override:
        try:
            class_margin_overrides = parse_class_edge_overrides(args.class_edge_override)
        except ValueError as e:
            print(f"Error: --class_edge_override 파싱 실패: {e}")
            sys.exit(1)
    
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
    if args.per_class_cap:
        print(f"Per-class cap: {args.per_class_cap} (rare threshold: {args.rare_class_threshold})")
    if class_margin_overrides:
        print(f"Edge margin overrides: {class_margin_overrides}")
    
    # 워커 수 결정
    num_workers = args.workers
    if num_workers < 0:
        cpu_count = os.cpu_count() or 2
        num_workers = max(1, cpu_count - 1)
        print(f"Workers: {num_workers} (auto-detected, CPU: {cpu_count})")
    else:
        print(f"Workers: {num_workers or 'sequential'}")
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
            print("\n[WARN] Validation warnings detected. Review the reports before proceeding.")
            print(f"   Reports saved to: {output_dir / 'validation'}")
            
            response = input("\nDo you want to continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborting. Please address the issues and try again.")
                sys.exit(1)
        else:
            print("\n[PASS] Dataset validation passed!")
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
        max_samples=args.max_samples,
        per_class_cap=args.per_class_cap,
        rare_class_threshold_count=args.rare_class_threshold,
        class_margin_overrides=class_margin_overrides,
        num_workers=num_workers,
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("[DONE] ControlNet Training Data Preparation Complete!")
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
