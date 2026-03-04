#!/usr/bin/env python3
"""
CASDA Benchmark Experiment Runner

Orchestrates the full benchmark experiment: 3 models x 4 dataset groups = 12 training runs.

Models:
  - YOLO-MFD (Multi-scale Edge Feature Enhancement) — ultralytics native training
  - EB-YOLOv8 (BiFPN-based Enhanced Backbone) — ultralytics native training
  - DeepLabV3+ (Standard Segmentation Baseline) — BenchmarkTrainer

Dataset Groups:
  - baseline_raw  (alias: baseline, raw)  : Severstal original only
  - baseline_trad (alias: trad)           : Original + traditional augmentations
  - casda_full    (alias: full)           : Original + all CASDA synthetic images
  - casda_pruning (alias: pruning)        : Original + top CASDA images by suitability
  - all                                   : Run all groups

CASDA Inject/Clean Strategy:
  For CASDA groups (casda_full, casda_pruning), instead of creating separate YOLO
  directories, the script:
  1. Injects CASDA images/labels into baseline_raw/images/train/ (prefix: casda_*)
  2. Trains all models on the augmented baseline_raw
  3. Cleans CASDA files (removes casda_* prefix files) to restore baseline_raw
  This avoids duplicating baseline_raw (saves disk space and I/O time).
  Detection models use the injected baseline_raw; segmentation models use
  ConcatDataset internally (no inject needed, uses --casda-dir directly).

Usage:
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --models yolo_mfd --groups baseline
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --groups full pruning
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --groups all
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --list-groups
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --fid-only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --resume --output-dir outputs/benchmark_results/20260223_143000
"""

import os
import sys
import argparse
import logging
import shutil
import time
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeplabv3plus import DeepLabV3Plus
from src.training.dataset import (
    create_data_loaders,
    get_image_ids_with_defects,
    split_dataset,
)
from src.training.trainer import BenchmarkTrainer
from src.training.ultralytics_trainer import UltralyticsTrainer
from src.training.metrics import (
    DetectionEvaluator,
    SegmentationEvaluator,
    FIDCalculator,
    BenchmarkReporter,
)


# ============================================================================
# Detection Model Set (uses ultralytics)
# ============================================================================
ULTRALYTICS_MODELS = {"yolo_mfd", "eb_yolov8"}


# ============================================================================
# Dataset Group Aliases & Validation
# ============================================================================
# Short aliases for convenience. Keys are alias names, values are the
# canonical group key that appears in the YAML config.
GROUP_ALIASES = {
    # Shorthand aliases
    "baseline":  "baseline_raw",
    "raw":       "baseline_raw",
    "trad":      "baseline_trad",
    "traditional": "baseline_trad",
    "full":      "casda_full",
    "pruning":   "casda_pruning",
    "composed":  "casda_composed",
    "composed_pruning": "casda_composed_pruning",
    # Special
    "all":       "__ALL__",
}


def resolve_groups(
    requested: list,
    available: list,
) -> list:
    """
    Resolve CLI group names (with alias support) to canonical group keys.

    Args:
        requested: List of group names/aliases from CLI --groups
        available: List of canonical group keys from YAML config

    Returns:
        List of resolved canonical group keys (deduplicated, order-preserved)

    Raises:
        SystemExit: If any requested group is not valid
    """
    if requested is None:
        return available

    resolved = []
    seen = set()

    for g in requested:
        g_lower = g.lower().strip()

        # Check alias first
        if g_lower in GROUP_ALIASES:
            target = GROUP_ALIASES[g_lower]
            if target == "__ALL__":
                for k in available:
                    if k not in seen:
                        resolved.append(k)
                        seen.add(k)
                continue
            canonical = target
        elif g_lower in available:
            canonical = g_lower
        elif g in available:
            canonical = g
        else:
            # Not found — print helpful error
            alias_list = ", ".join(
                f"{alias} → {target}" for alias, target in sorted(GROUP_ALIASES.items())
                if target != "__ALL__"
            )
            print(f"\n[ERROR] Unknown dataset group: '{g}'")
            print(f"\nAvailable groups (from config):")
            for k in available:
                print(f"  - {k}")
            print(f"\nSupported aliases:")
            print(f"  {alias_list}")
            print(f"\nSpecial:")
            print(f"  all → run all {len(available)} groups")
            print(f"\nExamples:")
            print(f"  --groups baseline casda_full")
            print(f"  --groups full pruning")
            print(f"  --groups all")
            sys.exit(1)

        if canonical not in available:
            print(f"\n[ERROR] Resolved group '{canonical}' (from alias '{g}') "
                  f"not found in config.")
            print(f"Available groups: {available}")
            sys.exit(1)

        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)

    return resolved


# ============================================================================
# Split ID Resolution (shared by both detection and segmentation paths)
# ============================================================================

def get_split_ids(config: dict) -> tuple:
    """
    Get train/val/test image ID lists from config.
    
    Supports two modes:
      1. Pre-generated split CSV (config['dataset']['split_csv'])
      2. Dynamic split from annotation CSV
    
    Returns:
        (train_ids, val_ids, test_ids)
    """
    ds_config = config['dataset']
    raw_csv = ds_config['annotation_csv']
    annotation_csv = raw_csv if os.path.isabs(raw_csv) else str(PROJECT_ROOT / raw_csv)

    split_csv = ds_config.get('split_csv', None)

    if split_csv is not None and os.path.exists(split_csv):
        split_df = pd.read_csv(split_csv, comment='#')
        train_ids = split_df[split_df['Split'] == 'train']['ImageId'].tolist()
        val_ids = split_df[split_df['Split'] == 'val']['ImageId'].tolist()
        test_ids = split_df[split_df['Split'] == 'test']['ImageId'].tolist()
        logging.info(f"Loaded split from CSV: train={len(train_ids)}, "
                     f"val={len(val_ids)}, test={len(test_ids)}")
    else:
        image_ids, image_classes = get_image_ids_with_defects(annotation_csv)
        train_ids, val_ids, test_ids = split_dataset(
            image_ids, image_classes,
            train_ratio=ds_config['split']['train_ratio'],
            val_ratio=ds_config['split']['val_ratio'],
            test_ratio=ds_config['split']['test_ratio'],
            seed=ds_config['split']['seed'],
        )
        if split_csv is not None:
            logging.warning(f"Split CSV not found: {split_csv} — using dynamic split")

    return train_ids, val_ids, test_ids


# ============================================================================
# Segmentation Model Factory
# ============================================================================

def create_segmentation_model(model_key: str, model_config: dict, num_classes: int = 4):
    """Create a segmentation model."""
    if model_key == "deeplabv3plus":
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone=model_config.get('backbone', 'resnet101'),
            pretrained=model_config.get('pretrained', True),
            output_stride=model_config.get('output_stride', 16),
        )
    else:
        raise ValueError(f"Unknown segmentation model: {model_key}")


# ============================================================================
# CASDA Inject / Clean  — baseline_raw 에 직접 주입/삭제
# ============================================================================
CASDA_PREFIX = "casda_"


def inject_casda_to_baseline(
    baseline_dir: str,
    casda_dir: str,
    prefix: str = CASDA_PREFIX,
    max_samples: Optional[int] = None,
    suitability_threshold: Optional[float] = None,
) -> int:
    """
    CASDA 합성 이미지/라벨을 baseline_raw YOLO 데이터셋의 train/에 주입한다.

    이미지: symlink (실패 시 copy) → {baseline_dir}/images/train/{prefix}NNNNN_{name}
    라벨:  metadata.json의 bboxes를 YOLO .txt로 직접 작성

    metadata.json에 bbox_format="yolo" + bboxes 가 있으면 cv2 I/O 제로.
    없으면 mask_path → contour → bbox 변환 (레거시 호환).

    Args:
        baseline_dir: baseline_raw YOLO 데이터셋 경로 (images/train/, labels/train/ 포함)
        casda_dir: CASDA 패키징 디렉토리 (images/, masks/, metadata.json)
        prefix: 주입 파일 접두사 (clean 시 이 prefix로 삭제)
        max_samples: 주입할 최대 합성 이미지 수 (None이면 전체)
        suitability_threshold: 최소 suitability 점수 (None이면 필터링 없음)

    Returns:
        주입된 이미지 수
    """
    from src.training.dataset_yolo import _add_casda_to_training

    baseline_path = Path(baseline_dir)
    images_train = baseline_path / "images" / "train"
    labels_train = baseline_path / "labels" / "train"

    if not images_train.exists() or not labels_train.exists():
        logging.error(f"baseline train dirs not found: {images_train}, {labels_train}")
        return 0

    # max_samples 또는 suitability_threshold가 있으면 pruning 모드로 필터링
    casda_mode = "full"
    casda_config = {}
    if max_samples is not None or suitability_threshold is not None:
        casda_mode = "pruning"
        casda_config = {
            'pruning_top_k': max_samples or 99999,
            'suitability_threshold': suitability_threshold or 0.0,
        }

    count = _add_casda_to_training(
        casda_dir=casda_dir,
        casda_mode=casda_mode,
        casda_config=casda_config,
        images_train_dir=images_train,
        labels_train_dir=labels_train,
        num_classes=4,
    )

    logging.info(f"  Injected {count} CASDA images into {images_train}")
    return count


def clean_casda_from_baseline(
    baseline_dir: str,
    prefix: str = CASDA_PREFIX,
) -> int:
    """
    baseline_raw YOLO 데이터셋에서 CASDA prefix 파일을 모두 삭제한다.

    images/train/casda_* 와 labels/train/casda_* 를 삭제.
    baseline 원본 파일은 prefix가 다르므로 절대 삭제되지 않음.

    Args:
        baseline_dir: baseline_raw YOLO 데이터셋 경로
        prefix: 삭제할 파일 접두사

    Returns:
        삭제된 파일 수 (이미지 + 라벨 합계)
    """
    baseline_path = Path(baseline_dir)
    removed = 0

    for subdir in ["images/train", "labels/train"]:
        target_dir = baseline_path / subdir
        if not target_dir.exists():
            continue
        for f in target_dir.iterdir():
            if f.name.startswith(prefix):
                f.unlink(missing_ok=True)
                removed += 1

    logging.info(f"  Cleaned {removed} CASDA files from {baseline_path}")
    return removed


# ============================================================================
# Single Experiment Run
# ============================================================================

def run_single_experiment(
    model_key: str,
    dataset_group: str,
    config: dict,
    experiment_dir: Path,
    device: str = 'cuda',
    resume: bool = False,
    yolo_dir: Optional[str] = None,
    output_group_key: Optional[str] = None,
) -> dict:
    """
    Run a single training experiment.
    
    Routes to UltralyticsTrainer for detection models (YOLO-MFD, EB-YOLOv8)
    and BenchmarkTrainer for segmentation models (DeepLabV3+).
    
    Args:
        output_group_key: If set, used for output directory naming and metadata
            instead of dataset_group. This allows CASDA inject mode to train on
            baseline_raw (with injected files) but label the output as casda_full
            or casda_pruning.
    """
    # output_group_key: actual group name for dirs/metadata (e.g. "casda_full")
    # dataset_group: effective group used for training (e.g. "baseline_raw" after inject)
    actual_group_key = output_group_key or dataset_group
    
    model_config = config['models'][model_key]
    model_name = model_config['name']
    model_type = model_config['type']  # "detection" or "segmentation"
    group_name = config['dataset_groups'][actual_group_key]['name']
    group_config = config['dataset_groups'][actual_group_key]
    num_classes = config['dataset']['num_classes']

    # Create output directory — use actual_group_key for unique naming
    run_dir = experiment_dir / f"{model_key}_{actual_group_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume: check if this run is already completed ----
    meta_path = run_dir / "experiment_meta.json"

    if resume and meta_path.exists():
        # Check for completion marker
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('test_metrics'):
            logging.info(f"\n{'#'*70}")
            logging.info(f"# SKIP (completed): {model_name} + {group_name}")
            logging.info(f"# Loading existing results from: {meta_path}")
            logging.info(f"{'#'*70}")
            return meta['test_metrics']

    logging.info(f"\n{'#'*70}")
    logging.info(f"# Experiment: {model_name} + {group_name}")
    logging.info(f"# Type: {model_type}")
    if output_group_key and output_group_key != dataset_group:
        logging.info(f"# Training on: {dataset_group} (inject mode)")
    logging.info(f"{'#'*70}")

    # Get split IDs
    train_ids, val_ids, test_ids = get_split_ids(config)

    # Save split info once per experiment directory
    split_path = experiment_dir / "dataset_split.json"
    if not split_path.exists():
        split_info = {
            'num_train': len(train_ids),
            'num_val': len(val_ids),
            'num_test': len(test_ids),
            'split_config': config['dataset']['split'],
        }
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)

    # ====================================================================
    # Detection models: use UltralyticsTrainer (native ultralytics .train())
    # ====================================================================
    if model_type == "detection" and model_key in ULTRALYTICS_MODELS:
        trainer = UltralyticsTrainer(
            model_key=model_key,
            model_config=model_config,
            dataset_config=config['dataset'],
            group_config=group_config,
            dataset_group=dataset_group,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            output_dir=str(run_dir),
            device=device,
            resume=resume,
            yolo_dir=yolo_dir,
        )

        test_metrics = trainer.train()
        history = getattr(trainer, 'history', {})

    # ====================================================================
    # Segmentation models: use BenchmarkTrainer (existing training loop)
    # ====================================================================
    else:
        model = create_segmentation_model(model_key, model_config, num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create data loaders (segmentation uses existing dataset.py pipeline)
        input_size = tuple(model_config.get('input_size', [256, 512]))
        batch_size = model_config['training'].get('batch_size', 8)
        num_workers = config['experiment'].get('num_workers', 4)

        train_loader, val_loader, test_loader, split_info_ds = create_data_loaders(
            config=config,
            dataset_group=dataset_group,
            model_type=model_type,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        logging.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                     f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        # Resume checkpoint for segmentation
        resume_checkpoint = None
        if resume:
            checkpoint_dir = run_dir / "checkpoints"
            latest_path = checkpoint_dir / f"{model_key}_{actual_group_key}_latest.pth"
            if latest_path.exists():
                resume_checkpoint = str(latest_path)
                logging.info(f"Resuming from: {resume_checkpoint}")

        training_config = {**model_config['training'], 'num_classes': num_classes}
        seg_trainer = BenchmarkTrainer(
            model=model,
            model_name=f"{model_key}_{actual_group_key}",
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=training_config,
            output_dir=str(run_dir),
            device=device,
            resume_from=resume_checkpoint,
        )

        test_metrics = seg_trainer.train()
        history = seg_trainer.history

    # Save experiment metadata
    meta = {
        'model': model_name,
        'model_key': model_key,
        'model_type': model_type,
        'dataset_group': group_name,
        'dataset_group_key': actual_group_key,
        'training_dataset_group': dataset_group,
        'inject_mode': output_group_key is not None and output_group_key != dataset_group,
        'test_metrics': test_metrics,
        'best_epoch': history.get('best_epoch', 0),
        'best_metric': history.get('best_metric', 0.0),
        'total_epochs_trained': len(history.get('train_loss', [])) or len(history.get('val_metric', [])),
        'early_stopped': history.get('early_stopped', False),
        'stopped_epoch': history.get('stopped_epoch', 0),
        'max_epochs': history.get('max_epochs', 0),
        'total_time_seconds': history.get('total_time_seconds', 0.0),
        'use_amp': history.get('use_amp', False),
        'training_pipeline': 'ultralytics' if model_key in ULTRALYTICS_MODELS else 'benchmark_trainer',
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "experiment_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return test_metrics


# ============================================================================
# FID Evaluation
# ============================================================================

def run_fid_evaluation(config: dict, experiment_dir: Path, device: str = 'cuda') -> dict:
    """Compute FID scores between real and CASDA-generated images."""
    logging.info(f"\n{'#'*70}")
    logging.info(f"# FID Score Evaluation")
    logging.info(f"{'#'*70}")

    fid_calc = FIDCalculator(device=device)
    ds_config = config['dataset']
    casda_config = ds_config.get('casda', {})

    raw_image_dir = ds_config['image_dir']
    image_dir = Path(raw_image_dir) if os.path.isabs(raw_image_dir) else PROJECT_ROOT / raw_image_dir
    real_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    real_images = [str(p) for p in real_images]

    if not real_images:
        logging.warning("No real images found for FID computation")
        return {'fid_overall': float('inf')}

    raw_casda_dir = casda_config.get('full_dir', 'data/augmented/casda_full')
    casda_full_dir = Path(raw_casda_dir) if os.path.isabs(raw_casda_dir) else PROJECT_ROOT / raw_casda_dir
    casda_images = []
    if casda_full_dir.exists():
        for subdir in [casda_full_dir, casda_full_dir / "images"]:
            if subdir.exists():
                casda_images.extend(sorted(subdir.glob("*.png")))
                casda_images.extend(sorted(subdir.glob("*.jpg")))
    casda_images = [str(p) for p in casda_images]

    results = {}
    if casda_images:
        logging.info(f"Computing FID: {len(real_images)} real vs {len(casda_images)} synthetic")
        overall_fid = fid_calc.compute_fid(
            real_images[:1000],
            casda_images[:1000],
            batch_size=config.get('evaluation', {}).get('fid', {}).get('batch_size', 64),
        )
        results['fid_overall'] = overall_fid
        logging.info(f"FID Score (overall): {overall_fid:.2f}")
    else:
        logging.warning("No CASDA images found for FID computation")
        results['fid_overall'] = float('inf')

    fid_path = experiment_dir / "fid_results.json"
    with open(fid_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"FID results saved to: {fid_path}")

    return results


# ============================================================================
# Hypothesis Testing
# ============================================================================

def run_hypothesis_tests(reporter: BenchmarkReporter, config: dict) -> dict:
    """Evaluate the 5 hypotheses defined in experiment.md."""
    logging.info(f"\n{'#'*70}")
    logging.info(f"# Hypothesis Testing")
    logging.info(f"{'#'*70}")

    hypotheses_results = {}

    for h_config in config.get('reporting', {}).get('hypothesis_tests', []):
        h_name = h_config['name']
        h_desc = h_config['description']
        metric = h_config.get('metric', 'mAP@0.5')
        compare = h_config.get('compare', [])

        logging.info(f"\n{h_name}: {h_desc}")

        if h_name == "H5":
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'see FID results'}
            continue

        if len(compare) < 2:
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'insufficient data'}
            continue

        group_a, group_b = compare[0], compare[1]

        a_values = []
        b_values = []

        for result in reporter.results:
            ds = result['dataset']
            metrics = result['metrics']

            if h_name == "H4":
                focus_classes = h_config.get('focus_classes', [3, 4])
                class_ap = metrics.get('class_ap', {})
                val = np.mean([class_ap.get(f"Class{c}", 0.0) for c in focus_classes])
            else:
                val = metrics.get(metric, metrics.get('mAP@0.5', 0.0))

            if ds == config['dataset_groups'].get(group_a, {}).get('name', group_a):
                a_values.append(val)
            elif ds == config['dataset_groups'].get(group_b, {}).get('name', group_b):
                b_values.append(val)

        if a_values and b_values:
            mean_a = np.mean(a_values)
            mean_b = np.mean(b_values)
            improvement = mean_a - mean_b

            hypotheses_results[h_name] = {
                'description': h_desc,
                f'{group_a}_mean': float(mean_a),
                f'{group_b}_mean': float(mean_b),
                'improvement': float(improvement),
                'supported': improvement > 0,
            }
            status = "SUPPORTED" if improvement > 0 else "NOT SUPPORTED"
            logging.info(f"  {group_a}: {mean_a:.4f} vs {group_b}: {mean_b:.4f} "
                         f"-> {status} (delta={improvement:+.4f})")
        else:
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'no data'}

    return hypotheses_results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CASDA Benchmark Experiment Runner (3 models x 4 datasets = 12 runs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Groups:
  baseline_raw     Baseline (Raw)      — Severstal original only, no augmentation
  baseline_trad    Baseline (Trad)     — Original + traditional geometric augmentations
  casda_full       CASDA-Full          — Original + all ~2,901 CASDA synthetic images
  casda_pruning    CASDA-Pruning       — Original + top CASDA images by suitability

Short Aliases:
  baseline → baseline_raw       raw → baseline_raw
  trad → baseline_trad          traditional → baseline_trad
  full → casda_full             pruning → casda_pruning
  all → run all groups

Examples:
  # Run full benchmark (all 12 experiments)
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml

  # Run baseline only (alias for baseline_raw)
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml \\
      --models yolo_mfd --groups baseline

  # Run CASDA experiments with short aliases
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml \\
      --groups full pruning

  # Run all groups explicitly
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml \\
      --groups all

  # Colab: specify data paths and output directory explicitly
  python scripts/run_benchmark.py \\
      --config /content/severstal-steel-defect-detection/configs/benchmark_experiment.yaml \\
      --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --casda-dir /content/drive/MyDrive/data/Severstal/data/augmented_v4_dataset \\
      --split-csv /content/drive/MyDrive/data/Severstal/casda/splits/split_70_15_15_seed42.csv \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/benchmark_results \\
      --models yolo_mfd --groups baseline --epochs 10

  # Fast re-run: use pre-converted YOLO dataset (skip CSV->YOLO conversion)
  python scripts/run_benchmark.py \\
      --config /content/severstal-steel-defect-detection/configs/benchmark_experiment.yaml \\
      --yolo-dir /content/yolo_datasets \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/benchmark_results \\
      --models yolo_mfd --groups baseline --epochs 10

  # Resume: add CASDA runs to existing experiment (skips completed runs)
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml \\
      --resume --output-dir outputs/benchmark_results/20260223_143000

  # List available groups and models from config
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --list-groups
        """,
    )
    parser.add_argument('--config', type=str, default='configs/benchmark_experiment.yaml',
                        help='Path to experiment config YAML')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to image directory (overrides config dataset.image_dir)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to annotation CSV (overrides config dataset.annotation_csv)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of models to run (e.g., yolo_mfd eb_yolov8)')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Dataset groups to run. Accepts config keys '
                             '(baseline_raw, baseline_trad, casda_full, casda_pruning) '
                             'or short aliases (baseline, trad, full, pruning, all). '
                             'Examples: --groups baseline full | --groups all')
    parser.add_argument('--list-groups', action='store_true',
                        help='List available dataset groups and aliases, then exit')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--fid-only', action='store_true',
                        help='Only compute FID scores, skip model training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from existing experiment directory (specified by --output-dir). '
                             'Skips already-completed runs (those with experiment_meta.json), '
                             'resumes interrupted runs from last.pt/latest.pth.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs for all models (for quick testing)')
    parser.add_argument('--casda-dir', type=str, default=None,
                        help='Parent dir containing casda_full/ and casda_pruning/ '
                             '(overrides config dataset.casda paths)')
    parser.add_argument('--split-csv', type=str, default=None,
                        help='사전 생성된 분할 CSV 파일 경로 '
                             '(scripts/create_dataset_split.py로 생성). '
                             '지정 시 동적 분할 대신 이 파일의 분할을 사용')
    parser.add_argument('--yolo-dir', type=str, default=None,
                        help='사전 변환된 YOLO 포맷 데이터셋 디렉토리. '
                             '그룹별 하위 디렉토리(예: baseline_raw/)에 '
                             'images/, labels/, dataset.yaml이 있으면 변환을 건너뜀. '
                             '없으면 이 디렉토리에 변환 결과를 저장하여 다음 실행 시 재사용')
    parser.add_argument('--casda-ratio', type=float, nargs='+', default=None,
                        help='CASDA 합성 데이터 주입 비율 (0.0~1.0). '
                             '지정 시 CASDA 데이터에서 비율에 맞는 수량만 선택. '
                             '복수 비율 지정 가능: --casda-ratio 0.1 0.2 0.3. '
                             '원본 train 수 대비 비율로 max_samples 자동 계산. '
                             '예: 0.1 → 원본 4666장의 10% ≈ 518장 합성 추가')
    parser.add_argument('--casda-ratio-source', type=str, default='full',
                        choices=['full', 'composed'],
                        help='--casda-ratio 그룹의 데이터 소스 선택. '
                             'full: casda_full 디렉토리 (기본), '
                             'composed: casda_composed 디렉토리. '
                             '예: --casda-ratio 0.3 --casda-ratio-source composed')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ---- --list-groups: show available groups and exit ----
    if args.list_groups:
        available = list(config.get('dataset_groups', {}).keys())
        print("\nAvailable Dataset Groups:")
        print("-" * 70)
        for key in available:
            grp = config['dataset_groups'][key]
            name = grp.get('name', key)
            desc = grp.get('description', '')
            casda = grp.get('casda_data')
            tag = ""
            if casda == "full":
                tag = " [CASDA full]"
            elif casda == "pruning":
                tag = " [CASDA pruning]"
            elif casda == "composed":
                pruning_cfg = grp.get('casda_pruning', {})
                if pruning_cfg.get('enabled', False):
                    tag = " [CASDA composed+pruning]"
                else:
                    tag = " [CASDA composed]"
            elif grp.get('augmentation') == 'traditional':
                tag = " [traditional aug]"
            else:
                tag = " [no augmentation]"
            print(f"  {key:<20s} {name:<22s} {tag}")
            if desc:
                print(f"  {'':<20s} {desc}")
        print(f"\nShort Aliases:")
        for alias, target in sorted(GROUP_ALIASES.items()):
            if target == "__ALL__":
                print(f"  {alias:<20s} → (all {len(available)} groups)")
            else:
                print(f"  {alias:<20s} → {target}")
        print(f"\nAvailable Models:")
        for mk, mv in config.get('models', {}).items():
            pipeline = "ultralytics" if mk in ULTRALYTICS_MODELS else "BenchmarkTrainer"
            print(f"  {mk:<20s} {mv.get('name', mk):<22s} [{pipeline}]")
        print()
        sys.exit(0)

    # Override data paths if specified via CLI
    if args.data_dir:
        config['dataset']['image_dir'] = args.data_dir
        print(f"[INFO] image_dir overridden to: {args.data_dir}")
    if args.csv:
        config['dataset']['annotation_csv'] = args.csv
        print(f"[INFO] annotation_csv overridden to: {args.csv}")
    if args.casda_dir:
        casda_base = args.casda_dir
        if 'casda' not in config['dataset']:
            config['dataset']['casda'] = {}
        config['dataset']['casda']['full_dir'] = os.path.join(casda_base, 'casda_full')
        config['dataset']['casda']['pruning_dir'] = os.path.join(casda_base, 'casda_pruning')
        config['dataset']['casda']['composed_dir'] = os.path.join(casda_base, 'casda_composed')
        print(f"[INFO] casda paths overridden: {casda_base}/casda_full, {casda_base}/casda_pruning, {casda_base}/casda_composed")

    # Override split CSV if specified
    if args.split_csv:
        split_csv_path = os.path.abspath(args.split_csv)
        config['dataset']['split_csv'] = split_csv_path
        print(f"[INFO] split_csv overridden to: {split_csv_path}")

    # Override epochs if specified (for quick testing)
    if args.epochs is not None:
        for model_key in config.get('models', {}):
            config['models'][model_key]['training']['epochs'] = args.epochs
        print(f"[INFO] Epochs overridden to {args.epochs} for all models")

    # Setup
    seed = args.seed or config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device or config['experiment'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Determine experiment directory
    output_dir = Path(args.output_dir or config['experiment'].get('output_dir', 'outputs/benchmark_results'))

    if args.resume:
        experiment_dir = output_dir
        if not experiment_dir.exists():
            print(f"Error: Resume directory not found: {experiment_dir}")
            sys.exit(1)
        print(f"[INFO] Resuming experiment from: {experiment_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = output_dir / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = experiment_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"CASDA Benchmark Experiment")
    logging.info(f"Config: {config_path}")
    logging.info(f"Output: {experiment_dir}")
    logging.info(f"Device: {device}")
    logging.info(f"Seed: {seed}")

    # Save config copy
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Determine models and groups to run
    model_keys = args.models or list(config['models'].keys())
    available_groups = list(config['dataset_groups'].keys())
    group_keys = resolve_groups(args.groups, available_groups)

    # ====== --casda-ratio: 동적 ratio 그룹 생성 ======
    # --casda-ratio 0.1 0.2 0.3 → casda_ratio_10, casda_ratio_20, casda_ratio_30 그룹 자동 생성
    # --casda-ratio-source full|composed → 데이터 소스 선택 (기본: full)
    # --groups 미지정 + --casda-ratio 지정 시: ratio 그룹만 실행
    # --groups 지정 + --casda-ratio 지정 시: 기존 그룹 + ratio 그룹 병행
    casda_ratio_map = {}  # group_key → (ratio, max_samples)
    if args.casda_ratio:
        # --groups가 없으면 ratio 그룹만 실행하도록 기존 그룹 비우기
        if args.groups is None:
            group_keys = []

        # ratio 소스 결정
        ratio_source = getattr(args, 'casda_ratio_source', 'full')
        if ratio_source == 'composed':
            ratio_casda_data = 'composed'
            ratio_casda_subdir = 'casda_composed'
            ratio_source_label = 'casda_composed'
        else:
            ratio_casda_data = 'full'
            ratio_casda_subdir = 'casda_full'
            ratio_source_label = 'casda_full'

        # 원본 train 수 계산 (split IDs로부터)
        train_ids_for_ratio, _, _ = get_split_ids(config)
        num_train_original = len(train_ids_for_ratio)
        logging.info(f"Original training set size: {num_train_original}")
        logging.info(f"CASDA ratio source: {ratio_source_label}")

        for ratio in args.casda_ratio:
            if not (0.0 < ratio < 1.0):
                logging.error(f"Invalid casda-ratio: {ratio} (must be 0.0 < ratio < 1.0)")
                continue

            # ratio = synthetic / (original + synthetic)
            # synthetic = original * ratio / (1 - ratio)
            max_samples = int(num_train_original * ratio / (1.0 - ratio))
            ratio_pct = int(ratio * 100)
            ratio_key = f"casda_ratio_{ratio_pct}"

            # 동적 dataset group 등록
            config['dataset_groups'][ratio_key] = {
                'name': f"CASDA-Ratio-{ratio_pct}%",
                'description': f"Original + {max_samples} CASDA images ({ratio_pct}% synthetic ratio, source={ratio_source_label})",
                'use_original': True,
                'augmentation': 'none',
                'casda_data': ratio_casda_data,
                '_casda_max_samples': max_samples,
                '_casda_ratio': ratio,
            }
            casda_ratio_map[ratio_key] = (ratio, max_samples)

            if ratio_key not in group_keys:
                group_keys.append(ratio_key)

            logging.info(f"  Created ratio group: {ratio_key} "
                         f"(ratio={ratio:.0%}, max_samples={max_samples}, source={ratio_source_label})")

    logging.info(f"Models: {model_keys}")
    logging.info(f"Dataset groups: {group_keys}")
    if args.groups:
        logging.info(f"  (resolved from CLI: {args.groups})")
    if args.casda_ratio:
        logging.info(f"  (casda-ratio groups added: {list(casda_ratio_map.keys())})")
    logging.info(f"Total experiments: {len(model_keys) * len(group_keys)}")

    # Log training pipeline info
    for mk in model_keys:
        pipeline = "ultralytics" if mk in ULTRALYTICS_MODELS else "BenchmarkTrainer"
        logging.info(f"  {mk}: {pipeline}")

    # Initialize reporter
    reporter = BenchmarkReporter(str(experiment_dir))

    # ====== FID Evaluation ======
    casda_groups_for_fid = {'casda_full', 'casda_pruning', 'casda_composed', 'casda_composed_pruning'}
    casda_groups_for_fid.update(casda_ratio_map.keys())
    has_casda = any(g in casda_groups_for_fid for g in group_keys)

    if args.fid_only or (has_casda and config.get('evaluation', {}).get('fid', {}).get('compute', True)):
        fid_results = run_fid_evaluation(config, experiment_dir, device)
    elif not has_casda:
        logging.info("Skipping FID evaluation (no CASDA groups selected)")

    if args.fid_only:
        logging.info("FID-only mode complete.")
        return

    # ====== Run Experiments ======
    # Loop structure: group (outer) x model (inner)
    # For CASDA groups: inject → train all models → clean
    # This avoids duplicating the baseline_raw directory entirely.
    total_runs = len(model_keys) * len(group_keys)
    run_idx = 0
    start_time = time.time()

    CASDA_GROUPS = {"casda_full", "casda_pruning", "casda_composed", "casda_composed_pruning"}
    CASDA_GROUP_TO_SUBDIR = {
        "casda_full": "casda_full",
        "casda_pruning": "casda_pruning",
        "casda_composed": "casda_composed",
        "casda_composed_pruning": "casda_composed",
    }

    # ratio 그룹도 CASDA 그룹으로 취급 (소스에 따라 casda_full 또는 casda_composed)
    ratio_subdir = "casda_composed" if getattr(args, 'casda_ratio_source', 'full') == 'composed' else "casda_full"
    for rk in casda_ratio_map:
        CASDA_GROUPS.add(rk)
        CASDA_GROUP_TO_SUBDIR[rk] = ratio_subdir

    # Resolve baseline_raw YOLO directory
    baseline_yolo_dir = None
    if args.yolo_dir:
        baseline_yolo_dir = str(Path(args.yolo_dir) / "baseline_raw")
        if not (Path(baseline_yolo_dir) / "images" / "train").exists():
            logging.warning(f"baseline_raw YOLO dir not found at {baseline_yolo_dir}, "
                            f"inject/clean will be skipped for CASDA groups")
            baseline_yolo_dir = None

    for group_key in group_keys:
        is_casda = group_key in CASDA_GROUPS
        casda_injected = False

        # --- Inject CASDA if needed ---
        if is_casda and baseline_yolo_dir:
            casda_subdir = CASDA_GROUP_TO_SUBDIR[group_key]
            casda_data_dir = None

            # Resolve CASDA data directory from config
            if args.casda_dir:
                casda_data_dir = os.path.join(args.casda_dir, casda_subdir)
            else:
                casda_cfg = config.get('dataset', {}).get('casda', {})
                if casda_subdir == "casda_full":
                    casda_data_dir = casda_cfg.get('full_dir', '')
                elif casda_subdir == "casda_pruning":
                    casda_data_dir = casda_cfg.get('pruning_dir', '')
                elif casda_subdir == "casda_composed":
                    casda_data_dir = casda_cfg.get('composed_dir', '')
                else:
                    casda_data_dir = casda_cfg.get('full_dir', '')

            if casda_data_dir and os.path.exists(casda_data_dir):
                # ratio 그룹이면 max_samples 제한 적용
                ratio_max_samples = None
                if group_key in casda_ratio_map:
                    _, ratio_max_samples = casda_ratio_map[group_key]

                logging.info(f"\n{'='*70}")
                logging.info(f"INJECT: {group_key} → baseline_raw")
                logging.info(f"  Source: {casda_data_dir}")
                logging.info(f"  Target: {baseline_yolo_dir}")
                if ratio_max_samples is not None:
                    logging.info(f"  Max samples: {ratio_max_samples} (ratio-limited)")
                logging.info(f"{'='*70}")

                inject_count = inject_casda_to_baseline(
                    baseline_dir=baseline_yolo_dir,
                    casda_dir=casda_data_dir,
                    max_samples=ratio_max_samples,
                )
                casda_injected = True
                logging.info(f"  → {inject_count} images injected")
            else:
                logging.error(f"CASDA data dir not found: {casda_data_dir}")
                logging.error(f"  Skipping all models for group: {group_key}")
                run_idx += len(model_keys)
                continue

        for model_key in model_keys:
            run_idx += 1
            logging.info(f"\n{'='*70}")
            logging.info(f"Run {run_idx}/{total_runs}: {model_key} + {group_key}")
            logging.info(f"{'='*70}")

            try:
                # For CASDA groups with detection models:
                # Use baseline_raw (which now contains injected CASDA) as dataset_group
                # For segmentation models: use the actual group_key
                # (create_data_loaders handles CASDA via ConcatDataset internally)
                effective_group = group_key
                effective_yolo_dir = args.yolo_dir
                if is_casda and model_key in ULTRALYTICS_MODELS and casda_injected:
                    effective_group = "baseline_raw"
                    effective_yolo_dir = args.yolo_dir

                test_metrics = run_single_experiment(
                    model_key=model_key,
                    dataset_group=effective_group,
                    config=config,
                    experiment_dir=experiment_dir,
                    device=device,
                    resume=args.resume,
                    yolo_dir=effective_yolo_dir,
                    # Pass the actual group key for output directory naming
                    output_group_key=group_key,
                )

                model_name = config['models'][model_key]['name']
                group_name = config['dataset_groups'][group_key]['name']
                reporter.add_result(model_name, group_name, test_metrics)

            except Exception as e:
                logging.error(f"Experiment failed: {model_key} + {group_key}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue

        # --- Clean CASDA after all models are done ---
        if casda_injected and baseline_yolo_dir:
            logging.info(f"\n{'='*70}")
            logging.info(f"CLEAN: Removing {group_key} files from baseline_raw")
            logging.info(f"{'='*70}")
            removed = clean_casda_from_baseline(baseline_dir=baseline_yolo_dir)
            logging.info(f"  → {removed} files removed")

    total_time = time.time() - start_time
    logging.info(f"\nAll experiments completed in {total_time:.1f}s ({total_time/3600:.1f}h)")

    # ====== Results ======
    reporter.print_summary()
    reporter.save_results_json()
    reporter.save_comparison_csv()

    # Save PR curves for detection models
    for result in reporter.results:
        if 'precisions' in result['metrics']:
            reporter.save_pr_curves(
                result['metrics'],
                result['model'].replace(' ', '_'),
                result['dataset'].replace(' ', '_'),
            )

    # ====== Hypothesis Testing ======
    h_results = run_hypothesis_tests(reporter, config)
    h_path = experiment_dir / "hypothesis_results.json"
    with open(h_path, 'w') as f:
        json.dump(h_results, f, indent=2, default=str)

    logging.info(f"\nAll results saved to: {experiment_dir}")
    logging.info("Benchmark experiment complete!")


if __name__ == "__main__":
    main()
