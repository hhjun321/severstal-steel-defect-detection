"""
Ultralytics YOLO Trainer for CASDA Benchmark

Wraps ultralytics YOLO .train() / .val() to fit into the benchmark
experiment framework. Handles:
  1. Preparing YOLO-format dataset from Severstal CSV + split
  2. Creating the ultralytics YOLO model (YOLO-MFD or EB-YOLOv8)
  3. Training with ultralytics native pipeline
  4. Collecting results in benchmark-compatible format
  5. Running test evaluation and returning metrics

This replaces BenchmarkTrainer for detection models.
BenchmarkTrainer remains for segmentation (DeepLabV3+).
"""

import os
import json
import time
import logging
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UltralyticsTrainer:
    """
    Ultralytics-based trainer for YOLO detection models.
    
    Replaces BenchmarkTrainer for detection models by using
    ultralytics YOLO's native training pipeline.
    """

    def __init__(
        self,
        model_key: str,
        model_config: Dict,
        dataset_config: Dict,
        group_config: Dict,
        dataset_group: str,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        output_dir: str,
        device: str = 'cuda',
        resume: bool = False,
        yolo_dir: Optional[str] = None,
    ):
        """
        Args:
            model_key: "yolo_mfd" or "eb_yolov8"
            model_config: Model config dict from YAML
            dataset_config: Dataset config dict from YAML
            group_config: Dataset group config dict
            dataset_group: Group key string
            train_ids, val_ids, test_ids: Image ID lists
            output_dir: Base output directory for this run
            device: Training device
            resume: Whether to resume from existing checkpoint
            yolo_dir: Pre-converted YOLO dataset directory (skip conversion if valid)
        """
        self.model_key = model_key
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.group_config = group_config
        self.dataset_group = dataset_group
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.output_dir = Path(output_dir)
        self.device = device
        self.resume = resume
        self.yolo_dir = yolo_dir

        # Training config
        train_cfg = model_config.get('training', {})
        self.epochs = train_cfg.get('epochs', 300)
        self.batch_size = train_cfg.get('batch_size', 16)
        self.lr = train_cfg.get('learning_rate', 0.001)
        self.weight_decay = train_cfg.get('weight_decay', 0.0005)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 10)
        self.patience = train_cfg.get('early_stopping_patience', 30)
        self.use_amp = train_cfg.get('use_amp', True)
        self.conf_threshold = train_cfg.get('conf_threshold', 0.25)
        self.iou_threshold = train_cfg.get('iou_threshold', 0.5)
        self.input_size = model_config.get('input_size', [640, 640])
        self.num_classes = dataset_config.get('num_classes', 4)

    @staticmethod
    def _compute_scale_range(scale_range) -> float:
        """
        Convert YAML scale_range [min, max] to Ultralytics scale parameter.
        
        Ultralytics 'scale' is a symmetric deviation from 1.0.
        Example: scale_range=[0.8, 1.2] → scale=0.2 (±20%)
                 scale_range=[0.9, 1.1] → scale=0.1 (±10%)
        """
        if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
            return max(abs(1.0 - scale_range[0]), abs(scale_range[1] - 1.0))
        return 0.2  # default ±20%

    def _resolve_path(self, raw_path: str) -> str:
        """Resolve path: return as-is if absolute, else relative to project root."""
        if os.path.isabs(raw_path):
            return raw_path
        project_root = Path(__file__).resolve().parent.parent.parent
        return str(project_root / raw_path)

    def _prepare_dataset(self) -> str:
        """
        Prepare YOLO-format dataset and return path to dataset.yaml.
        
        If yolo_dir is specified and contains a valid pre-converted dataset
        (for this dataset_group), skip conversion and return the existing
        dataset.yaml path directly. This saves significant time on repeated runs.
        """
        # Try to use pre-converted dataset first
        if self.yolo_dir:
            from src.training.dataset_yolo import validate_yolo_dataset
            cached_yaml = validate_yolo_dataset(
                yolo_dir=self.yolo_dir,
                dataset_group=self.dataset_group,
            )
            if cached_yaml:
                logger.info(f"Using pre-converted YOLO dataset (skipping conversion)")
                logger.info(f"  dataset.yaml: {cached_yaml}")
                return cached_yaml
            else:
                logger.info(f"Pre-converted dataset not found/invalid at {self.yolo_dir}, "
                            f"will convert from scratch")

        from src.training.dataset_yolo import prepare_yolo_dataset

        image_dir = self._resolve_path(self.dataset_config['image_dir'])
        annotation_csv = self._resolve_path(self.dataset_config['annotation_csv'])

        # Determine CASDA settings
        casda_data = self.group_config.get('casda_data', None)
        casda_dir = None
        casda_mode = None
        casda_config_dict = None

        if casda_data is not None:
            casda_cfg = self.dataset_config.get('casda', {})
            casda_config_dict = casda_cfg
            if casda_data == "full":
                raw_dir = casda_cfg.get('full_dir', 'data/augmented/casda_full')
                casda_dir = self._resolve_path(raw_dir)
                casda_mode = "full"
            elif casda_data == "pruning":
                raw_dir = casda_cfg.get('pruning_dir', 'data/augmented/casda_pruning')
                casda_dir = self._resolve_path(raw_dir)
                casda_mode = "pruning"
            elif casda_data == "composed":
                raw_dir = casda_cfg.get('composed_dir', 'data/augmented/casda_composed')
                casda_dir = self._resolve_path(raw_dir)
                # group_config의 casda_pruning 설정으로 pruning 여부 결정
                pruning_cfg = self.group_config.get('casda_pruning', {})
                if pruning_cfg.get('enabled', False):
                    casda_mode = "pruning"
                    # pruning 설정을 casda_config에 병합하여 전달
                    casda_config_dict = dict(casda_cfg)
                    casda_config_dict['suitability_threshold'] = pruning_cfg.get(
                        'suitability_threshold', casda_cfg.get('suitability_threshold', 0.0))
                    casda_config_dict['pruning_top_k'] = pruning_cfg.get(
                        'top_k', casda_cfg.get('pruning_top_k', 2000))
                    casda_config_dict['stratified'] = pruning_cfg.get('stratified', False)
                else:
                    casda_mode = "full"

        # Determine output directory for converted dataset
        # If yolo_dir specified (but invalid/empty), save there for future reuse
        # Otherwise save in run output directory
        if self.yolo_dir:
            yolo_dataset_dir = str(Path(self.yolo_dir) / self.dataset_group)
        else:
            yolo_dataset_dir = str(self.output_dir / "yolo_dataset")

        class_names = self.dataset_config.get('class_names',
                                               [f"Class{i+1}" for i in range(self.num_classes)])

        yaml_path = prepare_yolo_dataset(
            image_dir=image_dir,
            annotation_csv=annotation_csv,
            train_ids=self.train_ids,
            val_ids=self.val_ids,
            test_ids=self.test_ids,
            output_dir=yolo_dataset_dir,
            dataset_group=self.dataset_group,
            casda_dir=casda_dir,
            casda_mode=casda_mode,
            casda_config=casda_config_dict,
            num_classes=self.num_classes,
            class_names=class_names,
        )

        return yaml_path

    def _create_model(self):
        """Create the ultralytics YOLO model."""
        if self.model_key == "yolo_mfd":
            from src.models.yolo_mfd import create_yolo_mfd
            return create_yolo_mfd(num_classes=self.num_classes, pretrained=True)
        elif self.model_key == "eb_yolov8":
            from src.models.eb_yolov8 import create_eb_yolov8
            return create_eb_yolov8(num_classes=self.num_classes, pretrained=True)
        else:
            raise ValueError(f"Unknown detection model: {self.model_key}")

    def _parse_results(self, results) -> Dict:
        """
        Parse ultralytics training results into benchmark format.
        
        Args:
            results: ultralytics training results object
        
        Returns:
            Dict with training history and metadata
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],
            'learning_rate': [],
            'best_epoch': 0,
            'best_metric': 0.0,
        }

        # ultralytics saves results.csv in the run directory
        results_csv = Path(results.save_dir) / "results.csv"
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                # Clean column names (ultralytics adds spaces)
                df.columns = [c.strip() for c in df.columns]

                # Map ultralytics column names to our format
                if 'train/box_loss' in df.columns:
                    # Total train loss = box + cls + dfl
                    train_loss_cols = [c for c in df.columns if c.startswith('train/')]
                    history['train_loss'] = df[train_loss_cols].sum(axis=1).tolist()

                if 'val/box_loss' in df.columns:
                    val_loss_cols = [c for c in df.columns if c.startswith('val/')]
                    history['val_loss'] = df[val_loss_cols].sum(axis=1).tolist()

                if 'metrics/mAP50(B)' in df.columns:
                    history['val_metric'] = df['metrics/mAP50(B)'].tolist()
                    best_idx = df['metrics/mAP50(B)'].idxmax()
                    history['best_epoch'] = int(best_idx)
                    history['best_metric'] = float(df['metrics/mAP50(B)'].iloc[best_idx])

                if 'lr/pg0' in df.columns:
                    history['learning_rate'] = df['lr/pg0'].tolist()

            except Exception as e:
                logger.warning(f"Failed to parse results.csv: {e}")

        return history

    def _evaluate_test(self, model, dataset_yaml: str) -> Dict:
        """
        Run evaluation on test set using ultralytics .val().
        
        Returns:
            Metrics dict in benchmark format
        """
        logger.info("Running test set evaluation...")

        results = model.val(
            data=dataset_yaml,
            split='test',
            batch=self.batch_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            rect=True,  # rectangular batching for consistent eval
            verbose=False,
        )

        # Parse ultralytics metrics
        metrics = {
            'mAP@0.5': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'mAP@0.5:0.95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
        }

        # Per-class AP
        class_ap = {}
        if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
            ap50 = results.box.ap50
            class_names = self.dataset_config.get('class_names',
                                                   [f"Class{i+1}" for i in range(self.num_classes)])
            for i, name in enumerate(class_names):
                if i < len(ap50):
                    class_ap[name] = float(ap50[i])
                else:
                    class_ap[name] = 0.0
        metrics['class_ap'] = class_ap

        logger.info(f"Test mAP@0.5: {metrics['mAP@0.5']:.4f}")
        for name, ap in class_ap.items():
            logger.info(f"  {name}: {ap:.4f}")

        return metrics

    def train(self) -> Dict:
        """
        Run the full training pipeline using ultralytics.
        
        Steps:
          1. Prepare YOLO-format dataset
          2. Create model
          3. Train with ultralytics .train()
          4. Evaluate on test set
          5. Return metrics in benchmark format
        
        Returns:
            test_metrics dict compatible with BenchmarkReporter
        """
        start_time = time.time()
        run_name = f"{self.model_key}_{self.dataset_group}"

        logger.info(f"\n{'='*60}")
        logger.info(f"UltralyticsTrainer: {run_name}")
        logger.info(f"Epochs: {self.epochs}, LR: {self.lr}, Batch: {self.batch_size}")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        logger.info(f"Patience: {self.patience}, Input: {self.input_size}")
        logger.info(f"{'='*60}")

        # Step 1: Prepare dataset
        logger.info("Step 1: Preparing YOLO-format dataset...")
        dataset_yaml = self._prepare_dataset()
        logger.info(f"Dataset YAML: {dataset_yaml}")

        # Step 2: Create model
        logger.info("Step 2: Creating model...")
        model = self._create_model()

        # Log model info
        model_info = model.info(verbose=False)
        logger.info(f"Model created: {self.model_key}")

        # Step 3: Train
        logger.info("Step 3: Training with ultralytics...")
        train_dir = str(self.output_dir / "ultralytics_train")

        # Determine optimizer name for ultralytics
        optimizer = self.model_config.get('training', {}).get('optimizer', 'AdamW')

        # Check for resume
        resume_path = None
        if self.resume:
            last_pt = Path(train_dir) / run_name / "weights" / "last.pt"
            if last_pt.exists():
                resume_path = str(last_pt)
                logger.info(f"Resuming from: {resume_path}")

        # Run training
        train_kwargs = dict(
            data=dataset_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.input_size[0],  # longest side; rect=True handles 256x1600 aspect ratio
            device=self.device,
            optimizer=optimizer,
            lr0=self.lr,
            lrf=0.01,  # final LR factor (cosine decay to lr0 * 0.01)
            weight_decay=self.weight_decay,
            warmup_epochs=self.warmup_epochs,
            patience=self.patience,
            amp=self.use_amp,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            project=train_dir,
            name=run_name,
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=-1,  # only save best and last
            plots=True,
            rect=True,  # rectangular batching for Severstal 256x1600 (1:6.25 aspect ratio)
            val=True,
        )

        # ---- Apply augmentation settings based on dataset group config ----
        # baseline_trad: map YAML traditional_augmentation to Ultralytics params
        # baseline_raw: use Ultralytics defaults (no override)
        # CASDA groups: use Ultralytics defaults (no override)
        augmentation = self.group_config.get('augmentation', 'none')
        if augmentation == 'traditional':
            trad_config = self.group_config.get('traditional_augmentation', {})
            # Map YAML params to Ultralytics augmentation hyperparameters
            train_kwargs.update(dict(
                fliplr=trad_config.get('horizontal_flip', 0.5),
                flipud=trad_config.get('vertical_flip', 0.3),
                degrees=trad_config.get('rotation_limit', 15.0),
                scale=self._compute_scale_range(trad_config.get('scale_range', [0.8, 1.2])),
                hsv_v=trad_config.get('brightness_limit', 0.2),
                # Disable non-traditional augmentations to isolate effect
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
            ))
            logger.info(f"Traditional augmentation applied: "
                        f"fliplr={train_kwargs['fliplr']}, flipud={train_kwargs['flipud']}, "
                        f"degrees={train_kwargs['degrees']}, scale={train_kwargs['scale']}, "
                        f"hsv_v={train_kwargs['hsv_v']}, mosaic=0, mixup=0, copy_paste=0")

        if resume_path:
            # For resume, load from last.pt and continue
            from ultralytics import YOLO
            model = YOLO(resume_path)
            train_kwargs['resume'] = True

        results = model.train(**train_kwargs)

        # Step 4: Parse training history
        history = self._parse_results(results)
        total_time = time.time() - start_time

        # Determine early stopping info
        total_epochs_trained = len(history.get('val_metric', []))
        early_stopped = total_epochs_trained < self.epochs
        stopped_epoch = total_epochs_trained if early_stopped else self.epochs

        history['early_stopped'] = early_stopped
        history['stopped_epoch'] = stopped_epoch
        history['max_epochs'] = self.epochs
        history['total_time_seconds'] = round(total_time, 1)
        history['use_amp'] = self.use_amp

        logger.info(f"\nTraining completed in {total_time:.1f}s ({total_time/3600:.1f}h)")
        logger.info(f"Best mAP@0.5: {history['best_metric']:.4f} at epoch {history['best_epoch']+1}")
        if early_stopped:
            logger.info(f"Early stopped at epoch {stopped_epoch}/{self.epochs}")

        # Step 5: Load best model and evaluate on test set
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            from ultralytics import YOLO
            best_model = YOLO(str(best_pt))
            test_metrics = self._evaluate_test(best_model, dataset_yaml)
        else:
            logger.warning("No best.pt found, evaluating with last model")
            test_metrics = self._evaluate_test(model, dataset_yaml)

        # Save history
        history_path = self.output_dir / f"{run_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

        # Store history for external access
        self.history = history

        return test_metrics
