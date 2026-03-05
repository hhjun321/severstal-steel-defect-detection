"""
Evaluation Metrics for CASDA Benchmark Experiments

Implements:
  - mAP (Mean Average Precision) at IoU 0.5 for detection
  - Per-class AP for detection
  - Dice Score for segmentation
  - Per-class Dice Score
  - Precision-Recall curves
  - FID (Frechet Inception Distance) for synthetic data quality
"""

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import csv
import logging


# ============================================================================
# IoU / Overlap Utilities
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes."""
    n = len(boxes1)
    m = len(boxes2)
    iou_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    return iou_matrix


# ============================================================================
# Detection Metrics (mAP, Per-class AP, PR curves)
# ============================================================================

class DetectionEvaluator:
    """
    Evaluates object detection models using mAP and per-class AP.
    
    Accumulates predictions and ground truths across batches,
    then computes metrics on the full dataset.
    """

    def __init__(self, num_classes: int = 4, iou_threshold: float = 0.5,
                 image_size: Tuple[int, int] = (640, 640)):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.image_size = image_size  # (width, height) for GT denormalization
        self.reset()

    def reset(self):
        """Reset accumulated predictions and ground truths."""
        # Per-class accumulators
        self.all_detections = {c: [] for c in range(self.num_classes)}   # [(score, is_tp)]
        self.num_gt_per_class = {c: 0 for c in range(self.num_classes)}

    def update(
        self,
        predictions: List[Dict],
        targets: List[torch.Tensor],
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Accumulate batch predictions and ground truths.
        
        Args:
            predictions: List of dicts per image with 'boxes', 'scores', 'labels'
            targets: List of [N, 5] tensors (class, cx, cy, w, h) in YOLO format
            image_size: Optional (width, height) override; defaults to self.image_size
        """
        img_w, img_h = image_size if image_size is not None else self.image_size

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu().numpy() if isinstance(pred['boxes'], torch.Tensor) else np.array(pred['boxes'])
            pred_scores = pred['scores'].cpu().numpy() if isinstance(pred['scores'], torch.Tensor) else np.array(pred['scores'])
            pred_labels = pred['labels'].cpu().numpy() if isinstance(pred['labels'], torch.Tensor) else np.array(pred['labels'])

            # Convert YOLO target (normalized cxcywh) to xyxy pixel coords
            target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else np.array(target)
            gt_boxes = []
            gt_labels = []
            for t in target_np:
                cls_id = int(t[0])
                cx, cy, bw, bh = t[1], t[2], t[3], t[4]
                # Denormalize using image dimensions
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(cls_id)
                self.num_gt_per_class[cls_id] += 1

            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            gt_labels = np.array(gt_labels) if gt_labels else np.array([])

            # Match predictions to ground truths per class
            for cls_id in range(self.num_classes):
                cls_pred_mask = pred_labels == cls_id
                cls_gt_mask = gt_labels == cls_id

                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]
                cls_gt_boxes = gt_boxes[cls_gt_mask]

                if len(cls_pred_boxes) == 0:
                    continue

                # Sort by confidence (descending)
                sorted_idx = np.argsort(-cls_pred_scores)
                cls_pred_boxes = cls_pred_boxes[sorted_idx]
                cls_pred_scores = cls_pred_scores[sorted_idx]

                matched_gt = set()
                for i in range(len(cls_pred_boxes)):
                    is_tp = False
                    if len(cls_gt_boxes) > 0:
                        ious = np.array([
                            compute_iou(cls_pred_boxes[i], gt_box)
                            for gt_box in cls_gt_boxes
                        ])
                        best_gt = np.argmax(ious)
                        if ious[best_gt] >= self.iou_threshold and best_gt not in matched_gt:
                            is_tp = True
                            matched_gt.add(best_gt)

                    self.all_detections[cls_id].append((cls_pred_scores[i], is_tp))

    def compute_ap(self, class_id: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Average Precision for a single class.
        
        Returns:
            ap: Average Precision value
            precisions: Precision values at each recall level
            recalls: Recall values
        """
        detections = self.all_detections[class_id]
        num_gt = self.num_gt_per_class[class_id]

        if num_gt == 0:
            return 0.0, np.array([]), np.array([])

        if len(detections) == 0:
            return 0.0, np.array([0.0]), np.array([0.0])

        # Sort by score descending
        detections.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        for i, (score, is_tp) in enumerate(detections):
            if is_tp:
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Interpolate precision (PASCAL VOC 11-point or all-point)
        # All-point interpolation
        mrec = np.concatenate([[0.0], recalls, [1.0]])
        mpre = np.concatenate([[1.0], precisions, [0.0]])

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Compute AP as area under PR curve
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

        return ap, precisions, recalls

    def compute_metrics(self) -> Dict:
        """
        Compute all detection metrics.
        
        Returns:
            Dict with 'mAP@0.5', 'class_ap', 'precisions', 'recalls'
        """
        class_aps = {}
        class_precisions = {}
        class_recalls = {}

        for cls_id in range(self.num_classes):
            ap, prec, rec = self.compute_ap(cls_id)
            class_aps[f"Class{cls_id + 1}"] = ap
            class_precisions[f"Class{cls_id + 1}"] = prec.tolist() if len(prec) > 0 else []
            class_recalls[f"Class{cls_id + 1}"] = rec.tolist() if len(rec) > 0 else []

        mAP = np.mean(list(class_aps.values())) if class_aps else 0.0

        return {
            'mAP@0.5': float(mAP),
            'class_ap': class_aps,
            'precisions': class_precisions,
            'recalls': class_recalls,
            'num_gt_per_class': {f"Class{k+1}": v for k, v in self.num_gt_per_class.items()},
        }


# ============================================================================
# Segmentation Metrics (Dice Score, IoU)
# ============================================================================

class SegmentationEvaluator:
    """
    Evaluates segmentation models using Dice Score and IoU.
    """

    def __init__(self, num_classes: int = 4, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.pred_sum = np.zeros(self.num_classes)
        self.gt_sum = np.zeros(self.num_classes)
        self.num_samples = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Accumulate batch predictions and targets.
        
        Args:
            predictions: (B, C, H, W) logits or probabilities
            targets: (B, C, H, W) binary ground truth masks
        """
        # Apply sigmoid if logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        pred_binary = (predictions > self.threshold).float()

        # Ensure same spatial size
        if pred_binary.shape[2:] != targets.shape[2:]:
            pred_binary = F.interpolate(
                pred_binary, size=targets.shape[2:],
                mode='nearest',
            )

        pred_np = pred_binary.cpu().numpy()
        target_np = targets.cpu().numpy()

        batch_size = pred_np.shape[0]
        self.num_samples += batch_size

        for c in range(self.num_classes):
            p = pred_np[:, c].reshape(batch_size, -1)
            t = target_np[:, c].reshape(batch_size, -1)

            self.intersection[c] += (p * t).sum()
            self.union[c] += ((p + t) > 0).sum()
            self.pred_sum[c] += p.sum()
            self.gt_sum[c] += t.sum()

    def compute_metrics(self) -> Dict:
        """
        Compute all segmentation metrics.
        
        Returns:
            Dict with 'dice_mean', 'iou_mean', 'class_dice', 'class_iou'
        """
        smooth = 1e-6
        class_dice = {}
        class_iou = {}

        for c in range(self.num_classes):
            dice = (2.0 * self.intersection[c] + smooth) / (self.pred_sum[c] + self.gt_sum[c] + smooth)
            iou = (self.intersection[c] + smooth) / (self.union[c] + smooth)
            class_dice[f"Class{c + 1}"] = float(dice)
            class_iou[f"Class{c + 1}"] = float(iou)

        dice_values = list(class_dice.values())
        iou_values = list(class_iou.values())

        return {
            'dice_mean': float(np.mean(dice_values)),
            'iou_mean': float(np.mean(iou_values)),
            'class_dice': class_dice,
            'class_iou': class_iou,
            'num_samples': self.num_samples,
        }


# ============================================================================
# FID Score Computation
# ============================================================================

class _InceptionImageDataset(Dataset):
    """InceptionV3용 이미지 Dataset — DataLoader num_workers 활용을 위한 클래스.

    이미지를 299×299로 리사이즈하고 ImageNet 정규화를 적용한 뒤
    (C, H, W) float32 텐서로 반환한다.
    """

    # ImageNet 정규화 상수 (모듈 수준 상수로 재사용)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """반환: (image_tensor [C,H,W], original_index).

        이미지 로딩 실패 시 zero 텐서 + index=-1 반환 (collate에서 필터).
        """
        import cv2

        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            return torch.zeros(3, 299, 299, dtype=torch.float32), -1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        # HWC → CHW
        tensor = torch.from_numpy(img.transpose(2, 0, 1))
        return tensor, idx


def _fid_collate_fn(batch):
    """로딩 실패(index==-1) 항목을 필터링하는 collate 함수."""
    tensors, indices = zip(*batch)
    valid = [(t, i) for t, i in zip(tensors, indices) if i >= 0]
    if not valid:
        return torch.empty(0, 3, 299, 299), []
    imgs, idxs = zip(*valid)
    return torch.stack(imgs), list(idxs)


class FIDCalculator:
    """
    Frechet Inception Distance (FID) for evaluating synthetic data quality.

    Uses InceptionV3 features to compute the distance between
    real and generated image distributions.

    v5.5 성능 최적화:
      - O1: Feature 일괄 추출 + 딕셔너리 캐싱 (overall → per-class 재사용)
      - O2: DataLoader + num_workers로 이미지 I/O 병렬화
      - O3: Feature 디스크 캐시 (.npy) — 동일 이미지 세트 재실행 시 즉시 로드

    하위 호환: 기존 compute_fid() / compute_fid_per_class() 시그니처 유지.
    새 파라미터는 모두 기본값이 기존 동작과 동일.

    Requires: torch, torchvision
    """

    def __init__(self, device: str = 'cuda', dims: int = 2048):
        self.device = device
        self.dims = dims
        self._model = None
        # O1: 메모리 feature 캐시 {image_path → feature_vector}
        self._feature_cache: Dict[str, np.ndarray] = {}

    def _get_inception_model(self):
        """Lazy-load InceptionV3."""
        if self._model is None:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            except (ImportError, TypeError):
                from torchvision.models import inception_v3
                model = inception_v3(pretrained=True)

            # Remove final FC to get 2048-dim features
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    # ------------------------------------------------------------------
    # O3: 디스크 캐시 유틸리티
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cache_key(image_paths: List[str]) -> str:
        """이미지 경로 리스트의 결정적 해시를 생성 (디스크 캐시 키)."""
        h = hashlib.sha256()
        for p in sorted(image_paths):
            h.update(p.encode('utf-8'))
        return h.hexdigest()[:16]

    @staticmethod
    def _try_load_disk_cache(cache_dir: Optional[Path], cache_key: str,
                             dims: int) -> Optional[np.ndarray]:
        """디스크 캐시에서 feature array 로드 시도. 없으면 None 반환."""
        if cache_dir is None:
            return None
        cache_file = cache_dir / f"fid_features_{cache_key}.npy"
        if cache_file.exists():
            try:
                features = np.load(str(cache_file))
                if features.ndim == 2 and features.shape[1] == dims:
                    logging.info(f"  디스크 캐시 로드: {cache_file.name} "
                                 f"({features.shape[0]} features)")
                    return features
            except Exception as e:
                logging.warning(f"  디스크 캐시 로드 실패 ({cache_file.name}): {e}")
        return None

    @staticmethod
    def _save_disk_cache(cache_dir: Optional[Path], cache_key: str,
                         features: np.ndarray) -> None:
        """feature array를 디스크에 저장."""
        if cache_dir is None:
            return
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"fid_features_{cache_key}.npy"
            np.save(str(cache_file), features)
            logging.info(f"  디스크 캐시 저장: {cache_file.name} "
                         f"({features.shape[0]} features)")
        except Exception as e:
            logging.warning(f"  디스크 캐시 저장 실패: {e}")

    # ------------------------------------------------------------------
    # O2: DataLoader 기반 feature 추출
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(
        self,
        image_paths: List[str],
        batch_size: int = 64,
        num_workers: int = 4,
        cache_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """Extract InceptionV3 features from image files.

        O1: 메모리 캐시(_feature_cache)에 이미 있는 이미지는 건너뛰고
            새 이미지만 추출한 뒤 캐시를 업데이트한다.
        O2: DataLoader + num_workers로 이미지 I/O를 병렬화한다.
        O3: cache_dir이 지정되면 전체 feature array를 디스크에 캐싱한다.

        Args:
            image_paths: 이미지 파일 경로 리스트
            batch_size: InceptionV3 추론 배치 크기
            num_workers: DataLoader worker 수 (기본 4, Windows에서는 0 권장)
            cache_dir: .npy 디스크 캐시 디렉토리 (None이면 비활성)

        Returns:
            (N, dims) feature array
        """
        if not image_paths:
            return np.array([]).reshape(0, self.dims)

        # O3: 디스크 캐시 확인 (전체 세트 단위)
        cache_key = self._compute_cache_key(image_paths)
        disk_cached = self._try_load_disk_cache(cache_dir, cache_key, self.dims)
        if disk_cached is not None:
            # 메모리 캐시도 업데이트
            for i, p in enumerate(image_paths):
                if i < len(disk_cached):
                    self._feature_cache[p] = disk_cached[i]
            return disk_cached

        # O1: 메모리 캐시에서 이미 추출된 이미지 식별
        uncached_paths = [p for p in image_paths if p not in self._feature_cache]

        if uncached_paths:
            model = self._get_inception_model()
            dataset = _InceptionImageDataset(uncached_paths)

            # O2: num_workers > 0 으로 I/O 병렬화
            # Windows에서는 num_workers=0 이 안전 (fork 미지원)
            import sys
            effective_workers = num_workers if sys.platform != 'win32' else 0

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=effective_workers,
                pin_memory=(self.device != 'cpu'),
                collate_fn=_fid_collate_fn,
                prefetch_factor=2 if effective_workers > 0 else None,
                persistent_workers=effective_workers > 0,
            )

            extracted_count = 0
            for batch_tensor, batch_indices in loader:
                if batch_tensor.numel() == 0:
                    continue
                batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                feat = model(batch_tensor).cpu().numpy()
                for j, orig_idx in enumerate(batch_indices):
                    path = uncached_paths[orig_idx]
                    self._feature_cache[path] = feat[j]
                    extracted_count += 1

            logging.info(f"  InceptionV3 feature 추출: {extracted_count}장 "
                         f"(캐시 히트: {len(image_paths) - len(uncached_paths)}장)")

        # 요청된 순서대로 feature 수집
        feature_list = []
        for p in image_paths:
            if p in self._feature_cache:
                feature_list.append(self._feature_cache[p])
        # 로딩 실패 이미지는 feature_cache에 없으므로 자연스럽게 제외됨

        if not feature_list:
            return np.array([]).reshape(0, self.dims)

        features = np.stack(feature_list, axis=0)

        # O3: 디스크 캐시 저장
        self._save_disk_cache(cache_dir, cache_key, features)

        return features

    def clear_cache(self) -> None:
        """메모리 feature 캐시를 초기화한다."""
        self._feature_cache.clear()

    @staticmethod
    def _compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of feature set."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    @staticmethod
    def _calculate_fid(mu1, sigma1, mu2, sigma2) -> float:
        """
        Compute FID between two Gaussians.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        from scipy import linalg

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def compute_fid(
        self,
        real_image_paths: List[str],
        generated_image_paths: List[str],
        batch_size: int = 64,
        num_workers: int = 4,
        cache_dir: Optional[Path] = None,
    ) -> float:
        """
        Compute FID between real and generated image sets.

        Args:
            real_image_paths: Paths to real images
            generated_image_paths: Paths to generated images
            batch_size: Batch size for feature extraction
            num_workers: DataLoader worker 수 (O2)
            cache_dir: 디스크 캐시 디렉토리 (O3, None이면 비활성)

        Returns:
            FID score (lower is better)
        """
        real_features = self._extract_features(
            real_image_paths, batch_size, num_workers, cache_dir)
        gen_features = self._extract_features(
            generated_image_paths, batch_size, num_workers, cache_dir)

        if len(real_features) < 2 or len(gen_features) < 2:
            return float('inf')

        mu1, sigma1 = self._compute_statistics(real_features)
        mu2, sigma2 = self._compute_statistics(gen_features)

        return self._calculate_fid(mu1, sigma1, mu2, sigma2)

    def compute_fid_per_class(
        self,
        real_image_paths_by_class: Dict[int, List[str]],
        gen_image_paths_by_class: Dict[int, List[str]],
        batch_size: int = 64,
        num_workers: int = 4,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Compute FID per defect class.

        O1 최적화: overall FID에서 이미 추출된 feature는 메모리 캐시에서
        재사용되므로 per-class 계산 시 InceptionV3를 다시 통과하지 않는다.
        """
        results = {}
        for cls_id in sorted(real_image_paths_by_class.keys()):
            real_paths = real_image_paths_by_class.get(cls_id, [])
            gen_paths = gen_image_paths_by_class.get(cls_id, [])
            if len(real_paths) < 2 or len(gen_paths) < 2:
                results[f"Class{cls_id + 1}_FID"] = float('inf')
            else:
                results[f"Class{cls_id + 1}_FID"] = self.compute_fid(
                    real_paths, gen_paths, batch_size,
                    num_workers, cache_dir,
                )
        return results

    def compute_fid_with_preextract(
        self,
        all_real_paths: List[str],
        all_gen_paths: List[str],
        real_by_class: Optional[Dict[int, List[str]]] = None,
        gen_by_class: Optional[Dict[int, List[str]]] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Overall + per-class FID를 한 번의 feature 추출로 계산 (O1 핵심).

        1) 모든 unique 이미지 경로를 수집하여 한 번에 feature 추출
        2) 메모리 캐시에서 슬라이싱하여 overall FID 계산
        3) 메모리 캐시에서 슬라이싱하여 per-class FID 계산

        Args:
            all_real_paths: overall FID용 real 이미지 경로
            all_gen_paths: overall FID용 synthetic 이미지 경로
            real_by_class: per-class real 이미지 {cls_id: [paths]}
            gen_by_class: per-class synthetic 이미지 {cls_id: [paths]}
            batch_size: InceptionV3 추론 배치 크기
            num_workers: DataLoader worker 수
            cache_dir: 디스크 캐시 디렉토리

        Returns:
            {'fid_overall': float, 'Class1_FID': float, ...}
        """
        results: Dict[str, float] = {}

        # ── 1) 모든 unique 경로 수집하여 일괄 추출 ──
        all_paths_set: set = set(all_real_paths) | set(all_gen_paths)
        if real_by_class:
            for paths in real_by_class.values():
                all_paths_set.update(paths)
        if gen_by_class:
            for paths in gen_by_class.values():
                all_paths_set.update(paths)

        all_unique = sorted(all_paths_set)
        logging.info(f"FID 일괄 feature 추출: {len(all_unique)}장 unique 이미지 "
                     f"(real={len(all_real_paths)}, gen={len(all_gen_paths)})")
        self._extract_features(all_unique, batch_size, num_workers, cache_dir)

        # ── 2) Overall FID — 캐시에서 feature 슬라이싱 ──
        real_feats = self._gather_cached_features(all_real_paths)
        gen_feats = self._gather_cached_features(all_gen_paths)

        if len(real_feats) < 2 or len(gen_feats) < 2:
            results['fid_overall'] = float('inf')
        else:
            mu1, s1 = self._compute_statistics(real_feats)
            mu2, s2 = self._compute_statistics(gen_feats)
            results['fid_overall'] = self._calculate_fid(mu1, s1, mu2, s2)

        logging.info(f"FID Score (overall): {results.get('fid_overall', 'inf'):.2f}")

        # ── 3) Per-class FID — 캐시에서 feature 슬라이싱 ──
        if real_by_class and gen_by_class:
            for cls_id in sorted(real_by_class.keys()):
                real_paths = real_by_class.get(cls_id, [])
                gen_paths = gen_by_class.get(cls_id, [])
                if len(real_paths) < 2 or len(gen_paths) < 2:
                    results[f"Class{cls_id + 1}_FID"] = float('inf')
                    continue
                r_feats = self._gather_cached_features(real_paths)
                g_feats = self._gather_cached_features(gen_paths)
                if len(r_feats) < 2 or len(g_feats) < 2:
                    results[f"Class{cls_id + 1}_FID"] = float('inf')
                else:
                    m1, c1 = self._compute_statistics(r_feats)
                    m2, c2 = self._compute_statistics(g_feats)
                    results[f"Class{cls_id + 1}_FID"] = self._calculate_fid(
                        m1, c1, m2, c2)

            for key, val in sorted(results.items()):
                if key.startswith("Class"):
                    logging.info(f"  {key}: {val:.2f}")

        return results

    def _gather_cached_features(self, paths: List[str]) -> np.ndarray:
        """메모리 캐시에서 지정된 경로의 feature를 수집하여 반환."""
        feats = [self._feature_cache[p] for p in paths
                 if p in self._feature_cache]
        if not feats:
            return np.array([]).reshape(0, self.dims)
        return np.stack(feats, axis=0)


# ============================================================================
# Results Reporter
# ============================================================================

class BenchmarkReporter:
    """Formats and saves benchmark experiment results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def add_result(
        self,
        model_name: str,
        dataset_group: str,
        metrics: Dict,
        training_history: Optional[Dict] = None,
    ):
        """Add a single experiment result."""
        result = {
            'model': model_name,
            'dataset': dataset_group,
            'metrics': metrics,
        }
        if training_history:
            result['training_history'] = training_history
        self.results.append(result)

    def save_results_json(self):
        """Save all results to JSON."""
        path = self.output_dir / "benchmark_results.json"
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to: {path}")

    def save_comparison_csv(self):
        """
        Save comparison table as CSV matching the experiment.md format:
        
        | Model | Dataset | mAP@0.5 | Dice | Class1 AP | ... | Class4 AP |
        """
        path = self.output_dir / "benchmark_comparison.csv"
        
        fieldnames = [
            'Model', 'Dataset', 'mAP@0.5', 'Dice',
            'Class1_AP', 'Class2_AP', 'Class3_AP', 'Class4_AP',
        ]

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                metrics = result['metrics']
                row = {
                    'Model': result['model'],
                    'Dataset': result['dataset'],
                    'mAP@0.5': f"{metrics.get('mAP@0.5', 0.0):.4f}",
                    'Dice': f"{metrics.get('dice_mean', 0.0):.4f}",
                }
                # Per-class AP
                class_ap = metrics.get('class_ap', {})
                for i in range(1, 5):
                    key = f"Class{i}"
                    row[f"Class{i}_AP"] = f"{class_ap.get(key, 0.0):.4f}"

                writer.writerow(row)

        print(f"Comparison table saved to: {path}")

    def save_pr_curves(self, metrics: Dict, model_name: str, dataset_group: str):
        """Save precision-recall curves as JSON data (for later plotting)."""
        pr_data = {
            'model': model_name,
            'dataset': dataset_group,
            'precisions': metrics.get('precisions', {}),
            'recalls': metrics.get('recalls', {}),
        }
        path = self.output_dir / f"pr_curve_{model_name}_{dataset_group}.json"
        with open(path, 'w') as f:
            json.dump(pr_data, f, indent=2)

    def print_summary(self):
        """Print formatted summary table."""
        print("\n" + "=" * 100)
        print("CASDA Benchmark Results Summary")
        print("=" * 100)
        print(f"{'Model':<15} {'Dataset':<20} {'mAP@0.5':>8} {'Dice':>8} "
              f"{'C1 AP':>8} {'C2 AP':>8} {'C3 AP':>8} {'C4 AP':>8}")
        print("-" * 100)

        for result in self.results:
            m = result['metrics']
            cap = m.get('class_ap', {})
            print(f"{result['model']:<15} {result['dataset']:<20} "
                  f"{m.get('mAP@0.5', 0.0):>8.4f} {m.get('dice_mean', 0.0):>8.4f} "
                  f"{cap.get('Class1', 0.0):>8.4f} {cap.get('Class2', 0.0):>8.4f} "
                  f"{cap.get('Class3', 0.0):>8.4f} {cap.get('Class4', 0.0):>8.4f}")

        print("=" * 100)
