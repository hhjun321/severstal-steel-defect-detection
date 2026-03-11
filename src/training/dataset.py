"""
Severstal Steel Defect Dataset Loaders

Provides PyTorch Dataset classes for both detection and segmentation tasks,
supporting 4 dataset groups:
  - Baseline (Raw): Original Severstal only
  - Baseline (Trad): Original + traditional augmentations
  - CASDA-Full: Original + all 5,000 CASDA synthetic images
  - CASDA-Pruning: Original + top 2,000 CASDA images by suitability score
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.rle_utils import rle_decode


# ============================================================================
# Severstal Dataset - Detection Format (YOLO-style)
# ============================================================================

class SeverstalDetectionDataset(Dataset):
    """
    Severstal dataset for object detection (bounding box format).
    
    Returns images and bounding box annotations in YOLO format:
      [class_id, x_center, y_center, width, height] (normalized 0-1)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_csv: str,
        image_ids: List[str],
        input_size: Tuple[int, int] = (640, 640),
        transform=None,
        is_training: bool = True,
    ):
        """
        Args:
            image_dir: Directory containing steel images
            annotation_csv: Path to train.csv with RLE annotations
            image_ids: List of image filenames to include
            input_size: (H, W) resize target for detection models
            transform: Albumentations transform pipeline
            is_training: Whether this is a training set
        """
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.transform = transform
        self.is_training = is_training

        # Load annotations
        self.df = pd.read_csv(annotation_csv)
        # Severstal CSV format: ImageId_ClassId, EncodedPixels
        # Parse into ImageId and ClassId columns
        if 'ImageId_ClassId' in self.df.columns:
            self.df[['ImageId', 'ClassId']] = self.df['ImageId_ClassId'].str.rsplit('_', n=1, expand=True)
            self.df['ClassId'] = self.df['ClassId'].astype(int)
        elif 'ImageId' not in self.df.columns:
            raise ValueError("CSV must have 'ImageId_ClassId' or 'ImageId' column")

        # Filter to requested image IDs
        self.image_ids = [iid for iid in image_ids if iid in self.df['ImageId'].values]
        if len(self.image_ids) == 0:
            # All images (including those with no defects)
            self.image_ids = image_ids

        self.annotations = self._build_annotations()

    def _build_annotations(self) -> Dict[str, List[Dict]]:
        """Convert RLE masks to bounding boxes per image."""
        annotations = {}
        for img_id in self.image_ids:
            img_annots = self.df[self.df['ImageId'] == img_id]
            boxes = []
            for _, row in img_annots.iterrows():
                if pd.isna(row.get('EncodedPixels', None)):
                    continue
                mask = rle_decode(row['EncodedPixels'], (256, 1600))
                if mask.sum() == 0:
                    continue
                # Find connected components for multiple defect instances
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h < 16:  # skip tiny artifacts
                        continue
                    boxes.append({
                        'class_id': int(row['ClassId']) - 1,  # 0-indexed
                        'bbox': [x, y, w, h],  # pixel coordinates
                    })
            annotations[img_id] = boxes
        return annotations

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_path = self.image_dir / img_id

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        boxes = self.annotations.get(img_id, [])

        # Prepare bboxes and labels
        bboxes = []
        labels = []
        for box in boxes:
            x, y, w, h = box['bbox']
            bboxes.append([x, y, x + w, y + h])  # xyxy format
            labels.append(box['class_id'])

        # Apply transforms
        if self.transform and HAS_ALBUMENTATIONS:
            if len(bboxes) > 0:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    labels=labels,
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['labels']
            else:
                transformed = self.transform(image=image, bboxes=[], labels=[])
                image = transformed['image']
                bboxes = []
                labels = []
        else:
            # Manual resize
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Scale bboxes
            sx = self.input_size[1] / orig_w
            sy = self.input_size[0] / orig_h
            bboxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in bboxes]

        # Convert to YOLO format: [class, x_center, y_center, w, h] normalized
        target_h, target_w = self.input_size
        yolo_labels = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0 / target_w
            cy = (y1 + y2) / 2.0 / target_h
            bw = (x2 - x1) / target_w
            bh = (y2 - y1) / target_h
            yolo_labels.append([label, cx, cy, bw, bh])

        if len(yolo_labels) > 0:
            yolo_labels = torch.tensor(yolo_labels, dtype=torch.float32)
        else:
            yolo_labels = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'image': image if isinstance(image, torch.Tensor) else torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            'labels': yolo_labels,
            'image_id': img_id,
        }


# ============================================================================
# Severstal Dataset - Segmentation Format
# ============================================================================

class SeverstalSegmentationDataset(Dataset):
    """
    Severstal dataset for semantic segmentation.
    
    Returns images and multi-class segmentation masks.
    Mask shape: (num_classes, H, W) binary masks per class.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_csv: str,
        image_ids: List[str],
        input_size: Tuple[int, int] = (256, 512),
        num_classes: int = 4,
        transform=None,
        is_training: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.num_classes = num_classes
        self.transform = transform
        self.is_training = is_training

        # Load annotations
        self.df = pd.read_csv(annotation_csv)
        if 'ImageId_ClassId' in self.df.columns:
            self.df[['ImageId', 'ClassId']] = self.df['ImageId_ClassId'].str.rsplit('_', n=1, expand=True)
            self.df['ClassId'] = self.df['ClassId'].astype(int)

        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_path = self.image_dir / img_id

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build multi-class mask
        mask = np.zeros((self.num_classes, 256, 1600), dtype=np.uint8)
        img_annots = self.df[self.df['ImageId'] == img_id]
        for _, row in img_annots.iterrows():
            if pd.isna(row.get('EncodedPixels', None)):
                continue
            cls_id = int(row['ClassId']) - 1  # 0-indexed
            mask[cls_id] = rle_decode(row['EncodedPixels'], (256, 1600))

        # Apply transforms
        if self.transform and HAS_ALBUMENTATIONS:
            # Stack mask channels for albumentations
            mask_hw = np.transpose(mask, (1, 2, 0))  # (H, W, C)
            transformed = self.transform(image=image, mask=mask_hw)
            image = transformed['image']
            mask_hw = transformed['mask']
            if isinstance(mask_hw, np.ndarray):
                mask = torch.from_numpy(np.transpose(mask_hw, (2, 0, 1))).float()
            else:
                mask = mask_hw.permute(2, 0, 1).float()
        else:
            # Manual resize
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Resize each mask channel
            resized_mask = np.zeros(
                (self.num_classes, self.input_size[0], self.input_size[1]),
                dtype=np.float32,
            )
            for c in range(self.num_classes):
                resized_mask[c] = cv2.resize(
                    mask[c].astype(np.float32),
                    (self.input_size[1], self.input_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = torch.from_numpy(resized_mask)

        return {
            'image': image if isinstance(image, torch.Tensor) else torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            'mask': mask,
            'image_id': img_id,
        }


# ============================================================================
# CASDA Synthetic Dataset Loader
# ============================================================================

class CASDASyntheticDataset(Dataset):
    """
    Loads CASDA-generated synthetic steel defect images.
    
    Expects a directory with:
      - images/: generated defect images
      - metadata.json or annotations.csv: class labels and optional masks/bboxes
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "detection",  # "detection" or "segmentation"
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 4,
        suitability_threshold: Optional[float] = None,
        max_samples: Optional[int] = None,
        transform=None,
        stratified: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.input_size = input_size
        self.num_classes = num_classes
        self.transform = transform
        self.stratified = stratified

        # Load metadata
        self.samples = self._load_metadata(suitability_threshold, max_samples)

    def _load_metadata(
        self,
        suitability_threshold: Optional[float],
        max_samples: Optional[int],
    ) -> List[Dict]:
        """Load and filter CASDA metadata."""
        samples = []

        # Try JSON metadata first
        meta_path = self.data_dir / "metadata.json"
        csv_path = self.data_dir / "annotations.csv"

        if meta_path.exists():
            with open(meta_path) as f:
                all_samples = json.load(f)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            all_samples = df.to_dict('records')
        else:
            # Fallback: scan image directory and infer class from filename
            img_dir = self.data_dir / "images"
            if not img_dir.exists():
                img_dir = self.data_dir
            all_samples = []
            for img_path in sorted(img_dir.glob("*.png")):
                # Try to infer class from filename pattern: class{N}_*
                fname = img_path.stem
                class_id = 0
                for i in range(1, 5):
                    if f"class{i}" in fname or f"class_{i}" in fname:
                        class_id = i - 1
                        break
                all_samples.append({
                    'image_path': str(img_path),
                    'class_id': class_id,
                    'suitability_score': 0.0,
                })

        # Filter by suitability
        if suitability_threshold is not None:
            all_samples = [
                s for s in all_samples
                if s.get('suitability_score', 0.0) >= suitability_threshold
            ]

        # Sort by suitability (descending) and limit
        all_samples.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
        if max_samples is not None:
            if self.stratified:
                all_samples = CASDASyntheticDataset._stratified_top_k(all_samples, max_samples)
            else:
                all_samples = all_samples[:max_samples]

        # 이미지 파일 존재 여부 검증 — 누락 파일은 학습 중 crash를 유발하므로 제거
        before_count = len(all_samples)
        verified = []
        missing_count = 0
        for s in all_samples:
            img_path = s.get('image_path', '')
            if not os.path.isabs(img_path):
                img_path = str(self.data_dir / img_path)
            if os.path.exists(img_path):
                verified.append(s)
            else:
                missing_count += 1
        if missing_count > 0:
            logging.warning(
                f"[CASDASyntheticDataset] {missing_count}/{before_count} images not found, "
                f"skipped. ({before_count - missing_count} remaining)"
            )
        return verified

    @staticmethod
    def _stratified_top_k(samples: List[Dict], k: int) -> List[Dict]:
        """
        클래스 비율을 유지하면서 suitability_score 상위 k개를 선택한다.
        
        알고리즘:
          (a) 전체 샘플을 class_id별로 그룹화
          (b) 각 클래스에 비율 기반 할당량 계산 (최소 1개 보장)
          (c) 할당량 합계가 k를 초과하면 큰 그룹부터 1씩 감소
          (d) 할당량 합계가 k 미만이면 여유 있는 그룹에 1씩 추가
          (e) 각 그룹 내에서 suitability_score 내림차순 정렬 후 할당량만큼 선택
        """
        from collections import defaultdict

        if len(samples) <= k:
            return samples

        # (a) class_id별 그룹화
        groups: Dict[int, List[Dict]] = defaultdict(list)
        for s in samples:
            groups[s.get('class_id', 0)].append(s)

        n_total = len(samples)
        n_classes = len(groups)

        # (b) 비율 기반 할당량 (최소 1개 보장)
        quotas: Dict[int, int] = {}
        for cls_id, cls_samples in groups.items():
            quota = max(1, round(len(cls_samples) / n_total * k))
            # 그룹 크기를 초과하지 않도록 제한
            quota = min(quota, len(cls_samples))
            quotas[cls_id] = quota

        # (c) 할당량 합계가 k를 초과하면 큰 그룹부터 1씩 감소
        while sum(quotas.values()) > k:
            # 할당량이 가장 큰 클래스부터 감소 (최소 1 유지)
            largest_cls = max(quotas, key=lambda c: quotas[c])
            if quotas[largest_cls] > 1:
                quotas[largest_cls] -= 1
            else:
                break

        # (d) 할당량 합계가 k 미만이면 여유 있는 그룹에 1씩 추가
        while sum(quotas.values()) < k:
            added = False
            # 여유(그룹 크기 - 할당량)가 큰 클래스부터 추가
            for cls_id in sorted(quotas, key=lambda c: len(groups[c]) - quotas[c], reverse=True):
                if quotas[cls_id] < len(groups[cls_id]):
                    quotas[cls_id] += 1
                    added = True
                    if sum(quotas.values()) >= k:
                        break
            if not added:
                break

        # (e) 각 그룹에서 score 내림차순 상위 할당량만큼 선택
        result = []
        for cls_id, cls_samples in groups.items():
            cls_samples.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
            result.extend(cls_samples[:quotas[cls_id]])

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Resolve image path
        img_path = sample.get('image_path', '')
        if not os.path.isabs(img_path):
            img_path = str(self.data_dir / img_path)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Synthetic image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "detection":
            return self._get_detection_item(image, sample, idx)
        else:
            return self._get_segmentation_item(image, sample, idx)

    def _get_detection_item(self, image: np.ndarray, sample: Dict, idx: int) -> Dict:
        """Return detection-format item."""
        orig_h, orig_w = image.shape[:2]

        # Get bbox if available, otherwise create from mask
        bboxes = sample.get('bboxes', [])
        labels = sample.get('labels', [])

        if not bboxes and 'mask_path' in sample:
            mask_path = sample['mask_path']
            if not os.path.isabs(mask_path):
                mask_path = str(self.data_dir / mask_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h >= 16:
                        bboxes.append([x, y, x + w, y + h])
                        labels.append(sample.get('class_id', 0))

        if not bboxes:
            # Use entire image as bbox with given class
            bboxes = [[0, 0, orig_w, orig_h]]
            labels = [sample.get('class_id', 0)]

        # Resize
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        sx = self.input_size[1] / orig_w
        sy = self.input_size[0] / orig_h
        th, tw = self.input_size

        yolo_labels = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            cx = (x1 * sx + x2 * sx) / 2.0 / tw
            cy = (y1 * sy + y2 * sy) / 2.0 / th
            bw = (x2 - x1) * sx / tw
            bh = (y2 - y1) * sy / th
            yolo_labels.append([label, cx, cy, bw, bh])

        if yolo_labels:
            yolo_labels = torch.tensor(yolo_labels, dtype=torch.float32)
        else:
            yolo_labels = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'image': image,
            'labels': yolo_labels,
            'image_id': Path(sample.get('image_path', f'synthetic_{idx}')).stem,
        }

    def _get_segmentation_item(self, image: np.ndarray, sample: Dict, idx: int) -> Dict:
        """Return segmentation-format item."""
        orig_h, orig_w = image.shape[:2]

        mask = np.zeros((self.num_classes, orig_h, orig_w), dtype=np.uint8)

        # Load mask if available
        if 'mask_path' in sample:
            mask_path = sample['mask_path']
            if not os.path.isabs(mask_path):
                mask_path = str(self.data_dir / mask_path)
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                # Defensive resize: CASDA masks may be 256×256 (ROI patch size)
                # while generated images are 512×512 (ControlNet resolution)
                if m.shape[:2] != (orig_h, orig_w):
                    m = cv2.resize(m, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)
                cls_id = sample.get('class_id', 0)
                mask[cls_id] = (m > 127).astype(np.uint8)

        # Resize
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        resized_mask = np.zeros(
            (self.num_classes, self.input_size[0], self.input_size[1]),
            dtype=np.float32,
        )
        for c in range(self.num_classes):
            resized_mask[c] = cv2.resize(
                mask[c].astype(np.float32),
                (self.input_size[1], self.input_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = torch.from_numpy(resized_mask)

        return {
            'image': image,
            'mask': mask,
            'image_id': Path(sample.get('image_path', f'synthetic_{idx}')).stem,
        }


# ============================================================================
# Data Split & Group Builder
# ============================================================================

def get_image_ids_with_defects(annotation_csv: str) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Parse annotation CSV and return image IDs with their defect classes.
    
    Returns:
        image_ids: Sorted list of unique image IDs
        image_classes: Dict mapping image_id -> list of class_ids (1-indexed)
    """
    df = pd.read_csv(annotation_csv)
    if 'ImageId_ClassId' in df.columns:
        df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.rsplit('_', n=1, expand=True)
        df['ClassId'] = df['ClassId'].astype(int)

    # Only images with actual defects
    defect_df = df[df['EncodedPixels'].notna()]
    image_classes = {}
    for _, row in defect_df.iterrows():
        img_id = row['ImageId']
        cls_id = int(row['ClassId'])
        if img_id not in image_classes:
            image_classes[img_id] = []
        image_classes[img_id].append(cls_id)

    image_ids = sorted(image_classes.keys())
    return image_ids, image_classes


def split_dataset(
    image_ids: List[str],
    image_classes: Dict[str, List[int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified split by primary defect class.
    
    Returns:
        train_ids, val_ids, test_ids
    """
    # Use primary (first) class for stratification
    primary_classes = [image_classes[iid][0] for iid in image_ids]

    train_ids, temp_ids, train_cls, temp_cls = train_test_split(
        image_ids, primary_classes,
        test_size=1 - train_ratio,
        stratify=primary_classes,
        random_state=seed,
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1 - val_size,
        stratify=temp_cls,
        random_state=seed,
    )
    return train_ids, val_ids, test_ids


def build_transforms(
    mode: str,
    input_size: Tuple[int, int],
    augmentation: str = "none",
    trad_config: Optional[Dict] = None,
) -> Optional[object]:
    """
    Build albumentations transform pipeline.
    
    Args:
        mode: "detection" or "segmentation"
        input_size: (H, W)
        augmentation: "none" or "traditional"
        trad_config: Traditional augmentation config from YAML
    """
    if not HAS_ALBUMENTATIONS:
        return None

    transforms_list = []

    # Resize
    transforms_list.append(A.Resize(height=input_size[0], width=input_size[1]))

    # Traditional augmentation
    if augmentation == "traditional" and trad_config:
        transforms_list.extend([
            A.HorizontalFlip(p=trad_config.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=trad_config.get('vertical_flip', 0.3)),
            A.Rotate(limit=trad_config.get('rotation_limit', 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=trad_config.get('brightness_limit', 0.2),
                contrast_limit=trad_config.get('contrast_limit', 0.2),
                p=0.5,
            ),
        ])
        # Scale augmentation (if scale_range defined in YAML)
        scale_range = trad_config.get('scale_range', None)
        if scale_range and isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
            transforms_list.append(
                A.RandomScale(
                    scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
                    p=0.5,
                )
            )
            # RandomScale 후 크기가 변경되므로 다시 고정 크기로 resize (batch stack 보장)
            transforms_list.append(A.Resize(height=input_size[0], width=input_size[1]))

    # Normalize and convert to tensor
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    if mode == "detection":
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3,
            ),
        )
    else:
        return A.Compose(transforms_list)


def _filter_existing_images(
    image_ids: List[str],
    image_dir: str,
    split_name: str,
) -> List[str]:
    """디스크에 실제 존재하는 이미지만 필터링하고 누락 이미지는 경고 로그 출력.

    Args:
        image_ids: 필터링할 이미지 ID 목록
        image_dir: 이미지 파일이 위치한 디렉토리 경로
        split_name: 로그에 표시할 split 이름 (train/val/test)

    Returns:
        디스크에 존재하는 이미지 ID만 포함된 리스트

    Raises:
        ValueError: 존재하는 이미지가 하나도 없을 경우
    """
    existing = []
    missing = []
    for img_id in image_ids:
        if os.path.exists(os.path.join(image_dir, img_id)):
            existing.append(img_id)
        else:
            missing.append(img_id)

    if missing:
        examples = missing[:5]
        logging.warning(
            f"[{split_name}] {len(missing)}/{len(image_ids)} images not found "
            f"in {image_dir}, removed from dataset. Examples: {examples}"
        )
    if not existing:
        raise ValueError(
            f"[{split_name}] No images found in {image_dir}! "
            f"Total {len(image_ids)} IDs were all missing. Check --data-dir path."
        )
    return existing


def create_data_loaders(
    config: Dict,
    dataset_group: str,
    model_type: str,
    input_size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test DataLoaders for a given dataset group and model type.
    
    Args:
        config: Full experiment config dict
        dataset_group: One of "baseline_raw", "baseline_trad", "casda_full", "casda_pruning"
        model_type: "detection" or "segmentation"
        input_size: (H, W) for model input
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader, split_info
        split_info contains train_ids, val_ids, test_ids, and split config for reproducibility verification.
    """
    ds_config = config['dataset']
    group_config = config['dataset_groups'][dataset_group]

    # Resolve data paths relative to project root (skip if already absolute)
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_csv = ds_config['annotation_csv']
    raw_img = ds_config['image_dir']
    annotation_csv = raw_csv if os.path.isabs(raw_csv) else str(project_root / raw_csv)
    image_dir = raw_img if os.path.isabs(raw_img) else str(project_root / raw_img)

    # Get image IDs and split
    # split_csv가 지정되면 사전 생성된 분할 CSV에서 로드 (동적 분할 건너뜀)
    split_csv = ds_config.get('split_csv', None)
    if split_csv is not None and os.path.exists(split_csv):
        split_df = pd.read_csv(split_csv, comment='#')
        train_ids = split_df[split_df['Split'] == 'train']['ImageId'].tolist()
        val_ids = split_df[split_df['Split'] == 'val']['ImageId'].tolist()
        test_ids = split_df[split_df['Split'] == 'test']['ImageId'].tolist()
        # image_classes는 split_info 통계용으로 여전히 필요
        _, image_classes = get_image_ids_with_defects(annotation_csv)
        logging.info(f"Loaded pre-defined split from {split_csv}: "
                     f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
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
            logging.warning(f"Split CSV not found: {split_csv} — falling back to dynamic split")

    # Determine augmentation
    augmentation = group_config.get('augmentation', 'none')
    trad_config = group_config.get('traditional_augmentation', None)

    # Build transforms
    train_transform = build_transforms(model_type, input_size, augmentation, trad_config)
    eval_transform = build_transforms(model_type, input_size, "none")

    # 디스크에 실제 존재하는 이미지만 유지 (CSV에는 있지만 파일 누락된 경우 제거)
    train_ids = _filter_existing_images(train_ids, image_dir, "train")
    val_ids = _filter_existing_images(val_ids, image_dir, "val")
    test_ids = _filter_existing_images(test_ids, image_dir, "test")

    # Create base datasets
    DatasetClass = SeverstalDetectionDataset if model_type == "detection" else SeverstalSegmentationDataset

    train_dataset = DatasetClass(
        image_dir=image_dir,
        annotation_csv=annotation_csv,
        image_ids=train_ids,
        input_size=input_size,
        transform=train_transform,
        is_training=True,
    )
    val_dataset = DatasetClass(
        image_dir=image_dir,
        annotation_csv=annotation_csv,
        image_ids=val_ids,
        input_size=input_size,
        transform=eval_transform,
        is_training=False,
    )
    test_dataset = DatasetClass(
        image_dir=image_dir,
        annotation_csv=annotation_csv,
        image_ids=test_ids,
        input_size=input_size,
        transform=eval_transform,
        is_training=False,
    )

    # Add CASDA synthetic data to training set if applicable
    casda_data = group_config.get('casda_data', None)
    if casda_data is not None:
        casda_config = ds_config.get('casda', {})
        # ratio 그룹의 _casda_max_samples 지원
        group_max_samples = group_config.get('_casda_max_samples', None)

        if casda_data == "full":
            raw_casda = casda_config.get('full_dir', 'data/augmented/casda_full')
            casda_dir = raw_casda if os.path.isabs(raw_casda) else str(project_root / raw_casda)
            casda_dataset = CASDASyntheticDataset(
                data_dir=casda_dir,
                mode=model_type,
                input_size=input_size,
                num_classes=ds_config.get('num_classes', 4),
                max_samples=group_max_samples,  # ratio 제한 (None이면 전체)
                transform=train_transform,
            )
        elif casda_data == "pruning":
            raw_casda = casda_config.get('pruning_dir', 'data/augmented/casda_pruning')
            casda_dir = raw_casda if os.path.isabs(raw_casda) else str(project_root / raw_casda)
            casda_dataset = CASDASyntheticDataset(
                data_dir=casda_dir,
                mode=model_type,
                input_size=input_size,
                num_classes=ds_config.get('num_classes', 4),
                    suitability_threshold=casda_config.get('suitability_threshold', 0.0),
                max_samples=group_max_samples or casda_config.get('pruning_top_k', 2000),
                transform=train_transform,
            )
        elif casda_data == "composed":
            # casda_dir_override: 그룹별 디렉토리 지정 (ablation_no_blending, copypaste 등)
            # run_benchmark.py의 inject 로직(line 1367-1376)과 동일한 resolution 적용
            dir_override = group_config.get('casda_dir_override', '')
            if dir_override:
                logging.info(f"[CASDA] casda_dir_override 적용: {dir_override}")
                if os.path.isabs(dir_override):
                    raw_casda = dir_override
                else:
                    # composed_dir의 부모를 CASDA base로 사용하여 상대경로 해석
                    # 예: composed_dir="/local/casda/casda_composed"
                    #   → base="/local/casda", override basename="copypaste_baseline"
                    #   → raw_casda="/local/casda/copypaste_baseline"
                    composed_dir = casda_config.get('composed_dir', 'data/augmented/casda_composed')
                    casda_base = str(Path(composed_dir).parent)
                    raw_casda = os.path.join(casda_base, os.path.basename(dir_override))
            else:
                raw_casda = casda_config.get('composed_dir', 'data/augmented/casda_composed')
            casda_dir = raw_casda if os.path.isabs(raw_casda) else str(project_root / raw_casda)
            # casda_pruning 설정 병합 (casda_composed_pruning 그룹용)
            pruning_cfg = group_config.get('casda_pruning', {})
            pruning_enabled = pruning_cfg.get('enabled', False)
            stratified = pruning_cfg.get('stratified', False)
            if pruning_enabled:
                casda_dataset = CASDASyntheticDataset(
                    data_dir=casda_dir,
                    mode=model_type,
                    input_size=input_size,
                    num_classes=ds_config.get('num_classes', 4),
                    suitability_threshold=pruning_cfg.get('suitability_threshold',
                                                          casda_config.get('suitability_threshold', 0.0)),
                    max_samples=group_max_samples or pruning_cfg.get('top_k',
                                                                     casda_config.get('pruning_top_k', 2000)),
                    transform=train_transform,
                    stratified=stratified,
                )
            else:
                casda_dataset = CASDASyntheticDataset(
                    data_dir=casda_dir,
                    mode=model_type,
                    input_size=input_size,
                    num_classes=ds_config.get('num_classes', 4),
                    max_samples=group_max_samples,
                    transform=train_transform,
                )
        else:
            casda_dataset = None

        if casda_dataset is not None and len(casda_dataset) > 0:
            train_dataset = ConcatDataset([train_dataset, casda_dataset])

    # Create data loaders
    def detection_collate_fn(batch):
        images = torch.stack([b['image'] for b in batch])
        labels = [b['labels'] for b in batch]
        image_ids = [b['image_id'] for b in batch]
        return {'image': images, 'labels': labels, 'image_id': image_ids}

    def segmentation_collate_fn(batch):
        images = torch.stack([b['image'] for b in batch])
        masks = torch.stack([b['mask'] for b in batch])
        image_ids = [b['image_id'] for b in batch]
        return {'image': images, 'mask': masks, 'image_id': image_ids}

    collate_fn = detection_collate_fn if model_type == "detection" else segmentation_collate_fn

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # split_info: split CSV 사용 시 실제 비율을 계산하여 반영
    _total = len(train_ids) + len(val_ids) + len(test_ids)
    if split_csv is not None and os.path.exists(split_csv) and _total > 0:
        # CSV 기반 split — 실제 개수에서 비율 역산
        _split_config = {
            'train_ratio': round(len(train_ids) / _total, 4),
            'val_ratio': round(len(val_ids) / _total, 4),
            'test_ratio': round(len(test_ids) / _total, 4),
            'seed': ds_config['split'].get('seed', 42),
            'source': split_csv,
        }
    else:
        # 동적 split — config 비율 그대로 사용
        _split_config = {
            'train_ratio': ds_config['split']['train_ratio'],
            'val_ratio': ds_config['split']['val_ratio'],
            'test_ratio': ds_config['split']['test_ratio'],
            'seed': ds_config['split']['seed'],
        }
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'split_config': _split_config,
        'num_train': len(train_ids),
        'num_val': len(val_ids),
        'num_test': len(test_ids),
    }

    return train_loader, val_loader, test_loader, split_info
