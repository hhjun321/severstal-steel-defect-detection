"""
YOLO Format Dataset Converter for Ultralytics

Converts Severstal Steel Defect Detection data (CSV + images) into the
ultralytics YOLO directory format:

    yolo_dataset/
      images/
        train/  val/  test/
      labels/
        train/  val/  test/
      dataset.yaml

Each .txt label file contains one line per object:
    <class_id> <x_center> <y_center> <width> <height>
(all values normalized to [0,1])

Also supports adding CASDA synthetic data to the training set.
"""

import os
import shutil
import logging
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.rle_utils import rle_decode

logger = logging.getLogger(__name__)


def validate_yolo_dataset(yolo_dir: str, dataset_group: str = "") -> Optional[str]:
    """
    Validate an existing YOLO-format dataset directory.
    
    Checks:
      1. dataset.yaml exists and is readable
      2. images/{train,val,test} directories exist with images
      3. labels/{train,val,test} directories exist with label files
    
    Args:
        yolo_dir: Path to the YOLO dataset root (or parent containing per-group subdirs)
        dataset_group: If non-empty, look for yolo_dir/{dataset_group}/ subdirectory
    
    Returns:
        Path to dataset.yaml if valid, None otherwise
    """
    base = Path(yolo_dir)
    
    # If dataset_group specified, check for group subdirectory
    if dataset_group:
        group_dir = base / dataset_group
        if group_dir.exists():
            base = group_dir
    
    yaml_path = base / "dataset.yaml"
    if not yaml_path.exists():
        logger.debug(f"No dataset.yaml found at {yaml_path}")
        return None
    
    # Check directory structure
    required_splits = ['train', 'val', 'test']
    for split in required_splits:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            logger.warning(f"Missing images/{split} directory in {base}")
            return None
        if not lbl_dir.exists():
            logger.warning(f"Missing labels/{split} directory in {base}")
            return None
        
        # Check there are actual files
        img_count = sum(1 for _ in img_dir.iterdir()) if img_dir.exists() else 0
        if split in ('train', 'val') and img_count == 0:
            logger.warning(f"Empty images/{split} directory in {base}")
            return None
    
    # Read yaml to do basic sanity check
    try:
        import yaml
        with open(yaml_path) as f:
            ds_cfg = yaml.safe_load(f)
        if 'nc' not in ds_cfg or 'names' not in ds_cfg:
            logger.warning(f"dataset.yaml missing 'nc' or 'names' fields: {yaml_path}")
            return None
    except Exception as e:
        logger.warning(f"Failed to parse dataset.yaml: {e}")
        return None
    
    # Count stats for logging
    train_imgs = sum(1 for _ in (base / "images" / "train").iterdir())
    val_imgs = sum(1 for _ in (base / "images" / "val").iterdir())
    test_imgs = sum(1 for _ in (base / "images" / "test").iterdir())
    logger.info(f"Validated existing YOLO dataset at {base}")
    logger.info(f"  train: {train_imgs} images, val: {val_imgs} images, test: {test_imgs} images")
    logger.info(f"  nc: {ds_cfg['nc']}, names: {ds_cfg['names']}")
    
    # Update path field in dataset.yaml to match current location
    # (in case the dataset was moved)
    if ds_cfg.get('path') != base.as_posix():
        ds_cfg['path'] = base.as_posix()
        with open(yaml_path, 'w') as f:
            yaml.dump(ds_cfg, f, default_flow_style=False)
        logger.info(f"  Updated path field to: {base.as_posix()}")
    
    return str(yaml_path)


def _rle_to_bboxes(rle_string: str, shape: Tuple[int, int] = (256, 1600)) -> List[List[float]]:
    """
    Convert RLE mask to list of bounding boxes in normalized YOLO format.
    
    Returns:
        List of [x_center, y_center, width, height] normalized to [0,1]
    """
    mask = rle_decode(rle_string, shape)
    if mask.sum() == 0:
        return []

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    h, w = shape
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 16:  # skip tiny artifacts
            continue
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        bboxes.append([cx, cy, nw, nh])

    return bboxes


def prepare_yolo_dataset(
    image_dir: str,
    annotation_csv: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    output_dir: str,
    dataset_group: str = "baseline_raw",
    casda_dir: Optional[str] = None,
    casda_mode: Optional[str] = None,
    casda_config: Optional[Dict] = None,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Prepare a YOLO-format dataset directory for ultralytics training.
    
    This creates symlinks (or copies) of images and generates label .txt files.
    For CASDA groups, synthetic images are added to the training set.
    
    Args:
        image_dir: Path to Severstal train_images/
        annotation_csv: Path to train.csv
        train_ids, val_ids, test_ids: Image ID lists per split
        output_dir: Where to create the yolo_dataset/ structure
        dataset_group: Group name for logging
        casda_dir: Path to CASDA data directory (for casda_full/casda_pruning)
        casda_mode: "full" or "pruning" or None
        casda_config: CASDA config dict (threshold, top_k, etc.)
        num_classes: Number of defect classes
        class_names: Class name list
    
    Returns:
        Path to the generated dataset.yaml file
    """
    if class_names is None:
        class_names = [f"Class{i+1}" for i in range(num_classes)]

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    # Build annotation lookup: ImageId -> list of (class_id, bboxes)
    logger.info(f"Parsing annotations from {annotation_csv}")
    df = pd.read_csv(annotation_csv)
    if 'ImageId_ClassId' in df.columns:
        df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.rsplit('_', n=1, expand=True)
        df['ClassId'] = df['ClassId'].astype(int)

    # Group annotations by ImageId
    annotations = {}
    defect_rows = df[df['EncodedPixels'].notna()]
    for _, row in defect_rows.iterrows():
        img_id = row['ImageId']
        cls_id = int(row['ClassId']) - 1  # 0-indexed
        rle = row['EncodedPixels']
        bboxes = _rle_to_bboxes(rle)
        if img_id not in annotations:
            annotations[img_id] = []
        for bbox in bboxes:
            annotations[img_id].append((cls_id, bbox))

    # Process each split
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    stats = {}

    for split_name, ids in splits.items():
        img_split_dir = images_dir / split_name
        lbl_split_dir = labels_dir / split_name
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

        num_images = 0
        num_labels = 0

        for img_id in ids:
            src_path = Path(image_dir) / img_id
            if not src_path.exists():
                continue

            # Create symlink or copy for image
            dst_img = img_split_dir / img_id
            if not dst_img.exists():
                try:
                    os.symlink(src_path, dst_img)
                except (OSError, NotImplementedError):
                    shutil.copy2(str(src_path), str(dst_img))

            # Write label file
            label_name = Path(img_id).stem + ".txt"
            dst_lbl = lbl_split_dir / label_name

            annots = annotations.get(img_id, [])
            with open(dst_lbl, 'w') as f:
                for cls_id, bbox in annots:
                    cx, cy, w, h = bbox
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    num_labels += 1

            num_images += 1

        stats[split_name] = {'images': num_images, 'labels': num_labels}
        logger.info(f"  {split_name}: {num_images} images, {num_labels} labels")

    # Add CASDA synthetic data to training set
    casda_count = 0
    if casda_mode is not None and casda_dir is not None:
        casda_count = _add_casda_to_training(
            casda_dir=casda_dir,
            casda_mode=casda_mode,
            casda_config=casda_config or {},
            images_train_dir=images_dir / "train",
            labels_train_dir=labels_dir / "train",
            num_classes=num_classes,
        )
        logger.info(f"  CASDA ({casda_mode}): added {casda_count} synthetic images to training")

    # Generate dataset.yaml
    yaml_path = output_path / "dataset.yaml"
    yaml_content = (
        f"# YOLO dataset config for {dataset_group}\n"
        f"# Auto-generated by dataset_yolo.py\n"
        f"path: {output_path.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {num_classes}\n"
        f"names:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(class_names))
    )
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    logger.info(f"YOLO dataset prepared at {output_path}")
    logger.info(f"  dataset.yaml: {yaml_path}")
    total_train = stats['train']['images'] + casda_count
    logger.info(f"  Total training images: {total_train} "
                f"(original: {stats['train']['images']}, CASDA: {casda_count})")

    return str(yaml_path)


def _stratified_top_k_yolo(samples: List[Dict], k: int) -> List[Dict]:
    """
    클래스 비율을 유지하면서 suitability_score 상위 k개를 선택한다.
    YOLO 데이터 구조에 맞게 적용.
    
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

    # (b) 비율 기반 할당량 (최소 1개 보장)
    quotas: Dict[int, int] = {}
    for cls_id, cls_samples in groups.items():
        quota = max(1, round(len(cls_samples) / n_total * k))
        quota = min(quota, len(cls_samples))
        quotas[cls_id] = quota

    # (c) 할당량 합계가 k를 초과하면 큰 그룹부터 1씩 감소
    while sum(quotas.values()) > k:
        largest_cls = max(quotas, key=lambda c: quotas[c])
        if quotas[largest_cls] > 1:
            quotas[largest_cls] -= 1
        else:
            break

    # (d) 할당량 합계가 k 미만이면 여유 있는 그룹에 1씩 추가
    while sum(quotas.values()) < k:
        added = False
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


def _add_casda_to_training(
    casda_dir: str,
    casda_mode: str,
    casda_config: Dict,
    images_train_dir: Path,
    labels_train_dir: Path,
    num_classes: int,
) -> int:
    """
    Add CASDA synthetic images and labels to the YOLO training split.
    
    Supports the same metadata formats as CASDASyntheticDataset:
      - metadata.json
      - annotations.csv
      - Fallback: scan images/ for .png files, infer class from filename
    
    Bbox format support:
      - bbox_format="yolo": bboxes are [cx, cy, w, h] normalized (0~1).
        Written directly — no cv2.imread needed. (v5.1+ metadata)
      - bbox_format="xyxy" (or absent): bboxes are [x1, y1, x2, y2] pixel coords.
        Requires cv2.imread once per image for normalization.
      - mask_path fallback: derive bbox from mask contours via cv2.
      - No bbox, no mask: full-image bbox fallback.
    
    Returns:
        Number of synthetic images added
    """
    import json

    casda_path = Path(casda_dir)
    if not casda_path.exists():
        logger.warning(f"CASDA directory not found: {casda_dir}")
        return 0

    # Load metadata (same logic as CASDASyntheticDataset._load_metadata)
    meta_path = casda_path / "metadata.json"
    csv_path = casda_path / "annotations.csv"

    if meta_path.exists():
        with open(meta_path) as f:
            all_samples = json.load(f)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        all_samples = df.to_dict('records')
    else:
        # Fallback: scan image directory
        img_dir = casda_path / "images"
        if not img_dir.exists():
            img_dir = casda_path
        all_samples = []
        for img_path in sorted(img_dir.glob("*.png")):
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

    # Filter by suitability for pruning mode
    if casda_mode == "pruning":
        threshold = casda_config.get('suitability_threshold', 0.0)
        top_k = casda_config.get('pruning_top_k', 2000)
        stratified = casda_config.get('stratified', False)
        all_samples = [
            s for s in all_samples
            if s.get('suitability_score', 0.0) >= threshold
        ]
        all_samples.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
        if stratified:
            all_samples = _stratified_top_k_yolo(all_samples, top_k)
        else:
            all_samples = all_samples[:top_k]

    count = 0
    for idx, sample in enumerate(all_samples):
        # Resolve image path
        img_path = sample.get('image_path', '')
        if not os.path.isabs(img_path):
            img_path = str(casda_path / img_path)

        if not os.path.exists(img_path):
            continue

        # Create a unique filename to avoid collisions
        src = Path(img_path)
        dst_name = f"casda_{idx:05d}_{src.name}"
        dst_img = images_train_dir / dst_name

        if not dst_img.exists():
            try:
                os.symlink(src.resolve(), dst_img)
            except (OSError, NotImplementedError):
                shutil.copy2(str(src), str(dst_img))

        # Generate label file
        label_name = Path(dst_name).stem + ".txt"
        dst_lbl = labels_train_dir / label_name

        cls_id = sample.get('class_id', 0)
        bboxes = sample.get('bboxes', [])
        labels = sample.get('labels', [])
        bbox_format = sample.get('bbox_format', 'xyxy')

        with open(dst_lbl, 'w') as f:
            if bboxes and labels and bbox_format == 'yolo':
                # v5.1+: pre-computed YOLO normalized bboxes — write directly
                for bbox, lbl in zip(bboxes, labels):
                    cx, cy, bw, bh = bbox
                    f.write(f"{lbl} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            elif bboxes and labels:
                # Legacy: xyxy pixel coords — read image once for normalization
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    for bbox, lbl in zip(bboxes, labels):
                        x1, y1, x2, y2 = bbox
                        cx = ((x1 + x2) / 2.0) / w
                        cy = ((y1 + y2) / 2.0) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{lbl} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            elif 'mask_path' in sample:
                # Derive bbox from mask
                mask_path = sample['mask_path']
                if not os.path.isabs(mask_path):
                    mask_path = str(casda_path / mask_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    h, w = mask.shape[:2]
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for cnt in contours:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        if bw * bh >= 16:
                            cx = (bx + bw / 2.0) / w
                            cy = (by + bh / 2.0) / h
                            nw = bw / w
                            nh = bh / h
                            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                # Fallback: full-image bbox with class
                f.write(f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n")

        count += 1

    return count
