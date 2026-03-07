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


def _test_symlink_support(target_dir: Path) -> bool:
    """target_dir에서 symlink가 지원되는지 1회 테스트."""
    import tempfile
    test_src = target_dir / "__symlink_test_src__.tmp"
    test_dst = target_dir / "__symlink_test_dst__.tmp"
    try:
        test_src.write_text("test")
        os.symlink(test_src.resolve(), test_dst)
        result = test_dst.exists()
        return result
    except (OSError, NotImplementedError):
        return False
    finally:
        for p in (test_dst, test_src):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _generate_label_content(
    sample: Dict,
    img_path: str,
    casda_path: Path,
) -> str:
    """
    샘플 메타데이터로부터 YOLO 라벨 텍스트를 생성한다.
    파일 I/O 없이 문자열만 반환 (로컬 스테이징에 사용).
    
    bbox_format="yolo"이면 cv2 없이 바로 생성.
    legacy format이면 cv2.imread가 필요할 수 있음.
    """
    cls_id = sample.get('class_id', 0)
    bboxes = sample.get('bboxes', [])
    labels = sample.get('labels', [])
    bbox_format = sample.get('bbox_format', 'xyxy')

    lines: List[str] = []

    if bboxes and labels and bbox_format == 'yolo':
        # v5.1+: pre-computed YOLO normalized bboxes — 바로 기록
        for bbox, lbl in zip(bboxes, labels):
            cx, cy, bw, bh = bbox
            lines.append(f"{lbl} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    elif bboxes and labels:
        # Legacy: xyxy pixel coords — 이미지 읽어서 정규화
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            for bbox, lbl in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw_n = (x2 - x1) / w
                bh_n = (y2 - y1) / h
                lines.append(f"{lbl} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
    elif 'mask_path' in sample:
        # mask에서 bbox 유도
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
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    else:
        # Fallback: 전체 이미지 bbox
        lines.append(f"{cls_id} 0.500000 0.500000 1.000000 1.000000")

    return "\n".join(lines) + "\n" if lines else "\n"


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
    
    최적화 (v5.6):
      - symlink 지원 여부를 1회 사전 테스트하여 매번 예외 발생 방지
      - 라벨 .txt 파일을 로컬 tmpdir에 먼저 일괄 생성 후 대상 디렉토리에 복사
        (Google Drive FUSE 등 느린 파일시스템에서 per-file 메타데이터 오버헤드 감소)
      - 500건마다 progress 로그 출력
    
    Returns:
        Number of synthetic images added
    """
    import json
    import tempfile
    import time

    casda_path = Path(casda_dir)
    if not casda_path.exists():
        logger.warning(f"CASDA directory not found: {casda_dir}")
        return 0

    # ── 1. 메타데이터 로드 ──
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

    # ── 2. Pruning 필터링 ──
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

    total = len(all_samples)
    if total == 0:
        logger.warning("No CASDA samples to inject after filtering.")
        return 0

    logger.info(f"  Injecting {total} CASDA images...")

    # ── 3. symlink 지원 여부 사전 탐지 ──
    use_symlink = _test_symlink_support(images_train_dir)
    link_mode = "symlink" if use_symlink else "copy"
    logger.info(f"  Image link mode: {link_mode}")

    # ── 4. 라벨을 로컬 tmpdir에 일괄 생성 ──
    # 로컬 디스크에서 작은 .txt 파일을 빠르게 생성한 후 대상 디렉토리에 일괄 복사.
    # Google Drive FUSE 등에서 per-file open/write/close 오버헤드를 크게 줄인다.
    t0 = time.time()
    label_staging: List[Tuple[str, str]] = []  # (label_filename, label_content)
    img_tasks: List[Tuple[str, str]] = []      # (src_path, dst_name)
    skipped = 0

    for idx, sample in enumerate(all_samples):
        # 이미지 경로 resolve
        img_path = sample.get('image_path', '')
        if not os.path.isabs(img_path):
            img_path = str(casda_path / img_path)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # 파일명 생성
        src = Path(img_path)
        dst_name = f"casda_{idx:05d}_{src.name}"

        # 라벨 내용 생성 (메모리에서)
        label_content = _generate_label_content(sample, img_path, casda_path)
        label_name = Path(dst_name).stem + ".txt"
        label_staging.append((label_name, label_content))

        # 이미지 복사 작업 등록
        img_tasks.append((img_path, dst_name))

    if skipped > 0:
        logger.warning(f"  {skipped} images not found, skipped.")

    # ── 5. 로컬 tmpdir에 라벨 파일 일괄 쓰기 → 대상 디렉토리에 복사 ──
    with tempfile.TemporaryDirectory(prefix="casda_labels_") as tmpdir:
        tmp_path = Path(tmpdir)
        for label_name, label_content in label_staging:
            (tmp_path / label_name).write_text(label_content)

        # 일괄 복사: tmpdir → labels_train_dir
        for label_name, _ in label_staging:
            src_lbl = tmp_path / label_name
            dst_lbl = labels_train_dir / label_name
            shutil.copy2(str(src_lbl), str(dst_lbl))

    label_elapsed = time.time() - t0
    logger.info(f"  Labels: {len(label_staging)} files written ({label_elapsed:.1f}s)")

    # ── 6. 이미지 symlink/copy + progress 로그 ──
    t1 = time.time()
    count = 0
    log_interval = max(1, total // 4)  # 25% 간격 로그 (최소 500건마다)
    if log_interval > 500:
        log_interval = 500

    for src_path, dst_name in img_tasks:
        dst_img = images_train_dir / dst_name
        if not dst_img.exists():
            if use_symlink:
                os.symlink(Path(src_path).resolve(), dst_img)
            else:
                shutil.copy2(src_path, str(dst_img))
        count += 1

        if count % log_interval == 0:
            elapsed = time.time() - t1
            rate = count / elapsed if elapsed > 0 else 0
            logger.info(f"  Progress: {count}/{len(img_tasks)} images "
                        f"({count * 100 // len(img_tasks)}%, "
                        f"{rate:.0f} img/s, {elapsed:.1f}s)")

    img_elapsed = time.time() - t1
    total_elapsed = time.time() - t0
    logger.info(f"  Images: {count} files ({link_mode}, {img_elapsed:.1f}s)")
    logger.info(f"  Total inject time: {total_elapsed:.1f}s")

    return count
