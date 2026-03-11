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
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --resume --output-dir outputs/benchmark_results/20260223_143000

FID Evaluation (분리됨):
  FID 평가는 scripts/run_fid.py로 분리되었습니다.
  python scripts/run_fid.py --config configs/benchmark_experiment.yaml
  (--fid-only 플래그는 deprecated — 위 명령어를 사용하세요)

Reference Results (이전 실험 결과 참조):
  Hypothesis Test에 필요한 비교 대상(예: baseline_raw)이 이번 실행에 포함되지 않을 때,
  이전 실험의 benchmark_results.json을 --reference-results로 지정하면
  해당 결과를 로드하여 Hypothesis Test에 활용합니다.
  현재 실행 결과와 중복되는 (model, dataset) 조합은 현재 실행 우선.
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml \\
      --groups casda_composed_pruning --models eb_yolov8 \\
      --reference-results /path/to/baseline_raw/benchmark_results.json
"""

import os
import sys
import argparse
import logging
import random
import subprocess
import shutil
import time
import json
import traceback
import threading
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
    # Ablation study & CopyPaste baseline
    "no_blend":   "ablation_no_blending",
    "no_blending": "ablation_no_blending",
    "no_pruning": "ablation_no_pruning",
    "copypaste":  "copypaste",
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
    stratified: bool = False,
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
        stratified: True이면 클래스별 균등 분배 pruning (stratified top-k)

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
            'stratified': stratified,
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
    expected_count: int = 0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> int:
    """
    baseline_raw YOLO 데이터셋에서 CASDA prefix 파일을 모두 삭제한다.

    images/train/casda_* 와 labels/train/casda_* 를 삭제.
    baseline 원본 파일은 prefix가 다르므로 절대 삭제되지 않음.

    A7 강화: 디렉토리 접근 불가 시 경고 + Drive 헬스체크 + 재시도.
    expected_count > 0 인데 삭제 수가 0이면 마운트 이상으로 판단하고 재시도.

    Args:
        baseline_dir: baseline_raw YOLO 데이터셋 경로
        prefix: 삭제할 파일 접두사
        expected_count: 기대 삭제 파일 수 (이미지+라벨 합계). 0이면 검증 생략.
        max_retries: 마운트 이상 시 재시도 횟수
        retry_delay: 재시도 간 대기 시간(초)

    Returns:
        삭제된 파일 수 (이미지 + 라벨 합계)
    """
    baseline_path = Path(baseline_dir)

    removed = 0
    for attempt in range(1, max_retries + 1):
        removed = 0
        dirs_missing = []

        for subdir in ["images/train", "labels/train"]:
            target_dir = baseline_path / subdir
            if not target_dir.exists():
                dirs_missing.append(str(target_dir))
                continue
            for f in target_dir.iterdir():
                if f.name.startswith(prefix):
                    try:
                        f.unlink(missing_ok=True)
                        removed += 1
                    except (PermissionError, OSError) as e:
                        logging.warning(f"[Clean] 파일 삭제 실패: {f.name}: {e}")

        # 디렉토리 자체가 없는 경우 — Drive 마운트 문제 가능성
        if dirs_missing:
            logging.warning(
                f"[Clean] 디렉토리 접근 불가 (시도 {attempt}/{max_retries}): "
                f"{dirs_missing}"
            )
            if attempt < max_retries:
                # Drive 헬스체크 후 재시도
                healthy = _check_drive_health(str(baseline_path))
                if not healthy:
                    logging.warning(
                        f"[Clean] Drive 마운트 비정상 — {retry_delay}초 후 재시도"
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    # Drive는 정상이지만 디렉토리가 실제로 없음 → 재시도 무의미
                    logging.warning(
                        "[Clean] Drive 정상이나 디렉토리 부재 — 실제로 존재하지 않는 경로"
                    )
                    break
            else:
                logging.error(
                    f"[Clean] {max_retries}회 시도 후에도 디렉토리 접근 불가"
                )
                break

        # 삭제 수와 기대값 비교
        if expected_count > 0 and removed == 0 and not dirs_missing:
            # 기대값이 있는데 삭제 0 → 마운트가 stale하여 iterdir()이 빈 결과일 가능성
            if attempt < max_retries:
                healthy = _check_drive_health(str(baseline_path))
                if not healthy:
                    logging.warning(
                        f"[Clean] 기대 삭제 수={expected_count}이나 실제 삭제=0, "
                        f"Drive 비정상 — {retry_delay}초 후 재시도 "
                        f"(시도 {attempt}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    continue
            # Drive 정상이거나 재시도 소진 → 실제로 파일이 없는 것으로 판단
            logging.warning(
                f"[Clean] 기대 삭제 수={expected_count}이나 실제 삭제=0 "
                f"— CASDA 파일이 이미 없거나 prefix 불일치"
            )
        elif expected_count > 0 and abs(removed - expected_count) > expected_count * 0.1:
            logging.warning(
                f"[Clean] 삭제 수 불일치: 기대={expected_count}, 실제={removed} "
                f"(차이 {abs(removed - expected_count)})"
            )

        # 성공적으로 완료
        break

    logging.info(f"  Cleaned {removed} CASDA files from {baseline_path}")
    return removed


# ============================================================================
# Drive Mount Health Check & Inject Validation  (A6 / A8)
# ============================================================================

def _check_drive_health(base_path: str, timeout: float = 10.0) -> bool:
    """
    Google Drive FUSE 마운트가 정상인지 확인한다.

    base_path 에 임시 파일을 write → read → delete 하여 마운트 활성 여부를 판단.
    Colab 환경이 아닌 경우 (/content/drive 가 아닌 경로) 에는 항상 True 를 반환.

    timeout 초 내에 I/O가 완료되지 않으면 마운트 비정상으로 간주 (FUSE hung 방어).

    Args:
        base_path: 확인할 디렉토리 경로 (예: yolo_dir)
        timeout:   write/read 를 기다리는 최대 초 (기본 10초)

    Returns:
        True 이면 마운트 정상, False 이면 비정상
    """
    bp = Path(base_path)

    # 로컬 환경 (Drive 마운트가 아닌 경우) 은 항상 통과
    if not str(bp).startswith("/content/drive"):
        return True

    # 부모 디렉토리 존재 여부 먼저 확인
    if not bp.exists():
        logging.warning(f"[DriveHealth] 경로 자체가 존재하지 않음: {bp}")
        return False

    health_file = bp / f".drive_health_{int(time.time())}"
    token = f"health_check_{time.time()}"
    result = [False]  # threading 에서 결과 반환용

    def _io_check():
        try:
            health_file.write_text(token, encoding="utf-8")
            readback = health_file.read_text(encoding="utf-8")
            if readback != token:
                logging.warning(
                    f"[DriveHealth] read-back 불일치: wrote={token!r}, read={readback!r}"
                )
                return
            result[0] = True
        except OSError as e:
            logging.warning(f"[DriveHealth] I/O 오류 — 마운트 비정상 가능: {e}")
        finally:
            try:
                health_file.unlink(missing_ok=True)
            except OSError:
                pass

    t = threading.Thread(target=_io_check, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logging.warning(
            f"[DriveHealth] I/O 작업이 {timeout}초 내에 완료되지 않음 — FUSE 마운트 hung 가능"
        )
        return False

    return result[0]


def _wait_for_drive(base_path: str, max_retries: int = 6, delay: float = 10.0) -> bool:
    """
    Drive 마운트가 복구될 때까지 대기한다.

    Args:
        base_path: 확인할 디렉토리 경로
        max_retries: 최대 재시도 횟수 (기본 6 → 최대 60초 대기)
        delay: 재시도 간 대기 시간(초)

    Returns:
        True 이면 복구됨, False 이면 max_retries 소진
    """
    for attempt in range(1, max_retries + 1):
        if _check_drive_health(base_path):
            if attempt > 1:
                logging.info(f"[DriveHealth] 마운트 복구 확인 (시도 {attempt}/{max_retries})")
            return True
        logging.warning(
            f"[DriveHealth] 마운트 비정상 — {delay}초 후 재시도 ({attempt}/{max_retries})"
        )
        time.sleep(delay)
    logging.error(f"[DriveHealth] {max_retries}회 재시도 후에도 마운트 복구 실패")
    return False


def _validate_injected_files(
    baseline_dir: str,
    inject_count: int,
    prefix: str = CASDA_PREFIX,
    sample_size: int = 100,
) -> tuple:
    """
    CASDA 주입 후 심링크/파일이 실제로 접근 가능한지 검증한다.

    1) images/train 과 labels/train 모두에서 prefix 파일 수를 inject_count 와 대조
    2) sample_size 만큼 랜덤 샘플링하여 접근 가능 여부 확인
    3) 깨진 심링크 발견 시 경고 로그 + 추정 전체 깨진 수 반환

    Args:
        baseline_dir: baseline_raw YOLO 데이터셋 경로
        inject_count: inject_casda_to_baseline() 이 반환한 주입 이미지 수
        prefix: CASDA 파일 접두사
        sample_size: 검증할 랜덤 샘플 수

    Returns:
        (total_found, estimated_broken)
        total_found: images/train 에서 prefix 매칭 파일 수
        estimated_broken: 전체 추정 깨진 파일 수 (샘플 비율 기반 추정)
    """
    baseline_path = Path(baseline_dir)

    # images/train 검증
    images_dir = baseline_path / "images" / "train"
    if not images_dir.exists():
        logging.error(f"[Validate] images/train 디렉토리 없음: {images_dir}")
        return (0, 0)

    casda_files = [f for f in images_dir.iterdir() if f.name.startswith(prefix)]
    total_found = len(casda_files)

    if total_found != inject_count:
        logging.warning(
            f"[Validate] CASDA 이미지 수 불일치: 주입={inject_count}, 발견={total_found}"
        )

    # labels/train 검증
    labels_dir = baseline_path / "labels" / "train"
    if labels_dir.exists():
        casda_labels = [f for f in labels_dir.iterdir() if f.name.startswith(prefix)]
        label_count = len(casda_labels)
        if label_count != inject_count:
            logging.warning(
                f"[Validate] CASDA 라벨 수 불일치: 주입={inject_count}, 발견={label_count}"
            )
    else:
        logging.warning(f"[Validate] labels/train 디렉토리 없음: {labels_dir}")

    if total_found == 0:
        return (0, 0)

    # 랜덤 샘플링으로 접근 가능 여부 확인
    rng = random.Random(42)
    sample = rng.sample(casda_files, min(sample_size, total_found))

    broken = 0
    broken_files = []
    for f in sample:
        # 심링크인 경우: 타겟 존재 여부 확인
        if f.is_symlink():
            target = f.resolve()
            if not target.exists():
                broken += 1
                broken_files.append(f.name)
        else:
            # 일반 파일: 읽기 가능 여부 확인 (os.access로 실제 읽기 테스트)
            if not os.access(str(f), os.R_OK):
                broken += 1
                broken_files.append(f.name)

    # 깨진 비율을 전체로 추정
    estimated_broken = int(broken / len(sample) * total_found) if broken > 0 else 0

    if broken > 0:
        logging.warning(
            f"[Validate] 샘플 {len(sample)}개 중 {broken}개 접근 불가: "
            f"{broken_files[:5]}{'...' if broken > 5 else ''}"
        )
        logging.warning(
            f"[Validate] 전체 추정 깨진 파일: ~{estimated_broken}/{total_found}"
        )
    else:
        logging.info(
            f"[Validate] CASDA 검증 통과: {total_found}개 이미지 발견, "
            f"샘플 {len(sample)}개 모두 정상"
        )

    return (total_found, estimated_broken)


# ============================================================================
# --local-dir: Drive → Local 데이터 자동 복사
# ============================================================================

# 그룹별 CASDA 하위 디렉토리 매핑 (smart copy에 사용)
_CASDA_GROUP_SUBDIRS = {
    "casda_full": "casda_full",
    "casda_pruning": "casda_pruning",
    "casda_composed": "casda_composed",
    "casda_composed_pruning": "casda_composed",
    "ablation_no_blending": "casda_no_blend",
    "ablation_no_pruning": "casda_composed",
    "copypaste": "copypaste_baseline",
}


def _resolve_needed_casda_subdirs(
    groups_cli: list,
    available_groups: list,
    config: dict,
) -> set:
    """선택된 그룹이 필요로 하는 CASDA 하위 디렉토리 집합을 반환.

    casda_dir_override가 있는 그룹은 해당 basename을,
    없으면 _CASDA_GROUP_SUBDIRS 매핑을 사용.
    """
    group_keys = resolve_groups(groups_cli, available_groups)
    needed = set()
    for gk in group_keys:
        group_cfg = config.get('dataset_groups', {}).get(gk, {})
        dir_override = group_cfg.get('casda_dir_override', '')
        if dir_override:
            needed.add(os.path.basename(dir_override))
        elif gk in _CASDA_GROUP_SUBDIRS:
            needed.add(_CASDA_GROUP_SUBDIRS[gk])
    return needed


def _zip_transfer(src: str, dst: str, label: str) -> bool:
    """디렉토리 또는 파일을 zip → cp → unzip 방식으로 로컬화.

    Drive FUSE의 파일별 I/O 오버헤드를 회피하기 위해:
      디렉토리: zip(Drive측) → cp(단일파일) → unzip(로컬측)
      파일:     shutil.copy2 (단일 파일은 zip 불필요)

    이미 로컬화 완료된 경우 (.localized 마커 존재) 건너뜀.
    Returns True if transfer was performed, False if skipped.
    """
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        logging.warning(f"[LocalDir] {label}: 소스 없음 → 건너뜀: {src}")
        return False

    # 마커 파일: 로컬화 완료 여부 확인 (resume 시 재전송 방지)
    if dst_path.is_dir():
        marker = dst_path / ".localized"
    else:
        marker = dst_path.parent / f".localized_{dst_path.name}"

    if marker.exists():
        logging.info(f"[LocalDir] {label}: 이미 로컬화됨 → 건너뜀: {dst}")
        return False

    t0 = time.time()

    if src_path.is_file():
        # 단일 파일: 직접 복사 (zip 불필요)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"[LocalDir] {label}: 파일 복사: {src} → {dst}")
        shutil.copy2(str(src_path), str(dst_path))
        marker.touch()

    elif src_path.is_dir():
        # 기존 불완전 전송 잔여 제거
        if dst_path.exists():
            logging.info(f"[LocalDir]   불완전 전송 잔여 제거: {dst}")
            shutil.rmtree(str(dst_path))

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # zip 임시 파일 경로 (dst 부모에 생성 → 로컬 디스크)
        zip_path = str(dst_path.parent / f"_transfer_{dst_path.name}.zip")

        # Phase 1: zip (Drive 측에서 순차 읽기)
        # -r 재귀, -q quiet, -0 저장만(압축 안함, 속도 우선)
        # 작업 디렉토리를 src 부모로 설정하여 zip 내부 경로를 상대경로로 유지
        src_parent = str(src_path.parent)
        src_name = src_path.name
        logging.info(f"[LocalDir] {label}: zip 생성 중: {src}")
        zip_result = subprocess.run(
            ["zip", "-r", "-q", "-0", zip_path, src_name],
            cwd=src_parent,
            capture_output=True, text=True,
        )
        if zip_result.returncode != 0:
            logging.error(f"[LocalDir] {label}: zip 실패 (rc={zip_result.returncode}): "
                          f"{zip_result.stderr.strip()}")
            # fallback: shutil.copytree
            logging.info(f"[LocalDir] {label}: fallback → shutil.copytree")
            shutil.copytree(str(src_path), str(dst_path))
            marker = dst_path / ".localized"
            marker.touch()
            elapsed = time.time() - t0
            logging.info(f"[LocalDir] {label}: fallback 복사 완료 ({elapsed:.1f}s)")
            return True

        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        elapsed_zip = time.time() - t0
        logging.info(f"[LocalDir] {label}: zip 완료 ({zip_size_mb:.1f}MB, {elapsed_zip:.1f}s)")

        # Phase 2: unzip (로컬 디스크 내 작업 → 매우 빠름)
        # -q quiet, -o overwrite, -d 출력 디렉토리
        t1 = time.time()
        logging.info(f"[LocalDir] {label}: unzip 중 → {dst_path.parent}")
        unzip_result = subprocess.run(
            ["unzip", "-q", "-o", zip_path, "-d", str(dst_path.parent)],
            capture_output=True, text=True,
        )
        if unzip_result.returncode != 0:
            logging.error(f"[LocalDir] {label}: unzip 실패 (rc={unzip_result.returncode}): "
                          f"{unzip_result.stderr.strip()}")
            # zip 잔여 정리
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False

        elapsed_unzip = time.time() - t1
        logging.info(f"[LocalDir] {label}: unzip 완료 ({elapsed_unzip:.1f}s)")

        # Phase 3: zip 임시 파일 삭제
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # unzip 결과: {dst_path.parent}/{src_name}/ → dst_path로 이름 맞추기
        unzipped = dst_path.parent / src_name
        if unzipped != dst_path and unzipped.exists():
            if dst_path.exists():
                shutil.rmtree(str(dst_path))
            unzipped.rename(dst_path)

        # 완료 마커 생성
        marker = dst_path / ".localized"
        marker.touch()

    elapsed = time.time() - t0
    logging.info(f"[LocalDir] {label}: 전송 완료 ({elapsed:.1f}s)")
    return True


def _localize_data(args, config) -> argparse.Namespace:
    """--local-dir 지정 시 학습 데이터를 zip/unzip으로 로컬 디스크에 전송하고 args 경로를 갱신.

    전송 방식:
      디렉토리: zip(Drive측, -0 무압축) → unzip(로컬측) — Drive FUSE 파일별 오버헤드 회피
      파일:     shutil.copy2 (단일 파일은 zip 불필요)

    전송 대상:
      --data-dir   → {local_dir}/images/
      --csv        → {local_dir}/train.csv
      --split-csv  → {local_dir}/split.csv
      --casda-dir  → {local_dir}/casda/{subdir}/ (필요한 그룹 하위만)
      --yolo-dir   → {local_dir}/yolo/

    전송 제외:
      --output-dir (결과/체크포인트는 Drive에 유지)

    Resume 지원:
      각 전송 대상에 .localized 마커를 남겨 재전송 방지.
    """
    local_base = Path(args.local_dir)
    local_base.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*70}")
    logging.info(f"[LocalDir] 데이터 로컬화 시작: {local_base}")
    logging.info(f"{'='*70}")
    t_start = time.time()

    # ── 1. 이미지 디렉토리 (--data-dir) ──
    if args.data_dir:
        local_images = str(local_base / "images")
        if _zip_transfer(args.data_dir, local_images, "images"):
            pass
        args.data_dir = local_images
        logging.info(f"[LocalDir] args.data_dir → {args.data_dir}")

    # ── 2. Annotation CSV (--csv) ──
    if args.csv:
        csv_name = os.path.basename(args.csv)
        local_csv = str(local_base / csv_name)
        _zip_transfer(args.csv, local_csv, "csv")
        args.csv = local_csv
        logging.info(f"[LocalDir] args.csv → {args.csv}")

    # ── 3. Split CSV (--split-csv) ──
    if getattr(args, 'split_csv', None):
        split_name = os.path.basename(args.split_csv)
        local_split = str(local_base / split_name)
        _zip_transfer(args.split_csv, local_split, "split_csv")
        args.split_csv = local_split
        logging.info(f"[LocalDir] args.split_csv → {args.split_csv}")

    # ── 4. CASDA 디렉토리 (--casda-dir) — 스마트 복사 ──
    if args.casda_dir:
        local_casda = str(local_base / "casda")
        Path(local_casda).mkdir(parents=True, exist_ok=True)

        # 선택된 그룹이 필요로 하는 하위 디렉토리만 복사
        available_groups = list(config.get('dataset_groups', {}).keys())
        needed_subdirs = _resolve_needed_casda_subdirs(
            args.groups, available_groups, config
        )

        if needed_subdirs:
            logging.info(f"[LocalDir] CASDA 스마트 복사: 필요 하위 디렉토리 = {sorted(needed_subdirs)}")
            for subdir in sorted(needed_subdirs):
                src_sub = os.path.join(args.casda_dir, subdir)
                dst_sub = os.path.join(local_casda, subdir)
                _zip_transfer(src_sub, dst_sub, f"casda/{subdir}")
        else:
            logging.info("[LocalDir] CASDA: 선택된 그룹 중 CASDA가 필요한 그룹 없음 → 건너뜀")

        # CASDA 상위 파일 (suitability CSV 등) 복사
        casda_src_path = Path(args.casda_dir)
        for f in casda_src_path.iterdir():
            if f.is_file() and f.suffix in ('.csv', '.json', '.yaml', '.yml'):
                dst_f = os.path.join(local_casda, f.name)
                _zip_transfer(str(f), dst_f, f"casda/{f.name}")

        args.casda_dir = local_casda
        logging.info(f"[LocalDir] args.casda_dir → {args.casda_dir}")

    # ── 5. YOLO 디렉토리 (--yolo-dir) ──
    if args.yolo_dir:
        local_yolo = str(local_base / "yolo")
        _zip_transfer(args.yolo_dir, local_yolo, "yolo")
        args.yolo_dir = local_yolo
        logging.info(f"[LocalDir] args.yolo_dir → {args.yolo_dir}")

    t_total = time.time() - t_start
    logging.info(f"\n[LocalDir] 데이터 로컬화 완료: {t_total:.1f}s")
    logging.info(f"{'='*70}\n")

    return args


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
# FID Evaluation — DEPRECATED: 분리된 scripts/run_fid.py 사용 권장
# ============================================================================

def _run_fid_deprecated_wrapper(config: dict, experiment_dir, device: str = 'cuda') -> dict:
    """[DEPRECATED] FID 평가를 scripts/run_fid.py에 위임.

    run_benchmark.py --fid-only 또는 내부 FID 호출 시 사용.
    실제 FID 로직은 scripts/run_fid.py로 완전 이전됨.
    """
    import warnings
    warnings.warn(
        "run_benchmark.py에서의 FID 호출은 deprecated입니다. "
        "python scripts/run_fid.py --config <config> 를 직접 사용하세요.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        from scripts.run_fid import run_fid_evaluation
        return run_fid_evaluation(config, experiment_dir, device)
    except ImportError:
        logging.error(
            "scripts/run_fid.py를 import할 수 없습니다. "
            "FID 평가를 건너뜁니다. "
            "독립 실행: python scripts/run_fid.py --config <config>"
        )
        return {}



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
                        help='[DEPRECATED] FID만 계산. scripts/run_fid.py를 직접 사용 권장.')
    parser.add_argument('--no-fid', action='store_true',
                        help='FID 평가를 건너뜁니다. (run_fid.py로 별도 실행 시 사용)')
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
    parser.add_argument('--casda-roi-dir', type=str, default=None,
                        help='ControlNet 생성 ROI 이미지 디렉토리 (generated/). '
                             'FID-ROI 계산 시 casda_full 대신 이 경로를 사용. '
                             '(overrides config dataset.casda.roi_dir). '
                             '예: /content/drive/.../augmented_images_v5.5/generated')
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
    parser.add_argument('--reference-results', type=str, nargs='+', default=None,
                        help='이전 실험의 benchmark_results.json 경로 (복수 지정 가능). '
                             '현재 실행에서 생성되지 않은 (model, dataset) 조합의 결과를 '
                             '로드하여 Hypothesis Test에 활용. '
                             '중복 조합은 현재 실행 결과 우선. '
                             '예: --reference-results /path/to/baseline_raw/benchmark_results.json')
    parser.add_argument('--local-dir', type=str, default=None,
                        help='로컬 디스크 경로 (예: /content/local_data). '
                             '지정 시 --data-dir, --csv, --casda-dir, --yolo-dir 데이터를 '
                             '이 경로로 자동 복사한 후 로컬에서 학습. '
                             'Google Drive I/O 병목 해소용. '
                             '--output-dir은 Drive에 유지됨 (결과 영구 보존). '
                             'Colab 예: --local-dir /content/local_data')
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

    # ====== --local-dir: 자동 데이터 로컬화 ======
    # Drive I/O 병목 해소를 위해 학습 데이터를 로컬 디스크로 복사.
    # --output-dir은 변경하지 않음 (결과/체크포인트는 Drive에 보존).
    if args.local_dir:
        args = _localize_data(args, config)

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

        # FID-ROI용 roi_metadata.csv 자동 탐색
        # casda_dir 상위에 roi_patches*/roi_metadata.csv가 있는지 확인
        fid_cfg = config.setdefault('evaluation', {}).setdefault('fid', {})
        if not fid_cfg.get('roi_metadata_csv'):
            casda_parent = Path(casda_base).parent
            for candidate_name in ['roi_patches', 'roi_patches_v5.1', 'roi_patches_v5']:
                candidate = casda_parent / candidate_name / 'roi_metadata.csv'
                if candidate.exists():
                    fid_cfg['roi_metadata_csv'] = str(candidate)
                    print(f"[INFO] roi_metadata_csv auto-detected: {candidate}")
                    break

    # Override split CSV if specified
    if args.split_csv:
        split_csv_path = os.path.abspath(args.split_csv)
        config['dataset']['split_csv'] = split_csv_path
        print(f"[INFO] split_csv overridden to: {split_csv_path}")

    # Override casda ROI dir if specified (FID-ROI용)
    if args.casda_roi_dir:
        if 'casda' not in config['dataset']:
            config['dataset']['casda'] = {}
        config['dataset']['casda']['roi_dir'] = args.casda_roi_dir
        print(f"[INFO] casda roi_dir overridden to: {args.casda_roi_dir}")

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

    # YAML casda_ratio 프리셋 지원: CLI --casda-ratio 미지정 시 YAML에서 읽기
    # CLI가 지정되면 YAML 프리셋은 무시됨 (CLI 우선)
    yaml_ratio_cfg = config.get('casda_ratio', {})
    if args.casda_ratio is None and yaml_ratio_cfg.get('enabled', False):
        args.casda_ratio = yaml_ratio_cfg.get('ratios', [])
        yaml_source = yaml_ratio_cfg.get('source', 'full')
        if getattr(args, 'casda_ratio_source', 'full') == 'full':
            # CLI --casda-ratio-source가 기본값이면 YAML 소스로 override
            args.casda_ratio_source = yaml_source
        yaml_include_baseline = yaml_ratio_cfg.get('include_baseline', True)
        logging.info(f"Using YAML casda_ratio presets: ratios={args.casda_ratio}, "
                     f"source={args.casda_ratio_source}, include_baseline={yaml_include_baseline}")

    if args.casda_ratio:
        # --groups가 없으면 baseline_raw + ratio 그룹만 실행
        # baseline_raw를 자동 포함하여 비교 기준선 확보
        if args.groups is None:
            include_baseline = yaml_ratio_cfg.get('include_baseline', True)
            if include_baseline and 'baseline_raw' in available_groups:
                group_keys = ['baseline_raw']
                logging.info("  Auto-including baseline_raw for ratio comparison")
            else:
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
                         f"(ratio={ratio:.0%}, max_samples={max_samples}, "
                         f"source={ratio_source_label}, pruning=suitability_top_k)")

    logging.info(f"Models: {model_keys}")
    logging.info(f"Dataset groups: {group_keys}")
    if args.groups:
        logging.info(f"  (resolved from CLI: {args.groups})")
    if casda_ratio_map:
        ratio_source_info = "CLI" if not yaml_ratio_cfg.get('enabled', False) else "YAML preset"
        logging.info(f"  (casda-ratio groups [{ratio_source_info}]: {list(casda_ratio_map.keys())})")
    logging.info(f"Total experiments: {len(model_keys) * len(group_keys)}")

    # Log training pipeline info
    for mk in model_keys:
        pipeline = "ultralytics" if mk in ULTRALYTICS_MODELS else "BenchmarkTrainer"
        logging.info(f"  {mk}: {pipeline}")

    # Initialize reporter
    reporter = BenchmarkReporter(str(experiment_dir))

    # ====== FID Evaluation (DEPRECATED — use scripts/run_fid.py) ======
    casda_groups_for_fid = {'casda_full', 'casda_pruning', 'casda_composed', 'casda_composed_pruning'}
    casda_groups_for_fid.update(casda_ratio_map.keys())
    has_casda = any(g in casda_groups_for_fid for g in group_keys)

    if args.fid_only:
        print("\n" + "=" * 70)
        print("[DEPRECATED] --fid-only 플래그는 deprecated입니다.")
        print("FID 평가는 독립 스크립트로 분리되었습니다:")
        print("  python scripts/run_fid.py --config <config> [--casda-dir ...] [--casda-roi-dir ...]")
        print("=" * 70 + "\n")
        logging.warning("--fid-only is deprecated. Use: python scripts/run_fid.py --config <config>")
        fid_results = _run_fid_deprecated_wrapper(config, experiment_dir, device)
        logging.info("FID-only mode complete (via deprecated wrapper).")
        return

    if args.no_fid:
        logging.info("Skipping FID evaluation (--no-fid flag)")
    elif has_casda and config.get('evaluation', {}).get('fid', {}).get('compute', True):
        logging.info("FID evaluation delegated to scripts/run_fid.py (deprecated wrapper)")
        fid_results = _run_fid_deprecated_wrapper(config, experiment_dir, device)
    elif not has_casda:
        logging.info("Skipping FID evaluation (no CASDA groups selected)")

    # ====== Run Experiments ======
    # Loop structure: group (outer) x model (inner)
    # For CASDA groups: inject → train all models → clean
    # This avoids duplicating the baseline_raw directory entirely.
    total_runs = len(model_keys) * len(group_keys)
    run_idx = 0
    start_time = time.time()

    CASDA_GROUPS = {
        "casda_full", "casda_pruning", "casda_composed", "casda_composed_pruning",
        # Ablation study & CopyPaste baseline
        "ablation_no_blending", "ablation_no_pruning", "copypaste",
    }
    CASDA_GROUP_TO_SUBDIR = {
        "casda_full": "casda_full",
        "casda_pruning": "casda_pruning",
        "casda_composed": "casda_composed",
        "casda_composed_pruning": "casda_composed",
        # Ablation & CopyPaste: casda_dir_override 사용 (아래 resolution 로직 참조)
        "ablation_no_blending": "casda_no_blend",
        "ablation_no_pruning": "casda_composed",
        "copypaste": "copypaste_baseline",
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

    # A8: 벤치마크 시작 전 Drive 마운트 헬스체크
    if baseline_yolo_dir:
        if not _check_drive_health(baseline_yolo_dir):
            logging.warning("[DriveHealth] 벤치마크 시작 전 Drive 마운트 비정상 감지")
            if not _wait_for_drive(baseline_yolo_dir):
                logging.error("[DriveHealth] Drive 복구 실패 — 벤치마크를 계속 진행하지만 "
                              "CASDA inject/clean이 실패할 수 있음")
            else:
                logging.info("[DriveHealth] Drive 마운트 정상 확인 — 벤치마크 시작")

    for group_key in group_keys:
        is_casda = group_key in CASDA_GROUPS
        casda_injected = False
        inject_count = 0

        # --- Inject CASDA if needed ---
        if is_casda and baseline_yolo_dir:
            # 사전 정리: 이전 실행 잔류 casda_* 파일 제거
            # (이전 실행이 중단되어 CLEAN 단계가 실행되지 않았을 수 있음)
            pre_removed = clean_casda_from_baseline(
                baseline_dir=baseline_yolo_dir, expected_count=0)
            if pre_removed > 0:
                logging.info(f"  Pre-clean: removed {pre_removed} stale CASDA files")

            casda_subdir = CASDA_GROUP_TO_SUBDIR[group_key]
            casda_data_dir = None

            # Resolve CASDA data directory from config
            # 우선순위: casda_dir_override (그룹 설정) > args.casda_dir > config casda
            group_cfg_for_dir = config.get('dataset_groups', {}).get(group_key, {})
            dir_override = group_cfg_for_dir.get('casda_dir_override', '')

            if dir_override:
                # casda_dir_override: 절대 경로 또는 args.casda_dir 기준 상대 경로
                if args.casda_dir and not os.path.isabs(dir_override):
                    casda_data_dir = os.path.join(args.casda_dir, os.path.basename(dir_override))
                else:
                    casda_data_dir = dir_override
            elif args.casda_dir:
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
                # pruning 파라미터 결정: ratio 그룹 → casda_pruning 그룹 → 전체
                ratio_max_samples = None
                suitability_thresh = None
                use_stratified = False

                if group_key in casda_ratio_map:
                    # ratio 그룹: suitability score 기준 정렬 후 비율 기반 top-k 선택
                    # (프로젝트 규칙: 증강 ratio별 모델 성능 비교는 pruning 데이터만 사용)
                    _, ratio_max_samples = casda_ratio_map[group_key]
                    suitability_thresh = 0.0
                else:
                    # 그룹 config에서 casda_pruning 설정 읽기
                    group_cfg = config.get('dataset_groups', {}).get(group_key, {})
                    pruning_cfg = group_cfg.get('casda_pruning', {})
                    if pruning_cfg.get('enabled', False):
                        ratio_max_samples = pruning_cfg.get('top_k', 2000)
                        suitability_thresh = pruning_cfg.get(
                            'suitability_threshold', 0.0)
                        use_stratified = pruning_cfg.get('stratified', False)

                logging.info(f"\n{'='*70}")
                logging.info(f"INJECT: {group_key} → baseline_raw")
                logging.info(f"  Source: {casda_data_dir}")
                logging.info(f"  Target: {baseline_yolo_dir}")
                if ratio_max_samples is not None:
                    mode_str = 'stratified' if use_stratified else 'global'
                    logging.info(
                        f"  Max samples: {ratio_max_samples} ({mode_str} top-k)")
                if suitability_thresh is not None:
                    logging.info(
                        f"  Suitability threshold: {suitability_thresh}")
                logging.info(f"{'='*70}")

                inject_count = inject_casda_to_baseline(
                    baseline_dir=baseline_yolo_dir,
                    casda_dir=casda_data_dir,
                    max_samples=ratio_max_samples,
                    suitability_threshold=suitability_thresh,
                    stratified=use_stratified,
                )
                casda_injected = True
                logging.info(f"  → {inject_count} images injected")

                # A6: 주입된 파일 검증
                if inject_count > 0:
                    total_found, broken = _validate_injected_files(
                        baseline_dir=baseline_yolo_dir,
                        inject_count=inject_count,
                    )
                    if broken > 0:
                        logging.warning(
                            f"[Validate] 깨진 심링크 {broken}개 발견 "
                            f"— Drive 마운트 상태를 확인하세요"
                        )
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

            # A8: 각 모델 학습 전 Drive 마운트 확인 (CASDA 주입 상태인 경우)
            if is_casda and casda_injected and baseline_yolo_dir:
                if not _check_drive_health(baseline_yolo_dir):
                    logging.warning(
                        f"[DriveHealth] {model_key} 학습 전 Drive 비정상 감지 — 복구 대기"
                    )
                    if not _wait_for_drive(baseline_yolo_dir):
                        logging.error(
                            f"[DriveHealth] Drive 복구 실패 — {model_key} 건너뜀"
                        )
                        continue

            try:
                # For CASDA groups with detection models:
                # Use baseline_raw (which now contains injected CASDA) as dataset_group
                # For segmentation models: use the actual group_key
                # (create_data_loaders handles CASDA via ConcatDataset internally)
                effective_group = group_key
                effective_yolo_dir = args.yolo_dir
                if is_casda and model_key in ULTRALYTICS_MODELS and casda_injected:
                    effective_group = "baseline_raw"

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
                logging.error(traceback.format_exc())
                continue

        # --- Clean CASDA after all models are done ---
        if casda_injected and baseline_yolo_dir:
            # A8: Clean 전 Drive 헬스체크
            if not _check_drive_health(baseline_yolo_dir):
                logging.warning("[DriveHealth] Clean 전 Drive 비정상 감지 — 복구 대기")
                if not _wait_for_drive(baseline_yolo_dir):
                    logging.error(
                        "[DriveHealth] Clean 전 Drive 복구 실패 — "
                        "CASDA 파일이 남아있을 수 있음. 수동 정리 필요."
                    )

            logging.info(f"\n{'='*70}")
            logging.info(f"CLEAN: Removing {group_key} files from baseline_raw")
            logging.info(f"{'='*70}")
            # A7: expected_count = inject_count(이미지) + inject_count(라벨) = 2배
            expected_clean = inject_count * 2
            removed = clean_casda_from_baseline(
                baseline_dir=baseline_yolo_dir,
                expected_count=expected_clean,
            )
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

    # ====== Reference Results 병합 (Hypothesis Test용) ======
    # 이전 실험의 benchmark_results.json을 로드하여 reporter.results에 추가.
    # 현재 실행에서 이미 생성된 (model, dataset) 조합은 중복 추가하지 않음.
    if args.reference_results:
        existing_keys = {
            (r['model'], r['dataset']) for r in reporter.results
        }
        ref_count = 0
        for ref_path_str in args.reference_results:
            ref_path = Path(ref_path_str)
            if not ref_path.exists():
                logging.warning(f"[Reference] 파일 없음, 건너뜀: {ref_path}")
                continue
            try:
                with open(ref_path) as f:
                    ref_data = json.load(f)
                if not isinstance(ref_data, list):
                    logging.warning(f"[Reference] 유효하지 않은 형식 (list 아님): {ref_path}")
                    continue
                for entry in ref_data:
                    key = (entry.get('model', ''), entry.get('dataset', ''))
                    if key in existing_keys:
                        logging.info(
                            f"[Reference] 중복 건너뜀: {key[0]} + {key[1]} "
                            f"(현재 실행 결과 우선)"
                        )
                        continue
                    reporter.results.append(entry)
                    existing_keys.add(key)
                    ref_count += 1
                    logging.info(
                        f"[Reference] 병합: {key[0]} + {key[1]} "
                        f"(from {ref_path.name})"
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"[Reference] JSON 파싱 실패: {ref_path} — {e}")
                continue
        if ref_count > 0:
            logging.info(
                f"[Reference] 총 {ref_count}개 이전 결과 병합 완료 "
                f"→ reporter.results: {len(reporter.results)}개"
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
