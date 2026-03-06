#!/usr/bin/env python3
"""
CASDA 벤치마크 공통 유틸리티 모듈.

scripts/run_benchmark.py 와 scripts/run_fid.py 양쪽에서 import하여 사용.
독립 실행 불필요 — 라이브러리 전용.
"""

import os
import sys
import logging
import random
import yaml
from pathlib import Path
from typing import Optional


# ============================================================================
# Project Root 설정
# ============================================================================

def setup_project_root() -> Path:
    """PROJECT_ROOT를 결정하고 sys.path에 추가한다.

    scripts/ 하위에서 호출되므로 parent.parent 로 프로젝트 루트를 결정.
    이미 sys.path에 있으면 중복 추가하지 않음.

    Returns:
        PROJECT_ROOT Path 객체
    """
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


# ============================================================================
# Config 로드
# ============================================================================

def load_config(config_path: str) -> dict:
    """YAML config를 로드한다.

    Args:
        config_path: YAML 파일 경로 (절대 또는 상대)

    Returns:
        config dict

    Raises:
        SystemExit: 파일이 존재하지 않을 때
    """
    p = Path(config_path)
    if not p.exists():
        print(f"Error: Config file not found: {p}")
        sys.exit(1)

    with open(p) as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# 작은 유틸리티 함수들 (FID + Benchmark 양쪽 공통)
# ============================================================================

def remove_empty_classes(
    real_by_class: dict,
    gen_by_class: dict,
) -> None:
    """real 또는 gen 이미지가 부족한 클래스를 양쪽 dict에서 제거 (in-place).

    FID per-class 계산 시, 한쪽이 2장 미만이면 FID 계산이 불가능하므로
    사전에 제거한다.
    """
    empty = [
        cid for cid in gen_by_class
        if len(real_by_class.get(cid, [])) < 2 or len(gen_by_class[cid]) < 2
    ]
    for cid in empty:
        logging.warning(
            f"  Class {cid + 1}: real={len(real_by_class.get(cid, []))}, "
            f"synthetic={len(gen_by_class.get(cid, []))} — FID 계산 건너뜀"
        )
        real_by_class.pop(cid, None)
        gen_by_class.pop(cid, None)


def sample_images(images: list, max_images: int, rng: random.Random) -> list:
    """max_images 이하로 seeded 랜덤 샘플링."""
    if len(images) > max_images:
        return rng.sample(images, max_images)
    return images
