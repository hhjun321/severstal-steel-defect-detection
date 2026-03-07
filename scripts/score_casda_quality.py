#!/usr/bin/env python3
"""
CASDA Composed 이미지 후행(Post-hoc) 품질 평가 스크립트

compose_casda_images.py에서 --quality-json 없이 합성한 경우
모든 suitability_score가 기본값(0.5)으로 설정된다.
이 스크립트는 이미 합성된 1600×256 이미지에 대해 3-메트릭 품질 점수를
계산하여 metadata.json의 suitability_score를 실제 값으로 업데이트한다.

3-메트릭 (run_validation_phases.py에서 재사용):
  - color_score  (0.40): LAB 색공간 기반 금속 표면 색상 일관성
  - artifact_score (0.30): Gradient 기반 아티팩트 감지
  - blur_score   (0.30): Laplacian + Gradient 기반 선명도

Usage:
  # 기본 사용
  python scripts/score_casda_quality.py \
      --casda-dir /path/to/casda_composed

  # 출력 디렉토리 지정 (원본 metadata.json 보존)
  python scripts/score_casda_quality.py \
      --casda-dir /path/to/casda_composed \
      --output-dir /path/to/output

  # 커스텀 가중치 + 배치 크기
  python scripts/score_casda_quality.py \
      --casda-dir /path/to/casda_composed \
      --weight-color 0.40 --weight-artifact 0.30 --weight-blur 0.30 \
      --batch-size 200

  # 병렬 처리 (8 워커)
  python scripts/score_casda_quality.py \
      --casda-dir /path/to/casda_composed \
      --workers 8

  # 자동 워커 수 감지
  python scripts/score_casda_quality.py \
      --casda-dir /path/to/casda_composed \
      --workers -1

Dependencies:
  cv2, numpy (torch/lpips 불필요)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 프로젝트 루트를 sys.path에 추가 (Colab 호환)
# ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# 3-메트릭 품질 평가 함수
# run_validation_phases.py의 함수를 재사용한다.
# import가 실패할 경우 (독립 실행 시) 인라인 폴백을 사용한다.
# ======================================================================

def _import_scoring_functions():
    """run_validation_phases.py에서 3-메트릭 함수를 import 시도."""
    try:
        from scripts.run_validation_phases import (
            _score_color_consistency,
            _score_artifacts,
            _score_sharpness,
        )
        return _score_color_consistency, _score_artifacts, _score_sharpness
    except ImportError:
        pass

    # 직접 경로로 재시도
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_validation_phases",
            str(_SCRIPT_DIR / "run_validation_phases.py"),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return (
                getattr(mod, '_score_color_consistency'),
                getattr(mod, '_score_artifacts'),
                getattr(mod, '_score_sharpness'),
            )
    except Exception:
        pass

    return None, None, None


# ──────────────────────────────────────────────────────────────────────
# 인라인 폴백 구현 (run_validation_phases.py import 실패 시)
# 원본 함수와 동일한 로직이지만 독립 실행을 보장한다.
# ──────────────────────────────────────────────────────────────────────

def _score_color_consistency_fallback(img_rgb: np.ndarray) -> float:
    """LAB 색공간 기반 금속 표면 색상 일관성 점수. (0.0~1.0)"""
    import cv2

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # 1. 채도 (a, b 채널 편차)
    a_ch = lab[:, :, 1].astype(float) - 128.0
    b_ch = lab[:, :, 2].astype(float) - 128.0
    ab_abs_mean = (np.mean(np.abs(a_ch)) + np.mean(np.abs(b_ch))) / 2.0
    if ab_abs_mean <= 3.0:
        chroma_score = 1.0
    elif ab_abs_mean >= 25.0:
        chroma_score = 0.0
    else:
        chroma_score = 1.0 - (ab_abs_mean - 3.0) / 22.0

    # 2. 밝기 dynamic range (L 채널 5th-95th percentile)
    l_ch = lab[:, :, 0].astype(float)
    l_range = float(np.percentile(l_ch, 95) - np.percentile(l_ch, 5))
    if 30 <= l_range <= 150:
        range_score = 1.0
    elif l_range < 15:
        range_score = 0.2
    elif l_range < 30:
        range_score = 0.2 + 0.8 * (l_range - 15) / 15.0
    else:
        range_score = max(0.3, 1.0 - (l_range - 150) / 100.0)

    # 3. 히스토그램 엔트로피
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    if entropy >= 4.5:
        entropy_score = 1.0
    elif entropy <= 2.0:
        entropy_score = 0.1
    else:
        entropy_score = 0.1 + 0.9 * (entropy - 2.0) / 2.5

    # 4. 텍스처 복잡도
    l_std = float(np.std(l_ch))
    if 8 <= l_std <= 50:
        texture_score = 1.0
    elif l_std < 3:
        texture_score = 0.1
    elif l_std < 8:
        texture_score = 0.1 + 0.9 * (l_std - 3) / 5.0
    else:
        texture_score = max(0.3, 1.0 - (l_std - 50) / 30.0)

    return float(0.20 * chroma_score + 0.25 * range_score
                 + 0.30 * entropy_score + 0.25 * texture_score)


def _score_artifacts_fallback(gray: np.ndarray) -> float:
    """Gradient 분석 기반 아티팩트 감지 점수. (0.0~1.0, 1.0=아티팩트 없음)"""
    import cv2

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    if grad_std < 1e-6:
        outlier_ratio = 0.0
    else:
        outlier_ratio = float(
            np.sum(grad_mag > grad_mean + 3.0 * grad_std) / grad_mag.size
        )

    if outlier_ratio <= 0.01:
        edge_score = 1.0
    elif outlier_ratio >= 0.05:
        edge_score = 0.0
    else:
        edge_score = 1.0 - (outlier_ratio - 0.01) / 0.04

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_energy = float(np.mean(np.abs(laplacian)))
    hf_ratio = lap_energy / (grad_mean + 1e-6)

    if hf_ratio <= 2.5:
        hf_score = 1.0
    elif hf_ratio >= 5.0:
        hf_score = 0.0
    else:
        hf_score = 1.0 - (hf_ratio - 2.5) / 2.5

    return float(0.6 * edge_score + 0.4 * hf_score)


def _score_sharpness_fallback(gray: np.ndarray) -> float:
    """Laplacian + Gradient 기반 선명도 점수. (0.0~1.0, 1.0=선명)"""
    import cv2

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var >= 500:
        lap_score = 1.0
    elif lap_var <= 30:
        lap_score = 0.1
    else:
        lap_score = 0.1 + 0.9 * (lap_var - 30) / 470.0

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    p50 = float(np.percentile(gradient, 50))
    p90 = float(np.percentile(gradient, 90))

    if p50 < 1e-6:
        edge_sharpness = 0.1
    else:
        cr = p90 / p50
        if cr >= 4.0:
            edge_sharpness = 1.0
        elif cr <= 1.5:
            edge_sharpness = 0.1
        else:
            edge_sharpness = 0.1 + 0.9 * (cr - 1.5) / 2.5

    return float(0.5 * lap_score + 0.5 * edge_sharpness)


# ======================================================================
# 병렬 워커 함수 (ProcessPoolExecutor용 — 모듈 레벨 정의 필수)
# ======================================================================

def _score_single_image(args_tuple: Tuple) -> Optional[Dict[str, Any]]:
    """
    단일 이미지의 3-메트릭 품질 점수를 계산하는 워커 함수.

    ProcessPoolExecutor에서 pickle 직렬화가 가능하도록
    모듈 레벨에 정의하고, 인자를 tuple로 받는다.

    Args:
        args_tuple: (index, img_path_str, use_fallback)
            - index: 원본 metadata 리스트에서의 인덱스
            - img_path_str: 이미지 절대 경로
            - use_fallback: True이면 인라인 폴백 함수 사용

    Returns:
        성공 시: {"index": int, "color": float, "artifact": float,
                  "blur": float} dict
        실패 시: None
    """
    import cv2

    idx, img_path_str, use_fallback = args_tuple

    img_bgr = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 폴백 함수는 모듈 레벨이므로 항상 접근 가능
    # (import된 함수는 pickle 불가할 수 있으므로 워커에서는 항상 폴백 사용)
    color_s = _score_color_consistency_fallback(img_rgb)
    artifact_s = _score_artifacts_fallback(gray)
    blur_s = _score_sharpness_fallback(gray)

    return {
        "index": idx,
        "color": color_s,
        "artifact": artifact_s,
        "blur": blur_s,
    }


# ======================================================================
# 메인 로직
# ======================================================================

def resolve_scoring_functions():
    """3-메트릭 함수를 가져온다. import 우선, 실패 시 인라인 폴백."""
    fn_color, fn_artifact, fn_blur = _import_scoring_functions()
    if fn_color is not None:
        logger.info("3-메트릭 함수: run_validation_phases.py에서 import 성공")
        return fn_color, fn_artifact, fn_blur
    logger.info("3-메트릭 함수: 인라인 폴백 사용 (동일 로직)")
    return (
        _score_color_consistency_fallback,
        _score_artifacts_fallback,
        _score_sharpness_fallback,
    )


def load_metadata(casda_dir: Path) -> List[Dict[str, Any]]:
    """casda_composed/metadata.json을 로드한다."""
    meta_path = casda_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"metadata.json must be a JSON array, got {type(data).__name__}")

    logger.info(f"metadata.json 로드 완료: {len(data)} entries")
    return data


def score_images(
    metadata: List[Dict[str, Any]],
    casda_dir: Path,
    weight_color: float = 0.40,
    weight_artifact: float = 0.30,
    weight_blur: float = 0.30,
    batch_log_interval: int = 500,
    num_workers: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    각 이미지에 대해 3-메트릭 품질 점수를 계산하고 metadata를 업데이트한다.

    Args:
        num_workers: 병렬 워커 수 (0 = 순차 처리, >=2 = ProcessPoolExecutor 사용)

    Returns:
        (updated_metadata, stats_dict)
    """
    import cv2

    fn_color, fn_artifact, fn_blur = resolve_scoring_functions()

    images_dir = casda_dir / "images"
    if not images_dir.exists():
        # 일부 구조에서는 바로 casda_dir에 이미지가 있을 수 있음
        images_dir = casda_dir

    updated = copy.deepcopy(metadata)
    scores_all: List[float] = []
    scores_by_class: Dict[int, List[float]] = {}
    detail_scores: List[Dict[str, float]] = []
    skipped = 0
    t0 = time.time()

    total = len(updated)
    use_parallel = num_workers > 1 and total > 10
    logger.info(f"품질 평가 시작: {total}장, 가중치=(color={weight_color}, "
                f"artifact={weight_artifact}, blur={weight_blur}), "
                f"workers={'sequential' if not use_parallel else num_workers}")

    # ── 1단계: 이미지 경로 결정 (순차 — metadata 접근 필요) ──
    # 각 entry에 대해 이미지 경로를 결정하고, 유효한 경로만 작업 목록에 추가
    tasks: List[Tuple[int, str]] = []  # (index_in_updated, abs_path)
    for i, entry in enumerate(updated):
        img_path_str = entry.get("image_path", "")
        if not img_path_str:
            img_path_str = entry.get("composed_path", "")
        if not img_path_str:
            fname = entry.get("filename", entry.get("image_filename", ""))
            if fname:
                img_path_str = str(images_dir / fname)

        if not img_path_str:
            skipped += 1
            continue

        # 절대경로가 아니면 casda_dir 기준으로 해석
        if not os.path.isabs(img_path_str):
            img_path_str = str(casda_dir / img_path_str)

        if not os.path.exists(img_path_str):
            skipped += 1
            continue

        tasks.append((i, img_path_str))

    logger.info(f"  유효 이미지: {len(tasks)}장, 스킵(경로 없음/미존재): {skipped}장")

    # ── 2단계: 3-메트릭 계산 (병렬 또는 순차) ──
    # 결과를 {index: {color, artifact, blur}} dict로 수집
    results_map: Dict[int, Dict[str, float]] = {}

    if use_parallel:
        # ── 병렬 경로: ProcessPoolExecutor ──
        logger.info(f"  병렬 모드: {num_workers} workers, {len(tasks)} tasks")
        worker_args = [(idx, path, True) for idx, path in tasks]

        scored_count = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_score_single_image, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(futures):
                result = future.result()
                scored_count += 1
                if result is not None:
                    results_map[result["index"]] = result
                else:
                    skipped += 1

                # 진행률 로그
                if scored_count % batch_log_interval == 0 or scored_count == len(tasks):
                    elapsed = time.time() - t0
                    rate = scored_count / elapsed if elapsed > 0 else 0
                    logger.info(f"  [{scored_count}/{len(tasks)}] {rate:.1f} img/s, "
                                f"elapsed={elapsed:.1f}s")
    else:
        # ── 순차 경로 ──
        for task_i, (idx, img_path_str) in enumerate(tasks):
            img_bgr = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
            if img_bgr is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            color_s = fn_color(img_rgb)
            artifact_s = fn_artifact(gray)
            blur_s = fn_blur(gray)

            results_map[idx] = {
                "index": idx,
                "color": color_s,
                "artifact": artifact_s,
                "blur": blur_s,
            }

            # 진행률 로그
            done = task_i + 1
            if done % batch_log_interval == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                logger.info(f"  [{done}/{len(tasks)}] {rate:.1f} img/s, "
                            f"elapsed={elapsed:.1f}s")

    # ── 3단계: 결과 집계 + metadata 업데이트 ──
    for idx, result in results_map.items():
        entry = updated[idx]
        color_s = result["color"]
        artifact_s = result["artifact"]
        blur_s = result["blur"]

        quality = (weight_color * color_s
                   + weight_artifact * artifact_s
                   + weight_blur * blur_s)

        entry["suitability_score"] = round(quality, 6)
        entry["quality_detail"] = {
            "color_score": round(color_s, 6),
            "artifact_score": round(artifact_s, 6),
            "blur_score": round(blur_s, 6),
        }

        scores_all.append(quality)
        class_id = entry.get("class_id", -1)
        scores_by_class.setdefault(class_id, []).append(quality)
        detail_scores.append({
            "color": color_s,
            "artifact": artifact_s,
            "blur": blur_s,
            "quality": quality,
        })

    # 통계 계산
    elapsed_total = time.time() - t0
    scores_arr = np.array(scores_all) if scores_all else np.array([0.0])

    stats: Dict[str, Any] = {
        "total_images": total,
        "scored_images": len(scores_all),
        "skipped_images": skipped,
        "elapsed_seconds": round(elapsed_total, 1),
        "weights": {
            "color": weight_color,
            "artifact": weight_artifact,
            "blur": weight_blur,
        },
        "overall": {
            "mean": round(float(np.mean(scores_arr)), 6),
            "std": round(float(np.std(scores_arr)), 6),
            "min": round(float(np.min(scores_arr)), 6),
            "max": round(float(np.max(scores_arr)), 6),
            "p10": round(float(np.percentile(scores_arr, 10)), 6),
            "p25": round(float(np.percentile(scores_arr, 25)), 6),
            "p50": round(float(np.percentile(scores_arr, 50)), 6),
            "p75": round(float(np.percentile(scores_arr, 75)), 6),
            "p90": round(float(np.percentile(scores_arr, 90)), 6),
        },
        "by_class": {},
    }

    for cls_id in sorted(scores_by_class.keys()):
        cls_scores = np.array(scores_by_class[cls_id])
        stats["by_class"][f"class{cls_id + 1}"] = {
            "count": len(cls_scores),
            "mean": round(float(np.mean(cls_scores)), 6),
            "std": round(float(np.std(cls_scores)), 6),
            "min": round(float(np.min(cls_scores)), 6),
            "max": round(float(np.max(cls_scores)), 6),
            "p25": round(float(np.percentile(cls_scores, 25)), 6),
            "p50": round(float(np.percentile(cls_scores, 50)), 6),
            "p75": round(float(np.percentile(cls_scores, 75)), 6),
        }

    # 서브메트릭 통계
    if detail_scores:
        for metric_name in ["color", "artifact", "blur"]:
            vals = np.array([d[metric_name] for d in detail_scores])
            stats[f"{metric_name}_stats"] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
            }

    return updated, stats


def save_results(
    updated_metadata: List[Dict[str, Any]],
    stats: Dict[str, Any],
    casda_dir: Path,
    output_dir: Optional[Path],
    backup: bool = True,
) -> Tuple[Path, Path]:
    """업데이트된 metadata.json과 통계를 저장한다."""
    if output_dir is None:
        output_dir = casda_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # metadata.json 저장
    meta_out = output_dir / "metadata.json"
    meta_orig = casda_dir / "metadata.json"

    # 원본 백업 (in-place 업데이트 시)
    if backup and meta_orig.exists() and output_dir == casda_dir:
        backup_path = casda_dir / "metadata.json.bak"
        if not backup_path.exists():
            shutil.copy2(str(meta_orig), str(backup_path))
            logger.info(f"원본 백업: {backup_path}")

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"metadata.json 저장: {meta_out}")

    # 통계 저장
    stats_out = output_dir / "quality_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"quality_stats.json 저장: {stats_out}")

    return meta_out, stats_out


def print_summary(stats: Dict[str, Any]) -> None:
    """통계 요약을 로그에 출력한다."""
    overall = stats["overall"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("품질 평가 완료")
    logger.info("=" * 60)
    logger.info(f"  평가: {stats['scored_images']}/{stats['total_images']}장 "
                f"(스킵: {stats['skipped_images']})")
    logger.info(f"  소요: {stats['elapsed_seconds']}초")
    logger.info(f"  가중치: color={stats['weights']['color']}, "
                f"artifact={stats['weights']['artifact']}, "
                f"blur={stats['weights']['blur']}")
    logger.info("")
    logger.info(f"  전체 suitability_score:")
    logger.info(f"    mean={overall['mean']:.4f}, std={overall['std']:.4f}")
    logger.info(f"    min={overall['min']:.4f}, max={overall['max']:.4f}")
    logger.info(f"    P10={overall['p10']:.4f}, P25={overall['p25']:.4f}, "
                f"P50={overall['p50']:.4f}, P75={overall['p75']:.4f}, "
                f"P90={overall['p90']:.4f}")

    # 서브메트릭 통계
    for metric_name in ["color", "artifact", "blur"]:
        key = f"{metric_name}_stats"
        if key in stats:
            ms = stats[key]
            logger.info(f"    {metric_name}: mean={ms['mean']:.4f}, "
                        f"std={ms['std']:.4f}, "
                        f"range=[{ms['min']:.4f}, {ms['max']:.4f}]")

    logger.info("")
    logger.info("  클래스별 suitability_score:")
    for cls_key, cls_stats in stats.get("by_class", {}).items():
        logger.info(f"    {cls_key} (n={cls_stats['count']}): "
                    f"mean={cls_stats['mean']:.4f}, "
                    f"P25={cls_stats['p25']:.4f}, "
                    f"P50={cls_stats['p50']:.4f}, "
                    f"P75={cls_stats['p75']:.4f}")

    # pruning threshold 참고값
    logger.info("")
    logger.info("  Pruning threshold 참고:")
    logger.info(f"    threshold ≥ P25({overall['p25']:.4f}): "
                f"상위 ~75% 선택")
    logger.info(f"    threshold ≥ P50({overall['p50']:.4f}): "
                f"상위 ~50% 선택")
    logger.info(f"    threshold ≥ P75({overall['p75']:.4f}): "
                f"상위 ~25% 선택")
    logger.info("=" * 60)


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CASDA composed 이미지 후행 품질 평가 — "
                    "metadata.json의 suitability_score를 실제 품질 점수로 업데이트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용 (in-place 업데이트, 원본 백업)
  python scripts/score_casda_quality.py \\
      --casda-dir /content/drive/MyDrive/data/Severstal/augmented_dataset_v5.5/casda_composed

  # 별도 출력 디렉토리
  python scripts/score_casda_quality.py \\
      --casda-dir /content/drive/MyDrive/data/Severstal/augmented_dataset_v5.5/casda_composed \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/quality_scored

  # 커스텀 가중치
  python scripts/score_casda_quality.py \\
      --casda-dir /path/to/casda_composed \\
      --weight-color 0.35 --weight-artifact 0.35 --weight-blur 0.30
""",
    )

    parser.add_argument(
        "--casda-dir", type=str, required=True,
        help="CASDA composed 디렉토리 (metadata.json + images/ 포함)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="출력 디렉토리 (미지정 시 casda-dir에 in-place 업데이트)",
    )
    parser.add_argument(
        "--weight-color", type=float, default=0.40,
        help="color_score 가중치 (기본 0.40)",
    )
    parser.add_argument(
        "--weight-artifact", type=float, default=0.30,
        help="artifact_score 가중치 (기본 0.30)",
    )
    parser.add_argument(
        "--weight-blur", type=float, default=0.30,
        help="blur_score 가중치 (기본 0.30)",
    )
    parser.add_argument(
        "--batch-log-interval", type=int, default=500,
        help="진행률 로그 간격 (기본 500)",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="원본 metadata.json 백업을 생성하지 않음",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="점수 계산만 수행하고 metadata.json을 수정하지 않음 "
             "(통계만 출력)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="병렬 워커 수 (기본 0 = 순차 처리, -1 = CPU 코어 수 자동 감지, "
             "N >= 2 = N개 프로세스 병렬 처리)",
    )

    args = parser.parse_args(argv)

    # 가중치 합 검증
    w_sum = args.weight_color + args.weight_artifact + args.weight_blur
    if abs(w_sum - 1.0) > 0.01:
        parser.error(
            f"가중치 합이 1.0이 아닙니다: "
            f"{args.weight_color} + {args.weight_artifact} + "
            f"{args.weight_blur} = {w_sum:.4f}"
        )

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    casda_dir = Path(args.casda_dir)
    if not casda_dir.exists():
        logger.error(f"CASDA 디렉토리가 존재하지 않습니다: {casda_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None

    # 파일 핸들러 추가 (output_dir 또는 casda_dir에 로그 저장)
    log_dir = output_dir or casda_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        str(log_dir / "quality_scoring.log"),
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("CASDA Composed 이미지 후행 품질 평가")
    logger.info("=" * 60)
    logger.info(f"  casda_dir: {casda_dir}")
    logger.info(f"  output_dir: {output_dir or '(in-place)'}")
    logger.info(f"  가중치: color={args.weight_color}, "
                f"artifact={args.weight_artifact}, "
                f"blur={args.weight_blur}")
    logger.info(f"  dry_run: {args.dry_run}")

    # 워커 수 결정
    num_workers = args.workers
    if num_workers < 0:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count - 1)
        logger.info(f"  워커 수 자동 설정: {num_workers} (CPU: {cpu_count})")
    else:
        logger.info(f"  workers: {num_workers} "
                    f"({'순차 처리' if num_workers <= 1 else '병렬 처리'})")

    # 1. metadata 로드
    try:
        metadata = load_metadata(casda_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    # 2. 품질 평가
    updated_metadata, stats = score_images(
        metadata=metadata,
        casda_dir=casda_dir,
        weight_color=args.weight_color,
        weight_artifact=args.weight_artifact,
        weight_blur=args.weight_blur,
        batch_log_interval=args.batch_log_interval,
        num_workers=num_workers,
    )

    # 3. 결과 요약
    print_summary(stats)

    # 4. 저장
    if args.dry_run:
        logger.info("(dry-run 모드 — metadata.json 미수정)")
        # 통계만 저장
        stats_out = (output_dir or casda_dir) / "quality_stats.json"
        (output_dir or casda_dir).mkdir(parents=True, exist_ok=True)
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"quality_stats.json 저장: {stats_out}")
    else:
        save_results(
            updated_metadata=updated_metadata,
            stats=stats,
            casda_dir=casda_dir,
            output_dir=output_dir,
            backup=not args.no_backup,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
