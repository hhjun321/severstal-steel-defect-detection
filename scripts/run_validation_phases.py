"""
ControlNet Model Validation - Multi-Phase Batch Runner
=======================================================

재학습된 ControlNet 모델의 생성 품질을 4단계로 검증합니다.

Phase 1: 기본 생성 검증 (학습 데이터로 모델 작동 확인)
Phase 2: 파라미터 탐색 (guidance_scale, inference_steps 조합 비교)
Phase 3: 미학습 데이터 일반화 검증
Phase 4: 결과 종합 시각화 및 리포트

Usage (Colab):
    # 전체 Phase 실행 (권장: --model_path로 best_model 사용)
    python scripts/run_validation_phases.py \
        --model_path /content/drive/MyDrive/data/Severstal/controlnet_training/best_model \
        --jsonl_path /content/drive/MyDrive/data/Severstal/controlnet_dataset/train.jsonl \
        --roi_metadata_path /content/drive/MyDrive/data/Severstal/roi_patches/roi_metadata.csv \
        --training_log_path /content/drive/MyDrive/data/Severstal/controlnet_training/training_log.json \
        --output_base /content/drive/MyDrive/data/Severstal/test_results

    # 특정 Phase만 실행
    python scripts/run_validation_phases.py \
        --model_path .../best_model \
        --phases 1 2

    # pipeline 사용 (저장 최적화 이전 모델)
    python scripts/run_validation_phases.py \
        --pipeline_path /content/drive/MyDrive/data/Severstal/controlnet_training/pipeline \
        --phases 1

Note:
    저장 최적화(--skip_save_pipeline, --save_fp16) 적용 후에는 pipeline/
    디렉토리가 생성되지 않으므로 --model_path를 사용해야 합니다.
    --model_path는 test_controlnet.py에서 pipeline_reference.json을 참조하여
    base SD model을 자동으로 로드합니다.

Dependencies (Colab):
    pip install lpips scikit-image
    # lpips: LPIPS 지각적 유사도 메트릭 (선택사항, 없으면 자동 스킵)
    # scikit-image: SSIM 구조적 유사도 메트릭 (선택사항, 없으면 자동 스킵)
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Generation Quality Evaluation
# =============================================================================

def _evaluate_single_image_cpu(args_tuple: tuple) -> Optional[Dict]:
    """
    단일 이미지의 CPU-bound 품질 메트릭을 계산하는 워커 함수.

    ProcessPoolExecutor에서 pickle 직렬화가 가능하도록 모듈 레벨에 정의.
    3개 기본 메트릭(color, artifact, blur) + SSIM(선택)을 계산한다.
    LPIPS는 GPU(torch) 모델이 필요하므로 워커에서 제외, 메인 프로세스에서 처리.

    Args:
        args_tuple: (img_path_str, ref_path_str_or_none, ssim_available)

    Returns:
        성공 시: {"filename": str, "color": float, "artifact": float,
                  "blur": float, "ssim": float|None} dict
        실패 시: None
    """
    import cv2
    import numpy as _np

    img_path_str, ref_path_str, ssim_available = args_tuple

    img = cv2.imread(img_path_str)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3개 기본 메트릭 (모듈 레벨 함수 직접 호출)
    color_score = _score_color_consistency(img_rgb)
    artifact_score = _score_artifacts(gray)
    blur_score = _score_sharpness(gray)

    # SSIM (CPU-bound, skimage 사용)
    ssim_score = None
    if ssim_available and ref_path_str is not None:
        ref_img = cv2.imread(ref_path_str)
        if ref_img is not None:
            ssim_score = _compute_ssim(gray, ref_img)

    filename = str(Path(img_path_str).name)
    return {
        "filename": filename,
        "color": color_score,
        "artifact": artifact_score,
        "blur": blur_score,
        "ssim": ssim_score,
    }


def _calc_weighted_quality(
    color_score: float,
    artifact_score: float,
    blur_score: float,
    ssim_score: Optional[float],
    lpips_score: Optional[float],
) -> float:
    """SSIM/LPIPS 가용성에 따라 동적 가중치를 적용한 품질 점수를 계산한다."""
    if ssim_score is not None and lpips_score is not None:
        return (0.30 * color_score + 0.20 * artifact_score
                + 0.20 * blur_score + 0.15 * ssim_score
                + 0.15 * (1.0 - lpips_score))
    elif ssim_score is not None:
        return (0.30 * color_score + 0.25 * artifact_score
                + 0.25 * blur_score + 0.20 * ssim_score)
    elif lpips_score is not None:
        return (0.30 * color_score + 0.25 * artifact_score
                + 0.25 * blur_score + 0.20 * (1.0 - lpips_score))
    else:
        return (0.40 * color_score + 0.30 * artifact_score
                + 0.30 * blur_score)


def _compute_generation_quality(
    output_dir: Path,
    quality_threshold: float = 0.5,
    num_workers: int = 0,
) -> Dict:
    """생성된 이미지의 정량적 품질을 평가합니다.

    5개 메트릭을 사용합니다 (SSIM/LPIPS는 참조 이미지 존재 시에만 산출):
    - color_consistency: LAB 색공간 기반 금속 표면 색상 일관성
    - artifact_score: gradient 분석 기반 비현실적 패턴 감지
    - blur_score: Laplacian variance 기반 선명도
    - ssim_score: 구조적 유사도 (Structural Similarity Index)
    - lpips_score: 지각적 유사도 (Learned Perceptual Image Patch Similarity)

    Args:
        output_dir: 생성 이미지가 포함된 디렉토리
        quality_threshold: 품질 점수 합격 임계값 (기본 0.5)

    Returns:
        {
            "avg_quality_score": float,
            "quality_passed": bool,
            "sample_scores": [...],
            "avg_color_score": float,
            "avg_artifact_score": float,
            "avg_blur_score": float,
            "avg_ssim_score": float or None,
            "avg_lpips_score": float or None,
            "num_evaluated": int,
            "num_ssim_evaluated": int,
            "num_lpips_evaluated": int,
        }
    """
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not available, skipping quality evaluation")
        return {
            "avg_quality_score": -1.0,
            "quality_passed": True,  # 평가 불가 시 통과 처리
            "sample_scores": [],
            "num_evaluated": 0,
            "error": "cv2 not available",
        }

    # SSIM 사용 가능 여부 확인
    ssim_available = False
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_available = True
    except ImportError:
        logger.info("  skimage not available, SSIM will be skipped")

    # LPIPS 사용 가능 여부 확인
    lpips_available = False
    lpips_model = None
    try:
        import torch
        import lpips as lpips_module
        lpips_model = lpips_module.LPIPS(net="alex", verbose=False)
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
        lpips_available = True
        logger.info("  LPIPS model loaded (AlexNet)")
    except ImportError:
        logger.info("  lpips/torch not available, LPIPS will be skipped")
    except Exception as e:
        logger.info(f"  LPIPS initialization failed: {e}")

    # 생성 이미지 수집 (generated/ 하위 또는 직접)
    image_paths = []
    for pattern in ["generated/*_gen*.png", "generated/*_generated_*.png",
                     "*_gen*.png", "*_generated_*.png"]:
        image_paths.extend(output_dir.glob(pattern))

    # 중복 제거
    image_paths = list(set(image_paths))

    if not image_paths:
        logger.warning(f"No generated images found in {output_dir}")
        return {
            "avg_quality_score": -1.0,
            "quality_passed": True,
            "sample_scores": [],
            "num_evaluated": 0,
            "error": "no generated images found",
        }

    # 참조(소스) 이미지 매핑 구축 (SSIM/LPIPS용)
    # comparisons/ 디렉토리에서 원본 이미지를 찾거나,
    # 파일명 패턴에서 원본 파일명을 추출
    reference_map = _build_reference_map(output_dir)

    sample_scores = []
    use_parallel = num_workers > 1 and len(image_paths) > 10

    if use_parallel:
        # ── 병렬 경로: CPU-bound 3메트릭 + SSIM을 ProcessPoolExecutor로 처리 ──
        logger.info(f"  품질 평가 병렬 모드: {num_workers} workers, "
                    f"{len(image_paths)} images")

        # 워커 인자 준비
        worker_args = []
        sorted_paths = sorted(image_paths)
        for img_path in sorted_paths:
            ref_path_str = None
            if img_path.name in reference_map:
                ref_path_str = str(reference_map[img_path.name])
            worker_args.append((str(img_path), ref_path_str, ssim_available))

        # 병렬 실행: 3메트릭 + SSIM
        cpu_results: Dict[str, Dict] = {}  # filename -> result dict
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_evaluate_single_image_cpu, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    cpu_results[result["filename"]] = result

        # LPIPS는 GPU 모델이므로 순차적으로 메인 프로세스에서 처리
        for img_path in sorted_paths:
            fname = img_path.name
            if fname not in cpu_results:
                continue

            r = cpu_results[fname]
            color_score = r["color"]
            artifact_score = r["artifact"]
            blur_score = r["blur"]
            ssim_score = r["ssim"]

            # LPIPS (순차)
            lpips_score = None
            if lpips_available and fname in reference_map:
                ref_img = cv2.imread(str(reference_map[fname]))
                if ref_img is not None:
                    gen_img = cv2.imread(str(img_path))
                    if gen_img is not None:
                        lpips_score = _compute_lpips(gen_img, ref_img, lpips_model)

            # 가중 평균 계산
            quality_score = _calc_weighted_quality(
                color_score, artifact_score, blur_score, ssim_score, lpips_score
            )

            score_entry = {
                "filename": fname,
                "quality_score": round(float(quality_score), 4),
                "color_score": round(float(color_score), 4),
                "artifact_score": round(float(artifact_score), 4),
                "blur_score": round(float(blur_score), 4),
            }
            if ssim_score is not None:
                score_entry["ssim_score"] = round(float(ssim_score), 4)
            if lpips_score is not None:
                score_entry["lpips_score"] = round(float(lpips_score), 4)

            sample_scores.append(score_entry)
    else:
        # ── 순차 경로 (기존 로직) ──
        for img_path in sorted(image_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Color/distribution quality (weight: 0.30)
            color_score = _score_color_consistency(img_rgb)

            # 2. Artifact detection (weight: 0.20)
            artifact_score = _score_artifacts(gray)

            # 3. Sharpness (weight: 0.20)
            blur_score = _score_sharpness(gray)

            # 4. SSIM (weight: 0.15, 참조 이미지 존재 시)
            ssim_score = None
            if ssim_available and img_path.name in reference_map:
                ref_img = cv2.imread(str(reference_map[img_path.name]))
                if ref_img is not None:
                    ssim_score = _compute_ssim(gray, ref_img)

            # 5. LPIPS (weight: 0.15, 참조 이미지 존재 시)
            lpips_score = None
            if lpips_available and img_path.name in reference_map:
                ref_img = cv2.imread(str(reference_map[img_path.name]))
                if ref_img is not None:
                    lpips_score = _compute_lpips(img, ref_img, lpips_model)

            # 가중 평균 계산
            quality_score = _calc_weighted_quality(
                color_score, artifact_score, blur_score, ssim_score, lpips_score
            )

            score_entry = {
                "filename": img_path.name,
                "quality_score": round(float(quality_score), 4),
                "color_score": round(float(color_score), 4),
                "artifact_score": round(float(artifact_score), 4),
                "blur_score": round(float(blur_score), 4),
            }
            if ssim_score is not None:
                score_entry["ssim_score"] = round(float(ssim_score), 4)
            if lpips_score is not None:
                score_entry["lpips_score"] = round(float(lpips_score), 4)

            sample_scores.append(score_entry)

    if not sample_scores:
        return {
            "avg_quality_score": -1.0,
            "quality_passed": True,
            "sample_scores": [],
            "num_evaluated": 0,
            "error": "all images failed to load",
        }

    avg_quality = np.mean([s["quality_score"] for s in sample_scores])
    avg_color = np.mean([s["color_score"] for s in sample_scores])
    avg_artifact = np.mean([s["artifact_score"] for s in sample_scores])
    avg_blur = np.mean([s["blur_score"] for s in sample_scores])
    quality_passed = float(avg_quality) >= quality_threshold

    # SSIM 평균 (산출된 샘플만)
    ssim_scores = [s["ssim_score"] for s in sample_scores if "ssim_score" in s]
    avg_ssim = round(float(np.mean(ssim_scores)), 4) if ssim_scores else None

    # LPIPS 평균 (산출된 샘플만)
    lpips_scores = [s["lpips_score"] for s in sample_scores if "lpips_score" in s]
    avg_lpips = round(float(np.mean(lpips_scores)), 4) if lpips_scores else None

    logger.info(f"  Quality evaluation: {len(sample_scores)} images")
    logger.info(f"    avg_quality={avg_quality:.4f}  (threshold={quality_threshold})")
    logger.info(f"    color={avg_color:.4f}  artifact={avg_artifact:.4f}  "
                f"blur={avg_blur:.4f}")
    if avg_ssim is not None:
        logger.info(f"    ssim={avg_ssim:.4f}  ({len(ssim_scores)} samples)")
    if avg_lpips is not None:
        logger.info(f"    lpips={avg_lpips:.4f}  ({len(lpips_scores)} samples)")
    logger.info(f"    quality_passed={quality_passed}")

    result = {
        "avg_quality_score": round(float(avg_quality), 4),
        "quality_passed": quality_passed,
        "sample_scores": sample_scores,
        "avg_color_score": round(float(avg_color), 4),
        "avg_artifact_score": round(float(avg_artifact), 4),
        "avg_blur_score": round(float(avg_blur), 4),
        "avg_ssim_score": avg_ssim,
        "avg_lpips_score": avg_lpips,
        "num_evaluated": len(sample_scores),
        "num_ssim_evaluated": len(ssim_scores),
        "num_lpips_evaluated": len(lpips_scores),
    }
    return result


def _build_reference_map(output_dir: Path) -> Dict[str, Path]:
    """생성 이미지에 대응하는 참조(원본) 이미지 경로 매핑을 구축합니다.

    다음 위치에서 참조 이미지를 탐색:
    1. comparisons/ 디렉토리의 *_comparison.png 파일 (batch 모드)
    2. sources/ 디렉토리의 *.png 파일 (batch/single 모드 공통)
    3. output_dir 직접의 *_generated_*.png 패턴 (single-image 모드)
       → 파일명에서 base_name을 추출하여 sources/ 참조 매핑
    """
    ref_map = {}

    # 1. comparisons/ 디렉토리에서 source 이미지 탐색
    comparisons_dir = output_dir / "comparisons"
    if comparisons_dir.exists():
        for comp_file in comparisons_dir.glob("*_comparison.png"):
            # comparison 파일에서 base_id 추출
            # 형식: {image_id}_class{N}_region{M}_comparison.png
            base_name = comp_file.stem.replace("_comparison", "")
            # 대응하는 생성 이미지명 패턴
            for gen_pattern in [f"{base_name}_gen*.png",
                                f"{base_name}_generated_*.png"]:
                for gen_file in output_dir.glob(f"generated/{gen_pattern}"):
                    ref_map[gen_file.name] = comp_file
                for gen_file in output_dir.glob(gen_pattern):
                    ref_map[gen_file.name] = comp_file

    # 2. sources/ 디렉토리에서 원본 ROI/hint 이미지 탐색
    sources_dir = output_dir / "sources"
    if sources_dir.exists():
        for src_file in sources_dir.glob("*.png"):
            base_name = src_file.stem
            for gen_pattern in [f"{base_name}_gen*.png",
                                f"{base_name}_generated_*.png"]:
                # batch 모드: generated/ 하위
                for gen_file in output_dir.glob(f"generated/{gen_pattern}"):
                    ref_map[gen_file.name] = src_file
                # single-image 모드: output_dir 직접
                for gen_file in output_dir.glob(gen_pattern):
                    ref_map[gen_file.name] = src_file

    # 3. single-image 모드 폴백:
    #    sources/ 없이 output_dir에 *_generated_*.png가 직접 있는 경우,
    #    같은 디렉토리의 *_comparison.png를 참조로 사용
    if not ref_map:
        for gen_file in output_dir.glob("*_generated_*.png"):
            # {hint_stem}_generated_{i}.png → hint_stem 추출
            gen_stem = gen_file.stem
            # "_generated_" 이전의 모든 부분이 hint_stem
            idx = gen_stem.rfind("_generated_")
            if idx > 0:
                hint_stem = gen_stem[:idx]
                # comparison.png를 참조로 사용 (hint + generated 그리드)
                comp_file = output_dir / f"{hint_stem}_comparison.png"
                if comp_file.exists():
                    ref_map[gen_file.name] = comp_file

    if ref_map:
        logger.info(f"  Reference images found: {len(ref_map)} mappings")
    else:
        logger.info("  No reference images found; SSIM/LPIPS will be skipped")

    return ref_map


def _compute_ssim(gen_gray: np.ndarray, ref_img: np.ndarray) -> float:
    """생성 이미지와 참조 이미지의 구조적 유사도를 계산합니다.

    참조 이미지가 hint (에지 마스크)인 경우, 생성 이미지의 Canny 에지맵과
    hint 간 SSIM을 계산하여 구조적 정합성을 평가합니다.
    참조 이미지가 실사 이미지인 경우, 직접 SSIM을 계산합니다.

    Args:
        gen_gray: 생성 이미지 (grayscale)
        ref_img: 참조 이미지 (BGR)

    Returns:
        SSIM 값 (0.0~1.0, 높을수록 유사)
    """
    import cv2
    from skimage.metrics import structural_similarity as ssim

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # 크기가 다르면 참조 이미지를 생성 이미지 크기로 리사이즈
    if gen_gray.shape != ref_gray.shape:
        ref_gray = cv2.resize(ref_gray, (gen_gray.shape[1], gen_gray.shape[0]))

    # 참조가 hint (에지 마스크)인지 판별:
    # hint 이미지는 대부분 0(검정)과 255(흰색) 값을 가짐
    unique_ratio = len(np.unique(ref_gray)) / 256.0
    is_hint = unique_ratio < 0.15  # 전체 밝기의 15% 미만만 사용 → 바이너리 마스크

    if is_hint:
        # 생성 이미지의 Canny 에지맵을 추출하여 hint와 비교
        # → ControlNet이 hint 구조를 얼마나 잘 따르는지 평가
        edges = cv2.Canny(gen_gray, 50, 150)

        # 에지맵과 hint의 구조적 유사도 (win_size는 작은 이미지 대응)
        min_dim = min(edges.shape[0], edges.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
        score, _ = ssim(edges, ref_gray, full=True, win_size=win_size)
    else:
        # 실사 참조 이미지와 직접 비교
        min_dim = min(gen_gray.shape[0], gen_gray.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
        score, _ = ssim(gen_gray, ref_gray, full=True, win_size=win_size)

    return float(max(0.0, min(1.0, score)))


def _compute_lpips(gen_img: np.ndarray, ref_img: np.ndarray,
                   lpips_model) -> float:
    """두 이미지 간 LPIPS(Learned Perceptual Similarity)를 계산합니다.

    Args:
        gen_img: 생성 이미지 (BGR, uint8)
        ref_img: 참조 이미지 (BGR, uint8)
        lpips_model: 초기화된 LPIPS 모델

    Returns:
        LPIPS 거리 (0.0~1.0, 낮을수록 유사)
    """
    import cv2
    import torch

    # 크기 맞추기
    if gen_img.shape[:2] != ref_img.shape[:2]:
        ref_img = cv2.resize(ref_img, (gen_img.shape[1], gen_img.shape[0]))

    # BGR → RGB, [0,255] → [-1,1], HWC → CHW → NCHW
    def _to_tensor(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).float() / 255.0  # [0,1]
        t = t * 2.0 - 1.0  # [-1,1]
        t = t.permute(2, 0, 1).unsqueeze(0)  # NCHW
        return t

    gen_t = _to_tensor(gen_img)
    ref_t = _to_tensor(ref_img)

    device = next(lpips_model.parameters()).device
    gen_t = gen_t.to(device)
    ref_t = ref_t.to(device)

    with torch.no_grad():
        distance = lpips_model(gen_t, ref_t)

    return float(distance.item())


def _score_color_consistency(img_rgb: np.ndarray) -> float:
    """LAB 색공간 + 히스토그램 분석 기반 금속 표면 품질 점수.

    금속 표면 이미지의 시각적 품질을 다각도로 평가합니다:
    - 채도: 금속 표면은 무채색 계열이어야 함
    - 밝기 분포: 적절한 dynamic range를 가져야 함
    - 텍스처 풍부함: 너무 균일하거나 노이즈만 있으면 안 됨
    - 히스토그램 엔트로피: 밝기 값의 다양성

    Returns:
        0.0~1.0 (1.0 = 고품질 금속 표면)
    """
    import cv2

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # 1. 채도 점수 (a, b 채널 편차가 작을수록 무채색)
    a_channel = lab[:, :, 1].astype(float) - 128.0
    b_channel = lab[:, :, 2].astype(float) - 128.0
    ab_abs_mean = (np.mean(np.abs(a_channel)) + np.mean(np.abs(b_channel))) / 2.0

    if ab_abs_mean <= 3.0:
        chroma_score = 1.0
    elif ab_abs_mean >= 25.0:
        chroma_score = 0.0
    else:
        chroma_score = 1.0 - (ab_abs_mean - 3.0) / 22.0

    # 2. 밝기 범위 (dynamic range)
    #    L 채널의 5th-95th percentile 범위가 적절해야 함
    l_channel = lab[:, :, 0].astype(float)
    l_p5 = np.percentile(l_channel, 5)
    l_p95 = np.percentile(l_channel, 95)
    l_range = l_p95 - l_p5

    # 적절한 dynamic range: 30~150 (금속 표면의 자연스러운 밝기 변화)
    if 30 <= l_range <= 150:
        range_score = 1.0
    elif l_range < 15:
        range_score = 0.2  # 너무 균일 (빈 이미지 의심)
    elif l_range < 30:
        range_score = 0.2 + 0.8 * (l_range - 15) / 15.0
    else:
        # l_range > 150: 과도한 콘트라스트
        range_score = max(0.3, 1.0 - (l_range - 150) / 100.0)

    # 3. 히스토그램 엔트로피 (밝기 값 다양성)
    #    엔트로피가 높을수록 풍부한 밝기 분포 → 자연스러움
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]  # log(0) 방지
    entropy = -np.sum(hist * np.log2(hist))

    # 엔트로피 범위: 이론상 0~6 (64 bins), 자연 이미지 ~4.5-5.5
    if entropy >= 4.5:
        entropy_score = 1.0
    elif entropy <= 2.0:
        entropy_score = 0.1
    else:
        entropy_score = 0.1 + 0.9 * (entropy - 2.0) / 2.5

    # 4. 텍스처 복잡도 (L 채널의 국소 분산)
    #    금속 표면은 미세한 텍스처가 있어야 함
    l_std = np.std(l_channel)
    if 8 <= l_std <= 50:
        texture_score = 1.0
    elif l_std < 3:
        texture_score = 0.1  # 거의 단색
    elif l_std < 8:
        texture_score = 0.1 + 0.9 * (l_std - 3) / 5.0
    else:
        # l_std > 50: 과도한 변화
        texture_score = max(0.3, 1.0 - (l_std - 50) / 30.0)

    score = (0.20 * chroma_score
             + 0.25 * range_score
             + 0.30 * entropy_score
             + 0.25 * texture_score)
    return float(score)


def _score_artifacts(gray: np.ndarray) -> float:
    """Gradient 분석 기반 아티팩트 감지.

    비정상적으로 높은 gradient 비율 + 체커보드/반복 패턴을 감지합니다.
    금속 표면은 자연스러운 에지와 텍스처를 가지므로, 단순 gradient 임계값
    대신 통계적 이상치 비율 + 고주파 에너지 비율을 사용합니다.

    Returns:
        0.0~1.0 (1.0 = 아티팩트 없음)
    """
    import cv2

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 1. 통계적 이상치 비율 (mean + 3*std 초과)
    #    자연 이미지: 이상치 비율 < 1%
    #    아티팩트 이미지: 이상치 비율 > 5%
    grad_mean = np.mean(gradient_magnitude)
    grad_std = np.std(gradient_magnitude)
    outlier_threshold = grad_mean + 3.0 * grad_std
    # grad_std가 0이면 (균일한 이미지) 아티팩트 없음으로 판정
    if grad_std < 1e-6:
        outlier_ratio = 0.0
    else:
        outlier_ratio = np.sum(gradient_magnitude > outlier_threshold) / gradient_magnitude.size

    # outlier_ratio: 0~0.01 → 1.0, 0.05+ → 0.0 (선형)
    if outlier_ratio <= 0.01:
        edge_score = 1.0
    elif outlier_ratio >= 0.05:
        edge_score = 0.0
    else:
        edge_score = 1.0 - (outlier_ratio - 0.01) / 0.04

    # 2. 고주파 에너지 비율 (체커보드/반복 아티팩트 감지)
    #    Laplacian의 절대값 평균 대비 gradient 평균의 비율로,
    #    급격한 on-off 패턴이 많을수록 높아짐
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_energy = np.mean(np.abs(laplacian))
    grad_energy = grad_mean + 1e-6  # 0 방지

    hf_ratio = lap_energy / grad_energy
    # hf_ratio: 자연이미지 ~1.0-2.5, 체커보드 아티팩트 >3.0
    if hf_ratio <= 2.5:
        hf_score = 1.0
    elif hf_ratio >= 5.0:
        hf_score = 0.0
    else:
        hf_score = 1.0 - (hf_ratio - 2.5) / 2.5

    # 가중 결합
    score = 0.6 * edge_score + 0.4 * hf_score
    return float(score)


def _score_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance + gradient 분석 기반 선명도 점수.

    단순 Laplacian variance만으로는 diffusion 모델 출력 간 차이를
    구분하기 어려우므로, gradient 기반 에지 선명도를 추가합니다.

    Returns:
        0.0~1.0 (1.0 = 선명)
    """
    import cv2

    # 1. Laplacian variance (전체 선명도)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 금속 표면 이미지의 Laplacian variance 범위:
    # - 흐린 diffusion 출력: 50~200
    # - 적절한 출력: 200~800
    # - 매우 선명/노이즈: 800+
    if laplacian_var >= 500:
        lap_score = 1.0
    elif laplacian_var <= 30:
        lap_score = 0.1
    else:
        lap_score = 0.1 + 0.9 * (laplacian_var - 30) / 470.0

    # 2. Gradient 기반 에지 선명도
    #    Sobel gradient의 P90/P50 비율: 선명한 이미지는 에지가 뚜렷하여
    #    gradient 분포가 넓고, 흐린 이미지는 분포가 좁음
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    p50 = np.percentile(gradient, 50)
    p90 = np.percentile(gradient, 90)

    if p50 < 1e-6:
        # 거의 균일한 이미지
        edge_sharpness = 0.1
    else:
        contrast_ratio = p90 / p50
        # 자연 이미지: contrast_ratio ~3-8
        # 흐린 이미지: contrast_ratio ~1.5-2.5
        if contrast_ratio >= 4.0:
            edge_sharpness = 1.0
        elif contrast_ratio <= 1.5:
            edge_sharpness = 0.1
        else:
            edge_sharpness = 0.1 + 0.9 * (contrast_ratio - 1.5) / 2.5

    score = 0.5 * lap_score + 0.5 * edge_sharpness
    return float(score)


def _create_oversampled_jsonl(
    original_jsonl_path: str,
    output_dir: Path,
) -> Optional[Path]:
    """소수 클래스를 오버샘플링한 JSONL을 생성합니다.

    각 클래스의 샘플 수를 최다 클래스와 동일하게 맞춥니다.
    오버샘플링 시 seed를 변경하여 다양성을 확보합니다.

    Args:
        original_jsonl_path: 원본 train.jsonl 경로
        output_dir: 오버샘플링된 JSONL 저장 디렉토리

    Returns:
        생성된 오버샘플링 JSONL 경로, 실패 시 None
    """
    import re

    original_path = Path(original_jsonl_path)
    if not original_path.exists():
        logger.warning(f"Original JSONL not found: {original_jsonl_path}")
        return None

    # 원본 JSONL 로드
    samples = []
    with open(original_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        return None

    # 클래스별 분류 (hint 경로에서 class 정보 추출)
    class_pattern = re.compile(r"class(\d+)")
    samples_by_class = {}

    for sample in samples:
        hint = sample.get("hint", "")
        prompt = sample.get("prompt", "")
        # hint 파일명 또는 prompt에서 class 추출
        match = class_pattern.search(hint) or class_pattern.search(prompt)
        class_id = int(match.group(1)) if match else 0

        if class_id not in samples_by_class:
            samples_by_class[class_id] = []
        samples_by_class[class_id].append(sample)

    # 분포 로깅
    logger.info("  Original class distribution:")
    for cls in sorted(samples_by_class.keys()):
        logger.info(f"    Class {cls}: {len(samples_by_class[cls])} samples")

    if len(samples_by_class) <= 1:
        logger.info("  Only one class found, skipping oversampling")
        return None

    # 최다 클래스 샘플 수
    max_count = max(len(v) for v in samples_by_class.values())

    # 오버샘플링
    oversampled = []
    for cls in sorted(samples_by_class.keys()):
        cls_samples = samples_by_class[cls]
        oversampled.extend(cls_samples)  # 원본 그대로 추가

        # 부족한 수만큼 반복 추가 (seed_offset으로 다양성 확보)
        deficit = max_count - len(cls_samples)
        if deficit > 0:
            logger.info(
                f"    Oversampling Class {cls}: "
                f"{len(cls_samples)} → {max_count} (+{deficit})"
            )
            for i in range(deficit):
                # 원본 샘플을 순환 선택하고 seed_offset 표시
                src = cls_samples[i % len(cls_samples)].copy()
                src["_oversample_seed_offset"] = (i + 1) * 100
                oversampled.append(src)

    # 오버샘플링된 JSONL 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    oversampled_path = output_dir / "train_oversampled.jsonl"
    with open(oversampled_path, "w", encoding="utf-8") as f:
        for sample in oversampled:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"  Oversampled JSONL: {len(samples)} → {len(oversampled)} samples")
    logger.info(f"  Saved to: {oversampled_path}")

    # 오버샘플링 후 분포 확인
    logger.info("  Oversampled class distribution:")
    new_by_class = {}
    for sample in oversampled:
        hint = sample.get("hint", "")
        prompt = sample.get("prompt", "")
        match = class_pattern.search(hint) or class_pattern.search(prompt)
        class_id = int(match.group(1)) if match else 0
        new_by_class[class_id] = new_by_class.get(class_id, 0) + 1
    for cls in sorted(new_by_class.keys()):
        logger.info(f"    Class {cls}: {new_by_class[cls]} samples")

    return oversampled_path


# =============================================================================
# Phase 1: 기본 생성 검증
# =============================================================================

def run_phase1(args):
    """Phase 1: 학습 데이터로 기본 생성 검증

    --oversample_minority가 활성화된 경우, Class 3/4 등 소수 클래스의
    샘플을 seed 변경을 통해 오버샘플링하여 클래스 균형을 맞춥니다.
    """
    logger.info("=" * 60)
    logger.info("  Phase 1: Basic Generation Validation")
    logger.info("=" * 60)

    output_dir = Path(args.output_base) / "phase1_basic"

    # 클래스 불균형 오버샘플링 처리
    jsonl_path = args.jsonl_path
    oversample = getattr(args, "oversample_minority", False)

    if oversample:
        oversampled_path = _create_oversampled_jsonl(
            jsonl_path, output_dir
        )
        if oversampled_path:
            jsonl_path = str(oversampled_path)
            logger.info(f"  Using oversampled JSONL: {jsonl_path}")

    cmd = [
        sys.executable, str(args.script_dir / "test_controlnet.py"),
    ]

    # model source (model_path 우선, pipeline_path는 fallback)
    cmd += _model_source_args(args)

    cmd += [
        "--jsonl_path", jsonl_path,
        "--output_dir", str(output_dir),
        "--num_inference_steps", "30",
        "--guidance_scale", "7.5",
        "--controlnet_conditioning_scale", str(args.controlnet_conditioning_scale),
        "--seed", "42",
    ]

    if args.image_root:
        cmd += ["--image_root", args.image_root]

    if args.resolution is not None:
        cmd += ["--resolution", str(args.resolution)]

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Phase 1 failed:\n{result.stderr}")
        return False
    logger.info(result.stdout)

    # 결과 확인
    summary_path = output_dir / "generation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        generated = summary["generated"]
        total = summary["total_samples"]
        logger.info(
            f"Phase 1 generation: {generated}/{total} samples generated"
        )

        if generated == 0:
            logger.error("Phase 1 FAIL: No images generated")
            return False

        # 품질 평가
        quality_threshold = getattr(args, "quality_threshold", 0.5)
        quality_result = _compute_generation_quality(
            output_dir, quality_threshold,
            num_workers=getattr(args, "workers", 0),
        )

        # 품질 결과를 generation_summary에 병합하여 저장
        summary["quality"] = quality_result
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if quality_result.get("num_evaluated", 0) > 0:
            passed = quality_result["quality_passed"]
            avg_q = quality_result["avg_quality_score"]
            logger.info(
                f"Phase 1 quality: avg={avg_q:.4f}  "
                f"threshold={quality_threshold}  "
                f"{'PASS' if passed else 'FAIL'}"
            )
            return passed
        else:
            logger.warning("Phase 1: Quality evaluation skipped (no images evaluated)")
            return True  # 평가 불가 시 생성 성공만으로 통과
    else:
        logger.warning("Phase 1: No summary file generated")
        return False


# =============================================================================
# Phase 2: 파라미터 탐색
# =============================================================================

PHASE2_GUIDANCE_SCALES = [3.0, 5.0, 7.5, 10.0]
PHASE2_INFERENCE_STEPS = [20, 30, 50]
PHASE2_SEEDS = [42, 123, 456, 789]


def run_phase2(args):
    """Phase 2: 다양한 파라미터 조합으로 생성 품질 비교"""
    logger.info("=" * 60)
    logger.info("  Phase 2: Parameter Exploration")
    logger.info("=" * 60)

    # 대표 hint 1개 선택 (첫 번째 학습 샘플)
    representative_hint = _get_first_hint(args.jsonl_path)
    if representative_hint is None:
        logger.error("Phase 2: Cannot find representative hint from JSONL")
        return False

    phase2_results = []

    # guidance_scale 탐색
    logger.info("--- Guidance Scale Exploration ---")
    for gs in PHASE2_GUIDANCE_SCALES:
        output_dir = Path(args.output_base) / f"phase2_gs{gs}"

        cmd = [
            sys.executable, str(args.script_dir / "test_controlnet.py"),
        ]
        cmd += _model_source_args(args)

        cmd += [
            "--hint_image", representative_hint["hint_path"],
            "--prompt", representative_hint["prompt"],
            "--output_dir", str(output_dir),
            "--num_inference_steps", "30",
            "--guidance_scale", str(gs),
            "--controlnet_conditioning_scale", str(args.controlnet_conditioning_scale),
            "--num_images_per_sample", "4",
            "--seed", "42",
        ]

        if args.image_root:
            cmd += ["--image_root", args.image_root]

        if args.resolution is not None:
            cmd += ["--resolution", str(args.resolution)]

        logger.info(f"  guidance_scale={gs}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  gs={gs} failed: {result.stderr[:200]}")
        else:
            phase2_results.append({
                "param": "guidance_scale", "value": gs,
                "output_dir": str(output_dir),
            })

    # inference_steps 탐색
    logger.info("--- Inference Steps Exploration ---")
    for steps in PHASE2_INFERENCE_STEPS:
        output_dir = Path(args.output_base) / f"phase2_steps{steps}"

        cmd = [
            sys.executable, str(args.script_dir / "test_controlnet.py"),
        ]
        cmd += _model_source_args(args)

        cmd += [
            "--hint_image", representative_hint["hint_path"],
            "--prompt", representative_hint["prompt"],
            "--output_dir", str(output_dir),
            "--num_inference_steps", str(steps),
            "--guidance_scale", "7.5",
            "--controlnet_conditioning_scale", str(args.controlnet_conditioning_scale),
            "--num_images_per_sample", "4",
            "--seed", "42",
        ]

        if args.image_root:
            cmd += ["--image_root", args.image_root]

        if args.resolution is not None:
            cmd += ["--resolution", str(args.resolution)]

        logger.info(f"  num_inference_steps={steps}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  steps={steps} failed: {result.stderr[:200]}")
        else:
            phase2_results.append({
                "param": "num_inference_steps", "value": steps,
                "output_dir": str(output_dir),
            })

    # seed 다양성 확인
    logger.info("--- Seed Diversity Check ---")
    for seed in PHASE2_SEEDS:
        output_dir = Path(args.output_base) / f"phase2_seed{seed}"

        cmd = [
            sys.executable, str(args.script_dir / "test_controlnet.py"),
        ]
        cmd += _model_source_args(args)

        cmd += [
            "--hint_image", representative_hint["hint_path"],
            "--prompt", representative_hint["prompt"],
            "--output_dir", str(output_dir),
            "--num_inference_steps", "30",
            "--guidance_scale", "7.5",
            "--controlnet_conditioning_scale", str(args.controlnet_conditioning_scale),
            "--num_images_per_sample", "1",
            "--seed", str(seed),
        ]

        if args.image_root:
            cmd += ["--image_root", args.image_root]

        if args.resolution is not None:
            cmd += ["--resolution", str(args.resolution)]

        logger.info(f"  seed={seed}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  seed={seed} failed: {result.stderr[:200]}")
        else:
            phase2_results.append({
                "param": "seed", "value": seed,
                "output_dir": str(output_dir),
            })

    # Phase 2 결과에 대해 정량적 품질 평가 수행
    logger.info("--- Phase 2 Quality Evaluation ---")
    quality_threshold = getattr(args, "quality_threshold", 0.5)
    for entry in phase2_results:
        entry_output_dir = Path(entry["output_dir"])
        if entry_output_dir.exists():
            quality_result = _compute_generation_quality(
                entry_output_dir, quality_threshold,
                num_workers=getattr(args, "workers", 0),
            )
            entry["quality"] = {
                "avg_quality_score": quality_result.get("avg_quality_score"),
                "avg_color_score": quality_result.get("avg_color_score"),
                "avg_artifact_score": quality_result.get("avg_artifact_score"),
                "avg_blur_score": quality_result.get("avg_blur_score"),
                "avg_ssim_score": quality_result.get("avg_ssim_score"),
                "avg_lpips_score": quality_result.get("avg_lpips_score"),
                "num_evaluated": quality_result.get("num_evaluated", 0),
                "quality_passed": quality_result.get("quality_passed"),
            }
            logger.info(
                f"  {entry['param']}={entry['value']}: "
                f"quality={quality_result.get('avg_quality_score', -1):.4f}"
            )

    # 최적 파라미터 조합 결정
    best_result = None
    best_score = -1.0
    for entry in phase2_results:
        q = entry.get("quality", {})
        score = q.get("avg_quality_score", -1.0)
        if score is not None and score > best_score:
            best_score = score
            best_result = entry

    best_params = None
    if best_result is not None:
        best_params = {
            "param": best_result["param"],
            "value": best_result["value"],
            "quality_score": best_score,
        }
        logger.info(f"  Best combination: {best_params['param']}="
                     f"{best_params['value']} (score={best_score:.4f})")

    # Phase 2 결과 요약 저장
    summary_path = Path(args.output_base) / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "guidance_scales_tested": PHASE2_GUIDANCE_SCALES,
            "inference_steps_tested": PHASE2_INFERENCE_STEPS,
            "seeds_tested": PHASE2_SEEDS,
            "results": phase2_results,
            "best_combination": best_params,
        }, f, indent=2)

    logger.info(f"Phase 2 complete: {len(phase2_results)} parameter combinations tested")
    logger.info(f"Summary saved: {summary_path}")
    return len(phase2_results) > 0


# =============================================================================
# Phase 3: 미학습 데이터 일반화 검증
# =============================================================================

# 학습에 사용된 이미지 ID
TRAINED_IMAGE_IDS = {"0002cc93b.jpg", "0007a71bf.jpg", "000a4bcdd.jpg"}

# 클래스별 선별 개수
UNSEEN_SAMPLES_PER_CLASS = 3


def run_phase3(args):
    """Phase 3: 미학습 데이터로 일반화 능력 평가"""
    logger.info("=" * 60)
    logger.info("  Phase 3: Unseen Data Generalization")
    logger.info("=" * 60)

    if not args.roi_metadata_path:
        logger.error("Phase 3 requires --roi_metadata_path")
        return False

    # 1. 미학습 ROI 선별
    unseen_rois = _select_unseen_rois(args.roi_metadata_path)
    if not unseen_rois:
        logger.error("No unseen ROIs found")
        return False

    logger.info(f"Selected {len(unseen_rois)} unseen ROIs for testing")

    # 2. 미학습 ROI로부터 hint 생성 및 inference
    output_dir = Path(args.output_base) / "phase3_unseen"
    output_dir.mkdir(parents=True, exist_ok=True)
    hints_dir = output_dir / "generated_hints"
    hints_dir.mkdir(exist_ok=True)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)

    # hint 생성 및 inference를 위한 임시 JSONL 생성
    unseen_jsonl_path = output_dir / "unseen_test.jsonl"
    hint_generation_success = _generate_unseen_hints(
        unseen_rois, hints_dir, unseen_jsonl_path, args
    )

    if not hint_generation_success:
        logger.warning("Phase 3: Hint generation skipped or failed. "
                       "Attempting inference with ROI images directly.")
        # ROI 이미지를 hint 대신 사용하는 fallback jsonl 생성
        _create_fallback_jsonl(unseen_rois, unseen_jsonl_path)

    # 3. test_controlnet.py로 inference 실행
    cmd = [
        sys.executable, str(args.script_dir / "test_controlnet.py"),
    ]
    cmd += _model_source_args(args)

    cmd += [
        "--jsonl_path", str(unseen_jsonl_path),
        "--output_dir", str(output_dir),
        "--num_inference_steps", "30",
        "--guidance_scale", "7.5",
        "--controlnet_conditioning_scale", str(args.controlnet_conditioning_scale),
        "--seed", "42",
    ]

    if args.image_root:
        cmd += ["--image_root", args.image_root]

    if args.resolution is not None:
        cmd += ["--resolution", str(args.resolution)]

    logger.info(f"Running inference on unseen data...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Phase 3 inference failed:\n{result.stderr}")
        return False
    logger.info(result.stdout)

    # 결과 요약
    summary_path = output_dir / "generation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        generated = summary["generated"]
        total = summary["total_samples"]
        logger.info(
            f"Phase 3 generation: {generated}/{total} unseen samples generated"
        )

        if generated == 0:
            logger.error("Phase 3 FAIL: No images generated")
            return False

        # 품질 평가
        quality_threshold = getattr(args, "quality_threshold", 0.5)
        quality_result = _compute_generation_quality(
            output_dir, quality_threshold,
            num_workers=getattr(args, "workers", 0),
        )

        # 품질 결과를 generation_summary에 병합하여 저장
        summary["quality"] = quality_result
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if quality_result.get("num_evaluated", 0) > 0:
            passed = quality_result["quality_passed"]
            avg_q = quality_result["avg_quality_score"]
            logger.info(
                f"Phase 3 quality: avg={avg_q:.4f}  "
                f"threshold={quality_threshold}  "
                f"{'PASS' if passed else 'FAIL'}"
            )
            return passed
        else:
            logger.warning("Phase 3: Quality evaluation skipped (no images evaluated)")
            return True

    return False


def _select_unseen_rois(roi_metadata_path: str) -> List[Dict]:
    """roi_metadata.csv에서 미학습 ROI를 클래스별로 선별합니다.

    선별 기준:
    - 학습 미사용 이미지의 ROI
    - 클래스별 suitability_score 상위 UNSEEN_SAMPLES_PER_CLASS개
    - recommendation이 'suitable' 또는 'acceptable'인 것
    """
    rois_by_class = {}

    with open(roi_metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            if image_id in TRAINED_IMAGE_IDS:
                continue

            class_id = int(row["class_id"])
            try:
                score = float(row["suitability_score"])
            except (ValueError, KeyError):
                score = 0.0

            recommendation = row.get("recommendation", "")
            if recommendation not in ("suitable", "acceptable"):
                continue

            if class_id not in rois_by_class:
                rois_by_class[class_id] = []
            rois_by_class[class_id].append({
                "image_id": image_id,
                "class_id": class_id,
                "region_id": int(row["region_id"]),
                "suitability_score": score,
                "defect_subtype": row.get("defect_subtype", "general"),
                "background_type": row.get("background_type", "complex_pattern"),
                "stability_score": float(row.get("stability_score", 0.5)),
                "linearity": float(row.get("linearity", 0.5)),
                "solidity": float(row.get("solidity", 0.5)),
                "extent": float(row.get("extent", 0.5)),
                "aspect_ratio": float(row.get("aspect_ratio", 1.0)),
                "roi_image_path": row.get("roi_image_path", ""),
                "roi_mask_path": row.get("roi_mask_path", ""),
                "prompt": row.get("prompt", ""),
            })

    # 클래스별 상위 suitability_score 선별
    selected = []
    for class_id in sorted(rois_by_class.keys()):
        class_rois = sorted(
            rois_by_class[class_id],
            key=lambda x: x["suitability_score"],
            reverse=True,
        )
        n = min(UNSEEN_SAMPLES_PER_CLASS, len(class_rois))
        selected.extend(class_rois[:n])
        logger.info(
            f"  Class {class_id}: selected {n} ROIs "
            f"(total available: {len(class_rois)}, "
            f"top score: {class_rois[0]['suitability_score']:.3f})"
        )

    return selected


def _generate_unseen_hints(
    unseen_rois: List[Dict],
    hints_dir: Path,
    output_jsonl_path: Path,
    args,
) -> bool:
    """미학습 ROI로부터 hint 이미지를 생성하고 JSONL을 작성합니다."""
    try:
        import cv2
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.preprocessing.hint_generator import HintImageGenerator
    except ImportError as e:
        logger.warning(f"Cannot import hint generator dependencies: {e}")
        return False

    generator = HintImageGenerator()
    jsonl_entries = []

    for roi in unseen_rois:
        roi_image_path = roi["roi_image_path"]
        roi_mask_path = roi["roi_mask_path"]

        # 경로 resolve
        try:
            search_dirs = [args.image_root, str(Path(args.jsonl_path).parent)]
            img_path = _resolve_roi_path(roi_image_path, search_dirs)
            mask_path = _resolve_roi_path(roi_mask_path, search_dirs)
        except FileNotFoundError as e:
            logger.warning(f"  Skip {roi['image_id']} region {roi['region_id']}: {e}")
            continue

        # 이미지 로드
        roi_image = cv2.imread(str(img_path))
        if roi_image is None:
            logger.warning(f"  Cannot load image: {img_path}")
            continue
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

        roi_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            logger.warning(f"  Cannot load mask: {mask_path}")
            continue
        roi_mask = (roi_mask > 127).astype(np.uint8)

        # hint 생성
        defect_metrics = {
            "linearity": roi["linearity"],
            "solidity": roi["solidity"],
            "extent": roi["extent"],
            "aspect_ratio": roi["aspect_ratio"],
        }

        hint_image = generator.generate_hint_image(
            roi_image=roi_image,
            roi_mask=roi_mask,
            defect_metrics=defect_metrics,
            background_type=roi["background_type"],
            stability_score=roi["stability_score"],
        )

        # hint 저장
        hint_filename = (
            f"{roi['image_id']}_class{roi['class_id']}"
            f"_region{roi['region_id']}_hint.png"
        )
        hint_path = hints_dir / hint_filename
        generator.save_hint_image(hint_image, hint_path)

        # 프롬프트 생성
        prompt = roi.get("prompt", "")
        if not prompt:
            prompt = (
                f"a {roi['defect_subtype']} surface defect on "
                f"{roi['background_type']} metal surface, "
                f"steel defect class {roi['class_id']}"
            )

        jsonl_entries.append({
            "hint": str(hint_path),
            "target": str(img_path),
            "source": str(img_path),
            "prompt": prompt,
            "negative_prompt": (
                "blurry, low quality, artifacts, noise, distorted, "
                "warped, unrealistic, oversaturated"
            ),
        })

    if not jsonl_entries:
        return False

    # JSONL 저장
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(jsonl_entries)} hint images -> {output_jsonl_path}")
    return True


def _create_fallback_jsonl(unseen_rois: List[Dict], output_jsonl_path: Path):
    """hint 생성이 불가능할 때, ROI 이미지 정보만으로 JSONL을 작성합니다.

    이 경우 Colab에서 수동으로 hint 생성 후 재실행해야 합니다.
    """
    entries = []
    for roi in unseen_rois:
        prompt = roi.get("prompt", "")
        if not prompt:
            prompt = (
                f"a {roi['defect_subtype']} surface defect on "
                f"{roi['background_type']} metal surface, "
                f"steel defect class {roi['class_id']}"
            )

        entries.append({
            "hint": roi["roi_image_path"],  # hint 대신 원본 사용 (임시)
            "target": roi["roi_image_path"],
            "source": roi["roi_image_path"],
            "prompt": prompt,
            "negative_prompt": (
                "blurry, low quality, artifacts, noise, distorted, "
                "warped, unrealistic, oversaturated"
            ),
            "_note": "fallback: using ROI image as hint (hint generation failed)",
        })

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        f"Fallback JSONL created with {len(entries)} entries -> {output_jsonl_path}"
    )


def _resolve_roi_path(path_str: str, search_dirs: list) -> Path:
    """ROI 이미지/마스크 경로를 resolve합니다."""
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path

    filename = path.name

    for base_dir in search_dirs:
        if base_dir is None:
            continue
        base = Path(base_dir)

        candidate = base / path_str
        if candidate.exists():
            return candidate

        candidate = base / filename
        if candidate.exists():
            return candidate

        for subdir in ["images", "masks", "roi_patches/images", "roi_patches/masks"]:
            candidate = base / subdir / filename
            if candidate.exists():
                return candidate

    if path.exists():
        return path

    raise FileNotFoundError(f"Cannot find ROI file: '{path_str}'")


# =============================================================================
# Phase 4: 시각화 및 리포트
# =============================================================================

def run_phase4(args):
    """Phase 4: 학습 Loss 시각화 및 검증 결과 종합 리포트"""
    logger.info("=" * 60)
    logger.info("  Phase 4: Visualization & Report")
    logger.info("=" * 60)

    output_dir = Path(args.output_base) / "phase4_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for Phase 4")
        return False

    success = True

    # 4-1. Loss 곡선 시각화
    if args.training_log_path and Path(args.training_log_path).exists():
        _visualize_training_loss(args.training_log_path, output_dir)
    else:
        logger.warning("Training log not found, skipping loss visualization")

    # 4-2. Phase 1~3 결과 종합
    _generate_summary_report(args.output_base, output_dir)

    # 4-3. Phase 2 파라미터별 비교 그리드
    _create_phase2_comparison_grid(args.output_base, output_dir)

    # 4-4. v1 vs v4 생성 이미지 비교 (v1 결과 디렉토리가 지정된 경우)
    v1_output_dir = getattr(args, "v1_output_dir", None)
    if v1_output_dir and Path(v1_output_dir).exists():
        _compare_v1_v4_generations(
            v1_dir=Path(v1_output_dir),
            v4_dir=Path(args.output_base),
            report_dir=output_dir,
        )
    else:
        logger.info("  v1 output dir not specified, skipping v1/v4 comparison")

    logger.info(f"Phase 4 complete: reports saved to {output_dir}")
    return success


def _visualize_training_loss(log_path: str, output_dir: Path):
    """training_log.json 기반 Loss 곡선과 분포를 시각화합니다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(log_path) as f:
        log_data = json.load(f)

    steps = [entry["step"] for entry in log_data]
    losses = [entry["loss"] for entry in log_data]
    lrs = [entry["lr"] for entry in log_data]
    epochs = [entry["epoch"] for entry in log_data]

    # --- Figure 1: Loss 추이 + 이동 평균 ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Loss 추이
    ax1 = axes[0]
    ax1.plot(steps, losses, "b-", alpha=0.4, linewidth=0.8, label="Loss (raw)")

    # 10-step 이동 평균
    window = 10
    if len(losses) >= window:
        moving_avg = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(losses[start:i + 1]))
        ax1.plot(steps, moving_avg, "r-", linewidth=2.0,
                 label=f"Moving Average ({window}-step)")

    # Warmup 구간 표시
    warmup_end_step = 50
    ax1.axvline(x=warmup_end_step, color="green", linestyle="--",
                alpha=0.7, label=f"Warmup End (step {warmup_end_step})")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LR 추이 (secondary axis)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, lrs, "g--", alpha=0.4, linewidth=0.8, label="Learning Rate")
    ax1_twin.set_ylabel("Learning Rate", color="green")
    ax1_twin.tick_params(axis="y", labelcolor="green")

    # Loss 분포 히스토그램
    ax2 = axes[1]
    ax2.hist(losses, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=np.mean(losses), color="red", linestyle="--",
                label=f"Mean: {np.mean(losses):.4f}")
    ax2.axvline(x=np.median(losses), color="orange", linestyle="--",
                label=f"Median: {np.median(losses):.4f}")
    ax2.set_xlabel("Loss")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Loss Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_dir / 'training_loss_curve.png'}")

    # --- Figure 2: Epoch별 평균 Loss ---
    epoch_losses = {}
    for entry in log_data:
        ep = entry["epoch"]
        if ep not in epoch_losses:
            epoch_losses[ep] = []
        epoch_losses[ep].append(entry["loss"])

    epoch_nums = sorted(epoch_losses.keys())
    epoch_avg = [np.mean(epoch_losses[ep]) for ep in epoch_nums]

    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.bar(epoch_nums, epoch_avg, color="steelblue", alpha=0.7, width=0.8)
    ax.plot(epoch_nums, epoch_avg, "r-", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Average Loss per Epoch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / "epoch_avg_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved: {output_dir / 'epoch_avg_loss.png'}")

    # 통계 요약
    stats = {
        "total_steps": len(log_data),
        "total_epochs": max(epochs) + 1 if epochs else 0,
        "loss_mean": float(np.mean(losses)),
        "loss_median": float(np.median(losses)),
        "loss_std": float(np.std(losses)),
        "loss_min": float(np.min(losses)),
        "loss_min_step": steps[int(np.argmin(losses))],
        "loss_max": float(np.max(losses)),
        "loss_max_step": steps[int(np.argmax(losses))],
        "final_loss": losses[-1] if losses else None,
        "nan_count": sum(1 for l in losses if np.isnan(l)),
    }

    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved: {stats_path}")

    # 콘솔 출력
    logger.info(f"  Training Statistics:")
    logger.info(f"    Total Steps: {stats['total_steps'] * 10} "
                f"(logged every 10 steps)")
    logger.info(f"    Loss Mean:   {stats['loss_mean']:.4f}")
    logger.info(f"    Loss Min:    {stats['loss_min']:.4f} "
                f"(step {stats['loss_min_step']})")
    logger.info(f"    Loss Max:    {stats['loss_max']:.4f} "
                f"(step {stats['loss_max_step']})")
    logger.info(f"    Final Loss:  {stats['final_loss']:.4f}")
    logger.info(f"    NaN Count:   {stats['nan_count']}")


def _generate_summary_report(output_base: str, report_dir: Path):
    """Phase 1~3 결과를 종합하는 텍스트 리포트를 생성합니다."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  ControlNet Model Validation Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    base = Path(output_base)

    # Phase 1 결과
    p1_summary = base / "phase1_basic" / "generation_summary.json"
    if p1_summary.exists():
        with open(p1_summary) as f:
            p1 = json.load(f)
        report_lines.append("Phase 1: Basic Generation")
        report_lines.append("-" * 40)
        report_lines.append(f"  Samples: {p1['generated']}/{p1['total_samples']}")
        report_lines.append(f"  Model: {p1.get('model_path', 'N/A')}")
        report_lines.append(f"  Steps: {p1.get('num_inference_steps', 'N/A')}")
        report_lines.append(f"  Guidance: {p1.get('guidance_scale', 'N/A')}")

        # 품질 평가 결과
        q1 = p1.get("quality", {})
        if q1.get("num_evaluated", 0) > 0:
            report_lines.append(f"  Quality Score: {q1['avg_quality_score']:.4f}")
            report_lines.append(f"    Color Consistency: {q1['avg_color_score']:.4f}")
            report_lines.append(f"    Artifact Score: {q1['avg_artifact_score']:.4f}")
            report_lines.append(f"    Sharpness Score: {q1['avg_blur_score']:.4f}")
            if q1.get("avg_ssim_score") is not None:
                report_lines.append(f"    SSIM Score: {q1['avg_ssim_score']:.4f}"
                                    f"  ({q1.get('num_ssim_evaluated', 0)} samples)")
            if q1.get("avg_lpips_score") is not None:
                report_lines.append(f"    LPIPS Score: {q1['avg_lpips_score']:.4f}"
                                    f"  ({q1.get('num_lpips_evaluated', 0)} samples)")
            report_lines.append(f"    Images Evaluated: {q1['num_evaluated']}")
            q_status = "PASS" if q1["quality_passed"] else "FAIL"
            report_lines.append(f"  Quality Status: {q_status}")
        else:
            report_lines.append(f"  Quality: Not evaluated")

        gen_ok = p1['generated'] > 0
        q_ok = q1.get("quality_passed", True) if q1.get("num_evaluated", 0) > 0 else gen_ok
        overall = "PASS" if (gen_ok and q_ok) else "FAIL"
        report_lines.append(f"  Overall Status: {overall}")
        report_lines.append("")
    else:
        report_lines.append("Phase 1: Not executed")
        report_lines.append("")

    # Phase 2 결과
    p2_summary = base / "phase2_summary.json"
    if p2_summary.exists():
        with open(p2_summary) as f:
            p2 = json.load(f)
        report_lines.append("Phase 2: Parameter Exploration")
        report_lines.append("-" * 40)
        report_lines.append(f"  Guidance scales: {p2.get('guidance_scales_tested', [])}")
        report_lines.append(f"  Inference steps: {p2.get('inference_steps_tested', [])}")
        report_lines.append(f"  Seeds: {p2.get('seeds_tested', [])}")
        report_lines.append(f"  Combinations tested: {len(p2.get('results', []))}")

        # 파라미터별 품질 점수 표시
        results_with_quality = [
            r for r in p2.get("results", [])
            if r.get("quality", {}).get("num_evaluated", 0) > 0
        ]
        if results_with_quality:
            report_lines.append("  Quality Scores by Parameter:")
            for r in results_with_quality:
                q = r["quality"]
                score_str = f"{q['avg_quality_score']:.4f}"
                ssim_str = (f"  SSIM={q['avg_ssim_score']:.4f}"
                            if q.get('avg_ssim_score') is not None else "")
                lpips_str = (f"  LPIPS={q['avg_lpips_score']:.4f}"
                             if q.get('avg_lpips_score') is not None else "")
                report_lines.append(
                    f"    {r['param']}={r['value']}: "
                    f"quality={score_str}{ssim_str}{lpips_str}"
                )

        # 최적 조합 표시
        best = p2.get("best_combination")
        if best:
            report_lines.append(f"  Best: {best['param']}={best['value']} "
                                f"(score={best['quality_score']:.4f})")

        report_lines.append("")
    else:
        report_lines.append("Phase 2: Not executed")
        report_lines.append("")

    # Phase 3 결과
    p3_summary = base / "phase3_unseen" / "generation_summary.json"
    if p3_summary.exists():
        with open(p3_summary) as f:
            p3 = json.load(f)
        report_lines.append("Phase 3: Unseen Data Generalization")
        report_lines.append("-" * 40)
        report_lines.append(f"  Unseen samples: {p3['generated']}/{p3['total_samples']}")

        # 품질 평가 결과
        q3 = p3.get("quality", {})
        if q3.get("num_evaluated", 0) > 0:
            report_lines.append(f"  Quality Score: {q3['avg_quality_score']:.4f}")
            report_lines.append(f"    Color Consistency: {q3['avg_color_score']:.4f}")
            report_lines.append(f"    Artifact Score: {q3['avg_artifact_score']:.4f}")
            report_lines.append(f"    Sharpness Score: {q3['avg_blur_score']:.4f}")
            if q3.get("avg_ssim_score") is not None:
                report_lines.append(f"    SSIM Score: {q3['avg_ssim_score']:.4f}"
                                    f"  ({q3.get('num_ssim_evaluated', 0)} samples)")
            if q3.get("avg_lpips_score") is not None:
                report_lines.append(f"    LPIPS Score: {q3['avg_lpips_score']:.4f}"
                                    f"  ({q3.get('num_lpips_evaluated', 0)} samples)")
            report_lines.append(f"    Images Evaluated: {q3['num_evaluated']}")
            q_status = "PASS" if q3["quality_passed"] else "FAIL"
            report_lines.append(f"  Quality Status: {q_status}")
        else:
            report_lines.append(f"  Quality: Not evaluated")

        gen_ok = p3['generated'] > 0
        q_ok = q3.get("quality_passed", True) if q3.get("num_evaluated", 0) > 0 else gen_ok
        overall = "PASS" if (gen_ok and q_ok) else "FAIL"
        report_lines.append(f"  Overall Status: {overall}")
        report_lines.append("")
    else:
        report_lines.append("Phase 3: Not executed")
        report_lines.append("")

    # 저장
    report_text = "\n".join(report_lines)
    report_path = report_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"  Summary report saved: {report_path}")
    logger.info("\n" + report_text)


def _create_phase2_comparison_grid(output_base: str, report_dir: Path):
    """Phase 2의 파라미터별 생성 결과를 하나의 비교 그리드로 결합합니다."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("PIL not available, skipping Phase 2 comparison grid")
        return

    base = Path(output_base)
    images_with_labels = []

    # guidance_scale 비교
    for gs in PHASE2_GUIDANCE_SCALES:
        gs_dir = base / f"phase2_gs{gs}"
        if not gs_dir.exists():
            continue
        # 첫 번째 생성 이미지 찾기
        for img_file in sorted(gs_dir.glob("**/*_generated_*.png")):
            images_with_labels.append((img_file, f"gs={gs}"))
            break
        for img_file in sorted(gs_dir.glob("**/*_gen0.png")):
            images_with_labels.append((img_file, f"gs={gs}"))
            break

    if not images_with_labels:
        logger.info("  No Phase 2 images found for comparison grid")
        return

    # 그리드 생성
    imgs = []
    labels = []
    for img_path, label in images_with_labels:
        try:
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)
            labels.append(label)
        except Exception:
            pass

    if not imgs:
        return

    # 동일 크기로 리사이즈
    target_h = min(img.height for img in imgs)
    target_w = min(img.width for img in imgs)
    resized = [img.resize((target_w, target_h), Image.LANCZOS) for img in imgs]

    label_h = 30
    n = len(resized)
    grid_w = target_w * n
    grid_h = target_h + label_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(resized, labels)):
        x_offset = i * target_w
        grid.paste(img, (x_offset, label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (target_w - text_w) // 2
        draw.text((text_x, 5), label, fill=(0, 0, 0), font=font)

    grid_path = report_dir / "phase2_parameter_comparison.png"
    grid.save(grid_path)
    logger.info(f"  Phase 2 comparison grid saved: {grid_path}")


def _compare_v1_v4_generations(
    v1_dir: Path,
    v4_dir: Path,
    report_dir: Path,
):
    """v1과 v4 모델의 생성 결과를 시각적/정량적으로 비교합니다.

    동일한 ROI(hint)에 대해 v1, v4 각각의 생성 이미지를 나란히 비교하고,
    SSIM/품질 메트릭 차이를 리포트합니다.

    Args:
        v1_dir: v1 모델 검증 결과 디렉토리 (phase1_basic/generated/ 포함)
        v4_dir: v4 모델 검증 결과 base 디렉토리
        report_dir: 비교 결과 저장 디렉토리
    """
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("cv2/PIL not available, skipping v1/v4 comparison")
        return

    # v1, v4 생성 이미지 수집
    v1_generated = {}
    v4_generated = {}

    # v1 이미지 수집
    for pattern in ["phase1_basic/generated/*.png", "generated/*.png", "*.png"]:
        for img_path in v1_dir.glob(pattern):
            # base_id 추출 (예: d84ad849e_class1_region1)
            base_id = _extract_base_id(img_path.stem)
            if base_id and base_id not in v1_generated:
                v1_generated[base_id] = img_path

    # v4 이미지 수집
    v4_base = Path(v4_dir)
    for pattern in ["phase1_basic/generated/*.png", "generated/*.png"]:
        for img_path in v4_base.glob(pattern):
            base_id = _extract_base_id(img_path.stem)
            if base_id and base_id not in v4_generated:
                v4_generated[base_id] = img_path

    # 공통 base_id 찾기
    common_ids = set(v1_generated.keys()) & set(v4_generated.keys())
    logger.info(f"  v1/v4 comparison: {len(common_ids)} common ROIs found")
    logger.info(f"    v1: {len(v1_generated)} images, v4: {len(v4_generated)} images")

    if not common_ids:
        logger.info("  No matching images for v1/v4 comparison")
        return

    # 비교 결과 생성
    comparison_results = []

    for base_id in sorted(common_ids):
        v1_img = cv2.imread(str(v1_generated[base_id]))
        v4_img = cv2.imread(str(v4_generated[base_id]))

        if v1_img is None or v4_img is None:
            continue

        # 크기 맞추기
        if v1_img.shape[:2] != v4_img.shape[:2]:
            v1_img = cv2.resize(
                v1_img, (v4_img.shape[1], v4_img.shape[0])
            )

        v1_gray = cv2.cvtColor(v1_img, cv2.COLOR_BGR2GRAY)
        v4_gray = cv2.cvtColor(v4_img, cv2.COLOR_BGR2GRAY)

        # 품질 비교
        v1_color = _score_color_consistency(
            cv2.cvtColor(v1_img, cv2.COLOR_BGR2RGB)
        )
        v4_color = _score_color_consistency(
            cv2.cvtColor(v4_img, cv2.COLOR_BGR2RGB)
        )
        v1_artifact = _score_artifacts(v1_gray)
        v4_artifact = _score_artifacts(v4_gray)
        v1_blur = _score_sharpness(v1_gray)
        v4_blur = _score_sharpness(v4_gray)

        # RGB 채널 간 편차 비교 (grayscale 개선 효과 측정)
        v1_rgb_std = float(np.std(np.std(v1_img.astype(float), axis=2)))
        v4_rgb_std = float(np.std(np.std(v4_img.astype(float), axis=2)))

        comparison_results.append({
            "base_id": base_id,
            "v1_color": round(v1_color, 4),
            "v4_color": round(v4_color, 4),
            "v1_artifact": round(v1_artifact, 4),
            "v4_artifact": round(v4_artifact, 4),
            "v1_blur": round(v1_blur, 4),
            "v4_blur": round(v4_blur, 4),
            "v1_rgb_std": round(v1_rgb_std, 4),
            "v4_rgb_std": round(v4_rgb_std, 4),
        })

    if not comparison_results:
        return

    # 통계 요약
    summary = {
        "num_compared": len(comparison_results),
        "avg_v1_color": round(np.mean([r["v1_color"] for r in comparison_results]), 4),
        "avg_v4_color": round(np.mean([r["v4_color"] for r in comparison_results]), 4),
        "avg_v1_artifact": round(np.mean([r["v1_artifact"] for r in comparison_results]), 4),
        "avg_v4_artifact": round(np.mean([r["v4_artifact"] for r in comparison_results]), 4),
        "avg_v1_blur": round(np.mean([r["v1_blur"] for r in comparison_results]), 4),
        "avg_v4_blur": round(np.mean([r["v4_blur"] for r in comparison_results]), 4),
        "avg_v1_rgb_std": round(np.mean([r["v1_rgb_std"] for r in comparison_results]), 4),
        "avg_v4_rgb_std": round(np.mean([r["v4_rgb_std"] for r in comparison_results]), 4),
        "details": comparison_results,
    }

    # JSON 저장
    comparison_path = report_dir / "v1_v4_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  v1/v4 comparison saved: {comparison_path}")

    # 비교 이미지 그리드 생성 (최대 8개)
    grid_count = min(8, len(common_ids))
    grid_ids = sorted(common_ids)[:grid_count]

    try:
        imgs_v1 = []
        imgs_v4 = []
        for bid in grid_ids:
            img1 = Image.open(v1_generated[bid]).convert("RGB")
            img4 = Image.open(v4_generated[bid]).convert("RGB")
            imgs_v1.append(img1)
            imgs_v4.append(img4)

        if imgs_v1:
            target_h = min(min(i.height for i in imgs_v1),
                           min(i.height for i in imgs_v4))
            target_w = min(min(i.width for i in imgs_v1),
                           min(i.width for i in imgs_v4))

            label_h = 30
            n = len(imgs_v1)
            grid_w = target_w * n
            grid_h = (target_h + label_h) * 2  # v1 위, v4 아래

            grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
            draw = ImageDraw.Draw(grid)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except (IOError, OSError):
                font = ImageFont.load_default()

            for i, (img1, img4, bid) in enumerate(
                zip(imgs_v1, imgs_v4, grid_ids)
            ):
                x = i * target_w

                # v1 (상단)
                r1 = img1.resize((target_w, target_h), Image.LANCZOS)
                grid.paste(r1, (x, label_h))
                draw.text((x + 2, 2), f"v1: {bid[:12]}", fill=(0, 0, 255),
                          font=font)

                # v4 (하단)
                r4 = img4.resize((target_w, target_h), Image.LANCZOS)
                grid.paste(r4, (x, target_h + label_h * 2))
                draw.text((x + 2, target_h + label_h + 2),
                          f"v4: {bid[:12]}", fill=(255, 0, 0), font=font)

            grid_path = report_dir / "v1_v4_comparison_grid.png"
            grid.save(grid_path)
            logger.info(f"  v1/v4 comparison grid saved: {grid_path}")
    except Exception as e:
        logger.warning(f"  v1/v4 grid creation failed: {e}")

    # 로그 출력
    logger.info(f"  v1/v4 Comparison Summary ({summary['num_compared']} images):")
    logger.info(f"    Color:    v1={summary['avg_v1_color']:.4f}  "
                f"v4={summary['avg_v4_color']:.4f}")
    logger.info(f"    Artifact: v1={summary['avg_v1_artifact']:.4f}  "
                f"v4={summary['avg_v4_artifact']:.4f}")
    logger.info(f"    Blur:     v1={summary['avg_v1_blur']:.4f}  "
                f"v4={summary['avg_v4_blur']:.4f}")
    logger.info(f"    RGB Std:  v1={summary['avg_v1_rgb_std']:.4f}  "
                f"v4={summary['avg_v4_rgb_std']:.4f}  "
                f"(lower = more grayscale)")


def _extract_base_id(stem: str) -> Optional[str]:
    """파일명 stem에서 base_id를 추출합니다.

    예: 'd84ad849e_class1_region1_gen0' → 'd84ad849e_class1_region1'
        '8f8a23f39.jpg_class1_region1_gen0' → '8f8a23f39.jpg_class1_region1'
    """
    import re
    # _gen{N}, _generated_{N} 패턴 제거
    cleaned = re.sub(r"_gen\d+$", "", stem)
    cleaned = re.sub(r"_generated_\d+$", "", cleaned)
    return cleaned if cleaned else None


# =============================================================================
# Utility
# =============================================================================

def _model_source_args(args) -> List[str]:
    """model_path 또는 pipeline_path를 test_controlnet.py CLI 인자로 변환합니다.

    우선순위: model_path > pipeline_path
    저장 최적화(--skip_save_pipeline) 적용 후에는 pipeline이 없으므로
    model_path가 기본이 됩니다.
    """
    if args.model_path:
        return ["--model_path", args.model_path]
    elif args.pipeline_path:
        return ["--pipeline_path", args.pipeline_path]
    else:
        raise ValueError("Either --model_path or --pipeline_path must be specified")


def _get_first_hint(jsonl_path: str) -> Optional[Dict]:
    """JSONL에서 첫 번째 샘플의 hint 경로와 프롬프트를 반환합니다."""
    jsonl = Path(jsonl_path)
    if not jsonl.exists():
        return None

    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            hint_path = sample.get("hint", "")

            # hint 경로 resolve 시도
            resolved = None
            data_dir = jsonl.parent
            search_dirs = [str(data_dir), str(data_dir.parent)]

            path = Path(hint_path)
            if path.is_absolute() and path.exists():
                resolved = str(path)
            else:
                for base in search_dirs:
                    candidate = Path(base) / hint_path
                    if candidate.exists():
                        resolved = str(candidate)
                        break
                    candidate = Path(base) / path.name
                    if candidate.exists():
                        resolved = str(candidate)
                        break
                    for subdir in ["hints", "images"]:
                        candidate = Path(base) / subdir / path.name
                        if candidate.exists():
                            resolved = str(candidate)
                            break

            if resolved is None:
                resolved = hint_path  # Colab에서 resolve 될 수 있으므로 원본 유지

            return {
                "hint_path": resolved,
                "prompt": sample.get("prompt", "a surface defect on metal surface"),
            }

    return None


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ControlNet Model Validation - Multi-Phase Runner"
    )

    # Model
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained ControlNet model directory (best_model/ or final_model/). "
             "Recommended: 저장 최적화 후에는 이 옵션을 사용합니다.",
    )
    parser.add_argument(
        "--pipeline_path", type=str, default=None,
        help="Path to saved full pipeline directory (저장 최적화 이전 모델용)",
    )

    # Data
    parser.add_argument(
        "--jsonl_path", type=str, required=True,
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root directory for image resolution",
    )
    parser.add_argument(
        "--roi_metadata_path", type=str, default=None,
        help="Path to roi_metadata.csv (required for Phase 3)",
    )
    parser.add_argument(
        "--training_log_path", type=str, default=None,
        help="Path to training_log.json (required for Phase 4 loss visualization)",
    )

    # Output
    parser.add_argument(
        "--output_base", type=str, default="test_results",
        help="Base output directory for all phases",
    )

    # Phase selection
    parser.add_argument(
        "--phases", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Phases to run (default: 1 2 3 4)",
    )

    # Quality evaluation
    parser.add_argument(
        "--quality_threshold", type=float, default=0.5,
        help="Minimum average quality score for Phase 1/3 to PASS (default: 0.5). "
             "Score range: 0.0~1.0. Evaluated on color_consistency (0.4), "
             "artifact_detection (0.3), sharpness (0.3).",
    )

    # Generation parameters (passed through to test_controlnet.py)
    parser.add_argument(
        "--controlnet_conditioning_scale", type=float, default=0.7,
        help="ControlNet conditioning scale for inference (0.0~1.0). "
             "v3부터 기본값 0.7 적용 (v2의 1.0에서 neon/rainbow 패턴 발생). "
             "Passed through to test_controlnet.py.",
    )

    # Class imbalance
    parser.add_argument(
        "--oversample_minority", action="store_true", default=False,
        help="Phase 1에서 소수 클래스(Class 3/4)를 오버샘플링하여 "
             "클래스 균형을 맞춘 JSONL을 생성합니다. "
             "seed 변경을 통해 다양성을 확보합니다.",
    )

    # v1 vs v4 comparison
    parser.add_argument(
        "--v1_output_dir", type=str, default=None,
        help="v1 모델 검증 결과 디렉토리. Phase 4에서 v1 vs v4 "
             "생성 이미지 비교를 수행합니다. "
             "예: /content/drive/.../test_results_v1",
    )

    # Resolution
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="생성 해상도 (정사각형). 지정 시 resolution×resolution 고정 해상도 사용. "
             "미지정 시 hint 이미지 크기에서 8의 배수로 올림 자동 계산 (최소 64px). "
             "Passed through to test_controlnet.py. 예: --resolution 512",
    )

    # Parallelization
    parser.add_argument(
        "--workers", type=int, default=0,
        help="품질 평가 병렬 워커 수 (기본 0 = 순차 처리, -1 = CPU 코어 수 자동 감지, "
             "N >= 2 = N개 프로세스 병렬 처리). LPIPS 제외 CPU-bound 메트릭만 병렬화.",
    )

    args = parser.parse_args()

    # 워커 수 결정
    if args.workers < 0:
        cpu_count = os.cpu_count() or 1
        args.workers = max(1, cpu_count - 1)
        logger.info(f"워커 수 자동 설정: {args.workers} (CPU: {cpu_count})")

    # Validation: model_path 또는 pipeline_path 중 하나는 필수
    if not args.model_path and not args.pipeline_path:
        parser.error("Either --model_path or --pipeline_path must be specified. "
                     "--model_path is recommended for optimized models.")

    # model_path와 pipeline_path 동시 지정 시 model_path 우선 (경고 출력)
    if args.model_path and args.pipeline_path:
        logger.warning(
            "Both --model_path and --pipeline_path specified. "
            "Using --model_path (recommended)."
        )

    # Script directory
    args.script_dir = Path(__file__).parent

    return args


def main():
    args = parse_args()

    model_display = args.model_path or args.pipeline_path
    model_type = "model_path" if args.model_path else "pipeline_path"

    logger.info("=" * 70)
    logger.info("  ControlNet Model Validation - Multi-Phase Runner")
    logger.info("=" * 70)
    logger.info(f"  Phases to run: {args.phases}")
    logger.info(f"  Output base: {args.output_base}")
    logger.info(f"  Model ({model_type}): {model_display}")
    logger.info("")

    Path(args.output_base).mkdir(parents=True, exist_ok=True)

    results = {}

    if 1 in args.phases:
        results["phase1"] = run_phase1(args)

    if 2 in args.phases:
        results["phase2"] = run_phase2(args)

    if 3 in args.phases:
        results["phase3"] = run_phase3(args)

    if 4 in args.phases:
        results["phase4"] = run_phase4(args)

    # 최종 결과
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Validation Complete")
    logger.info("=" * 70)
    for phase, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {phase}: {status}")
    logger.info("=" * 70)

    # 전체 결과 저장
    final_summary = Path(args.output_base) / "validation_results.json"
    with open(final_summary, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {final_summary}")


if __name__ == "__main__":
    main()
