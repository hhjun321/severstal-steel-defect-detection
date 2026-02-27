"""
ControlNet Inference Script for Steel Defect Generation
========================================================

학습된 ControlNet 모델을 사용하여 hint 이미지로부터 결함 이미지를 생성합니다.

Usage:
    # 단일 hint 이미지로 생성
    python scripts/test_controlnet.py \\
        --model_path controlnet_training/final_model \\
        --hint_image data/processed/controlnet_dataset/hints/0002cc93b.jpg_class1_region0_hint.png \\
        --prompt "a general surface defect on complex patterned metal surface, steel defect class 1"

    # train.jsonl의 모든 샘플로 생성
    python scripts/test_controlnet.py \\
        --model_path controlnet_training/final_model \\
        --jsonl_path data/processed/controlnet_dataset/train.jsonl \\
        --output_dir outputs/test_results

    # 파이프라인 디렉토리에서 직접 로드
    python scripts/test_controlnet.py \\
        --pipeline_path controlnet_training/pipeline \\
        --jsonl_path data/processed/controlnet_dataset/train.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# class_id를 prompt에서 추출하는 정규식
import re
_CLASS_PATTERN = re.compile(r'class\s*(\d+)', re.IGNORECASE)


# =============================================================================
# Path Resolution (학습 스크립트와 동일)
# =============================================================================

def resolve_path(path_str: str, search_dirs: list) -> Path:
    """여러 디렉토리에서 파일을 탐색합니다.

    Colab 환경 호환성을 위해 다양한 경로 조합을 시도합니다:
    1. 절대 경로 직접 확인
    2. search_dirs 기준 전체 경로 조합
    3. search_dirs 기준 파일명만으로 탐색
    4. search_dirs의 하위 디렉토리(hints/, images/, masks/) 탐색
    5. 경로의 suffix 부분 매칭 (예: hints/xxx.png)
    """
    path = Path(path_str)

    # 절대 경로로 존재
    if path.is_absolute() and path.exists():
        return path

    filename = path.name
    # 경로의 마지막 2단계 추출 (예: "hints/xxx_hint.png")
    path_parts = Path(path_str).parts
    suffix_candidates = []
    for i in range(len(path_parts) - 1, 0, -1):
        suffix_candidates.append(str(Path(*path_parts[i:])))

    # search_dirs에서 탐색
    for base_dir in search_dirs:
        if base_dir is None:
            continue
        base = Path(base_dir)
        # 전체 경로 시도
        candidate = base / path_str
        if candidate.exists():
            return candidate
        # 파일명만으로 시도
        candidate = base / filename
        if candidate.exists():
            return candidate
        # 하위 디렉토리에서 파일명 탐색
        for subdir in ["hints", "images", "masks"]:
            candidate = base / subdir / filename
            if candidate.exists():
                return candidate
        # suffix 부분 경로 매칭 (예: base / "hints/xxx.png")
        for suffix in suffix_candidates:
            candidate = base / suffix
            if candidate.exists():
                return candidate

    # 상대 경로
    if path.exists():
        return path

    raise FileNotFoundError(f"Cannot find: '{path_str}'")


# =============================================================================
# Pipeline Loading
# =============================================================================

def load_pipeline(args, device):
    """ControlNet 파이프라인을 로드합니다.

    로드 우선순위:
    1. --pipeline_path: 전체 파이프라인 디렉토리에서 직접 로드
    2. --model_path: ControlNet 가중치만 로드 + base SD model 조합 (권장)
       - fp16 저장된 모델도 자동 감지하여 로드

    --model_path 방식이 디스크 절약에 유리합니다.
    SD base model은 HuggingFace 캐시에서 로드됩니다.
    """

    if args.pipeline_path and Path(args.pipeline_path).exists():
        # 저장된 전체 파이프라인에서 로드
        logger.info(f"Loading full pipeline from: {args.pipeline_path}")
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pipeline_path,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
    else:
        # ControlNet 가중치 + SD base model 조합 (권장 방식)
        model_path = args.model_path
        if not model_path:
            raise ValueError("--model_path or --pipeline_path must be specified")

        logger.info(f"Loading ControlNet from: {model_path}")

        # pipeline_reference.json이 있으면 base model 정보 자동 참조
        ref_path = Path(model_path).parent / "pipeline_reference.json"
        base_model = args.pretrained_model_name_or_path
        if ref_path.exists():
            import json as _json
            with open(ref_path) as f:
                ref = _json.load(f)
            if "base_model" in ref:
                base_model = ref["base_model"]
                logger.info(f"Using base model from pipeline_reference: {base_model}")

        # fp16/fp32 모델 자동 감지 로드
        try:
            controlnet = ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
        except Exception as e:
            logger.warning(f"Standard load failed ({e}), trying with safetensors...")
            controlnet = ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )

        logger.info(f"Loading base SD model: {base_model}")
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
        )

    # Scheduler 설정 (빠른 추론)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # 메모리 최적화
    if device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception:
            logger.info("xformers not available, using default attention")

    logger.info("Pipeline loaded successfully")
    return pipeline


# =============================================================================
# Image Generation
# =============================================================================

def generate_single(
    pipeline, hint_image, prompt, negative_prompt,
    num_inference_steps, guidance_scale, seed, device,
    num_images=1,
    controlnet_conditioning_scale=0.7,
    grayscale_postprocess=True,
    resolution=None,
):
    """단일 hint 이미지로부터 결함 이미지를 생성합니다.

    Args:
        grayscale_postprocess: True이면 생성 이미지를 grayscale 변환 후
            RGB 3채널로 복제합니다. SD 1.5 VAE의 3채널 독립 디코딩으로 인한
            RGB 컬러 아티팩트를 제거합니다.
        resolution: 생성 해상도. 지정 시 resolution×resolution 고정 해상도 사용.
            None이면 hint 이미지 크기에서 8의 배수로 올림 계산 (최소 64px).
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    # hint 이미지가 PIL Image인지 확인
    if not isinstance(hint_image, Image.Image):
        hint_image = Image.open(hint_image).convert("RGB")

    # 생성 해상도 결정
    if resolution is not None:
        # 고정 해상도 (예: 512x512)
        gen_h, gen_w = resolution, resolution
        logger.info(f"Using fixed resolution: {gen_w}x{gen_h}")
    else:
        # hint 이미지 크기 기반 자동 계산 (8의 배수, 최소 64px)
        orig_w, orig_h = hint_image.size
        gen_h = max(64, ((orig_h + 7) // 8) * 8)
        gen_w = max(64, ((orig_w + 7) // 8) * 8)
        logger.info(
            f"Auto resolution from hint ({orig_w}x{orig_h}): {gen_w}x{gen_h}"
        )

    results = []
    for i in range(num_images):
        gen = torch.Generator(device=device).manual_seed(seed + i)

        with torch.autocast(str(device)):
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=hint_image,
                height=gen_h,
                width=gen_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=gen,
            )

        image = output.images[0]

        # Grayscale 후처리: RGB 컬러 아티팩트 제거
        # SD 1.5 VAE가 3채널을 독립적으로 디코딩하여 발생하는
        # 비현실적 RGB 색상을 grayscale 변환으로 제거합니다.
        if grayscale_postprocess:
            image = image.convert("L").convert("RGB")

        results.append(image)

    return results


def create_comparison_grid(
    hint_image, generated_images, original_image=None,
    prompt="", max_width=1536,
):
    """
    hint 이미지와 생성된 이미지들을 나란히 비교하는 그리드를 생성합니다.

    Layout: [Hint] [Generated 1] [Generated 2] ... [Original (있으면)]
    """
    images = [hint_image] + generated_images
    if original_image is not None:
        images.append(original_image)

    # 동일 크기로 리사이즈
    target_h = min(img.height for img in images)
    target_w = min(img.width for img in images)

    resized = []
    for img in images:
        resized.append(img.resize((target_w, target_h), Image.LANCZOS))

    # 라벨 영역 높이
    label_h = 30
    n = len(resized)
    grid_w = target_w * n
    grid_h = target_h + label_h

    # 최대 너비 제한
    if grid_w > max_width:
        scale = max_width / grid_w
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)
        resized = [img.resize((target_w, target_h), Image.LANCZOS) for img in resized]
        grid_w = target_w * n
        grid_h = target_h + label_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # 라벨 텍스트
    labels = ["Hint (Input)"]
    for i in range(len(generated_images)):
        labels.append(f"Generated {i+1}")
    if original_image is not None:
        labels.append("Original (GT)")

    for i, (img, label) in enumerate(zip(resized, labels)):
        x_offset = i * target_w
        grid.paste(img, (x_offset, label_h))

        # 라벨 그리기
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (target_w - text_w) // 2
        draw.text((text_x, 5), label, fill=(0, 0, 0), font=font)

    # 프롬프트를 하단에 표시 (선택)
    if prompt:
        short_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        # 프롬프트는 그리드 아래가 아닌 별도 저장

    return grid


# =============================================================================
# Batch Generation from JSONL
# =============================================================================

def generate_from_jsonl(pipeline, args, device):
    """train.jsonl의 모든 샘플에 대해 결함 이미지를 생성합니다.
    
    v5.2 개선: --num_images_per_class로 클래스별 생성 수 차등 적용.
    예: --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}'
    클래스별 생성 수가 지정되면 prompt에서 class_id를 추출하여
    해당 클래스의 생성 수를 적용합니다.
    """

    jsonl_path = Path(args.jsonl_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 서브 디렉토리
    generated_dir = output_dir / "generated"
    comparison_dir = output_dir / "comparisons"
    generated_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)

    # JSONL 로드
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples from {jsonl_path}")

    # v5.2: 클래스별 생성 수 파싱
    class_num_images = None
    if args.num_images_per_class:
        try:
            class_num_images = json.loads(args.num_images_per_class)
            # 키를 문자열로 통일
            class_num_images = {str(k): int(v) for k, v in class_num_images.items()}
            logger.info(f"Class-aware multi-generation: {class_num_images}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid --num_images_per_class format: {e}")
            logger.error("Expected JSON: '{\"1\":2,\"2\":10,\"3\":1,\"4\":2}'")
            class_num_images = None

    # 총 생성 수 예측 (로깅용)
    if class_num_images:
        total_expected = 0
        class_sample_counts = {}
        for sample in samples:
            prompt = sample.get("prompt", "")
            match = _CLASS_PATTERN.search(prompt)
            cls_id = match.group(1) if match else "unknown"
            class_sample_counts[cls_id] = class_sample_counts.get(cls_id, 0) + 1
        for cls_id, count in sorted(class_sample_counts.items()):
            n_imgs = class_num_images.get(cls_id, args.num_images_per_sample)
            total_expected += count * n_imgs
            logger.info(f"  Class {cls_id}: {count} samples x {n_imgs} images = {count * n_imgs}")
        logger.info(f"  Total expected: {total_expected} images")

    # 경로 탐색 디렉토리 목록
    project_root = Path(__file__).parent.parent
    data_dir = jsonl_path.parent
    search_dirs = [
        args.image_root,
        str(data_dir),
        str(data_dir.parent),
        str(project_root),
    ]

    results_summary = []

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        prompt = sample.get("prompt", "")
        negative_prompt = sample.get(
            "negative_prompt",
            "blurry, low quality, artifacts, noise, distorted"
        )
        hint_path_str = sample.get("hint", "")
        target_path_str = sample.get("target", "")

        # v5.2: 클래스별 생성 수 결정
        num_images = args.num_images_per_sample
        if class_num_images:
            match = _CLASS_PATTERN.search(prompt)
            if match:
                cls_id = match.group(1)
                num_images = class_num_images.get(cls_id, args.num_images_per_sample)

        # Hint 이미지 로드
        try:
            hint_path = resolve_path(hint_path_str, search_dirs)
            hint_image = Image.open(hint_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"[{idx}] Hint not found: {hint_path_str}, skipping")
            continue

        # Original 이미지 로드 (비교용, 없으면 None)
        original_image = None
        try:
            target_path = resolve_path(target_path_str, search_dirs)
            original_image = Image.open(target_path).convert("RGB")
        except FileNotFoundError:
            logger.info(f"[{idx}] Original not found (OK for comparison-only)")

        # 이미지 생성
        generated_images = generate_single(
            pipeline=pipeline,
            hint_image=hint_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + idx,
            device=device,
            num_images=num_images,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            grayscale_postprocess=args.grayscale_postprocess,
            resolution=args.resolution,
        )

        # 결과 저장
        sample_name = Path(hint_path_str).stem.replace("_hint", "")

        for j, gen_img in enumerate(generated_images):
            save_name = f"{sample_name}_gen{j}.png"
            gen_img.save(generated_dir / save_name)

        # 비교 그리드 생성
        grid = create_comparison_grid(
            hint_image=hint_image,
            generated_images=generated_images,
            original_image=original_image,
            prompt=prompt,
        )
        grid.save(comparison_dir / f"{sample_name}_comparison.png")

        results_summary.append({
            "index": idx,
            "sample_name": sample_name,
            "prompt": prompt,
            "hint_path": str(hint_path),
            "original_available": original_image is not None,
            "num_generated": len(generated_images),
        })

    # 결과 요약 저장
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_samples": len(samples),
            "generated": len(results_summary),
            "total_images": sum(r["num_generated"] for r in results_summary),
            "model_path": args.model_path or args.pipeline_path,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "class_num_images": class_num_images,
            "results": results_summary,
        }, f, indent=2)

    total_imgs = sum(r["num_generated"] for r in results_summary)
    logger.info(f"Generation complete: {len(results_summary)}/{len(samples)} samples, "
                f"{total_imgs} total images")
    logger.info(f"Generated images: {generated_dir}")
    logger.info(f"Comparison grids: {comparison_dir}")
    logger.info(f"Summary: {summary_path}")


# =============================================================================
# Single Image Generation
# =============================================================================

def generate_single_image(pipeline, args, device):
    """단일 hint 이미지로부터 결함 이미지를 생성합니다."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hint_image = Image.open(args.hint_image).convert("RGB")
    logger.info(f"Hint image: {args.hint_image} (size={hint_image.size})")

    prompt = args.prompt or "a surface defect on metal surface, steel defect"
    negative_prompt = args.negative_prompt or (
        "blurry, low quality, artifacts, noise, distorted, "
        "warped, unrealistic, oversaturated"
    )

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")

    generated_images = generate_single(
        pipeline=pipeline,
        hint_image=hint_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
        num_images=args.num_images_per_sample,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        grayscale_postprocess=args.grayscale_postprocess,
        resolution=args.resolution,
    )

    # 저장
    hint_stem = Path(args.hint_image).stem
    for i, img in enumerate(generated_images):
        save_path = output_dir / f"{hint_stem}_generated_{i}.png"
        img.save(save_path)
        logger.info(f"Saved: {save_path}")

    # hint 이미지를 sources/ 디렉토리에 참조용으로 저장
    # (run_validation_phases.py의 품질 평가에서 SSIM/LPIPS 참조로 사용)
    sources_dir = output_dir / "sources"
    sources_dir.mkdir(exist_ok=True)
    hint_image.save(sources_dir / f"{hint_stem}.png")
    logger.info(f"Source hint saved: {sources_dir / f'{hint_stem}.png'}")

    # 비교 그리드
    grid = create_comparison_grid(
        hint_image=hint_image,
        generated_images=generated_images,
        prompt=prompt,
    )
    grid_path = output_dir / f"{hint_stem}_comparison.png"
    grid.save(grid_path)
    logger.info(f"Comparison grid saved: {grid_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ControlNet for Steel Defect Generation"
    )

    # Model
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained ControlNet model directory "
             "(e.g., controlnet_training/final_model)",
    )
    parser.add_argument(
        "--pipeline_path", type=str, default=None,
        help="Path to saved full pipeline directory "
             "(e.g., controlnet_training/pipeline)",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="runwayml/stable-diffusion-v1-5",
    )

    # Input (둘 중 하나 지정)
    parser.add_argument(
        "--hint_image", type=str, default=None,
        help="Single hint image path for generation",
    )
    parser.add_argument(
        "--jsonl_path", type=str, default=None,
        help="Path to train.jsonl for batch generation",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root dir for source/target images (for comparison)",
    )

    # Generation
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--controlnet_conditioning_scale", type=float, default=0.7,
        help="ControlNet conditioning scale (0.0~1.0). "
             "낮을수록 SD의 자연 이미지 생성 능력이 보존됩니다. "
             "v3부터 기본값 0.7 적용 (v2의 1.0에서 neon/rainbow 패턴 발생).",
    )
    parser.add_argument("--num_images_per_sample", type=int, default=1)
    parser.add_argument(
        "--num_images_per_class", type=str, default=None,
        help='v5.2 class-aware multi-generation: 클래스별 생성 수를 JSON으로 지정. '
             'prompt에서 class_id를 파싱하여 해당 클래스의 생성 수를 적용합니다. '
             '--num_images_per_sample은 fallback으로 사용. '
             '예: --num_images_per_class \'{"1":2,"2":10,"3":1,"4":2}\'',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="생성 해상도 (정사각형). 학습 시 사용한 해상도와 반드시 일치해야 합니다. "
             "기본값 512 (학습 해상도와 동일). "
             "v4에서 None (hint 크기 기반 자동 계산)으로 설정 시 "
             "tiny hint에서 64x64 등 극소 해상도로 생성하여 품질 저하 발생. "
             "예: --resolution 512",
    )
    parser.add_argument(
        "--grayscale_postprocess", action="store_true", default=True,
        help="생성 이미지를 grayscale 변환 후 RGB 복제. "
             "SD 1.5 VAE의 3채널 독립 디코딩으로 인한 RGB 컬러 아티팩트를 제거합니다. "
             "기본 활성화.",
    )
    parser.add_argument(
        "--no_grayscale_postprocess", dest="grayscale_postprocess",
        action="store_false",
        help="Grayscale 후처리를 비활성화합니다.",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/test_results",
    )

    # Device
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Validation
    if not args.model_path and not args.pipeline_path:
        parser.error("Either --model_path or --pipeline_path must be specified")

    if not args.hint_image and not args.jsonl_path:
        parser.error("Either --hint_image or --jsonl_path must be specified")

    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("  ControlNet Inference - Steel Defect Generation")
    logger.info("=" * 60)

    # Load pipeline
    pipeline = load_pipeline(args, device)

    # Generate
    if args.jsonl_path:
        generate_from_jsonl(pipeline, args, device)
    elif args.hint_image:
        generate_single_image(pipeline, args, device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
