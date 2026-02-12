"""
ControlNet Training Script for Steel Defect Augmentation
=========================================================

Hugging Face diffusers 라이브러리를 사용하여 ControlNet을 학습합니다.
[마스크/힌트 → 결함 이미지] 매핑을 학습하는 파이프라인입니다.

Architecture:
    - Base Model: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
    - ControlNet: lllyasviel/sd-controlnet-canny (초기 가중치)
    - Conditioning: Multi-channel hint image (R=defect shape, G=structure, B=texture)
    - Text Prompt: Defect + background description

Usage:
    # 기본 학습 (Colab 환경, 이미지는 Google Drive에 있음)
    python scripts/train_controlnet.py \\
        --data_dir data/processed/controlnet_dataset \\
        --output_dir outputs/controlnet_training \\
        --mixed_precision fp16 \\
        --gradient_checkpointing

    # 학습 재개
    python scripts/train_controlnet.py \\
        --data_dir data/processed/controlnet_dataset \\
        --output_dir outputs/controlnet_training \\
        --resume_from_checkpoint latest
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Hugging Face imports
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class SteelDefectControlNetDataset(Dataset):
    """
    ControlNet 학습용 Steel Defect 데이터셋.

    train.jsonl에서 로드하며, 각 샘플은:
    - source/target: 결함이 포함된 ROI 이미지 (학습 대상)
    - hint: 3채널 conditioning 이미지 (R=mask, G=structure, B=texture)
    - prompt: 결함 + 배경 텍스트 설명
    - negative_prompt: 네거티브 프롬프트
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str = None,
        resolution: int = 512,
        tokenizer=None,
    ):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        self.jsonl_path = Path(jsonl_path)
        self.data_dir = self.jsonl_path.parent

        # Load training entries
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    self.samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] 범위
        ])

        self.conditioning_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),  # [0, 1] 범위
        ])

    def _resolve_image_path(self, path_str: str) -> Path:
        """
        이미지 경로를 해석합니다.

        경로 해석 우선순위:
        1. 절대 경로로 존재하면 그대로 사용 (Colab의 /content/drive/... 포함)
        2. image_root + 파일명
        3. data_dir 기준 상대 경로
        4. 프로젝트 루트 기준 상대 경로
        """
        path = Path(path_str)

        # 1. 절대 경로로 존재
        if path.is_absolute() and path.exists():
            return path

        filename = path.name

        # 2. image_root
        if self.image_root:
            candidate = self.image_root / filename
            if candidate.exists():
                return candidate

        # 3. data_dir 기준
        for base in [self.data_dir, self.data_dir.parent]:
            candidate = base / path_str
            if candidate.exists():
                return candidate

        # 4. 프로젝트 루트 기준
        project_root = Path(__file__).parent.parent
        candidate = project_root / path_str
        if candidate.exists():
            return candidate

        # 5. 상대 경로 그대로
        if path.exists():
            return path

        raise FileNotFoundError(
            f"Cannot resolve image path: '{path_str}'\n"
            f"  Tried: absolute, image_root='{self.image_root}', "
            f"data_dir='{self.data_dir}'"
        )

    def _resolve_hint_path(self, path_str: str) -> Path:
        """hint 이미지 경로를 해석합니다."""
        path = Path(path_str)

        if path.is_absolute() and path.exists():
            return path

        # 프로젝트 루트 기준
        project_root = Path(__file__).parent.parent
        candidate = project_root / path_str
        if candidate.exists():
            return candidate

        # data_dir 기준
        for candidate in [
            self.data_dir / Path(path_str).name,
            self.data_dir / "hints" / Path(path_str).name,
        ]:
            if candidate.exists():
                return candidate

        if path.exists():
            return path

        raise FileNotFoundError(f"Cannot resolve hint path: '{path_str}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load target image
        target_path = self._resolve_image_path(sample["target"])
        target_image = Image.open(target_path).convert("RGB")

        # Load hint/conditioning image
        hint_path = self._resolve_hint_path(sample["hint"])
        conditioning_image = Image.open(hint_path).convert("RGB")

        # Apply transforms
        target_tensor = self.image_transforms(target_image)
        conditioning_tensor = self.conditioning_transforms(conditioning_image)

        # Tokenize prompt
        prompt = sample.get("prompt", "")
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        else:
            input_ids = torch.tensor([0])

        return {
            "pixel_values": target_tensor,
            "conditioning_pixel_values": conditioning_tensor,
            "input_ids": input_ids,
            "prompt": prompt,
        }


def collate_fn(examples):
    """DataLoader용 collate function."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    conditioning_pixel_values = torch.stack([ex["conditioning_pixel_values"] for ex in examples])
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


# =============================================================================
# Sanity Check
# =============================================================================

def run_sanity_check(dataset, tokenizer, device, num_samples=2):
    """
    학습 전 데이터 로딩 및 수치 안정성을 검증합니다.

    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("=" * 60)
    logger.info("  Running Data Sanity Check")
    logger.info("=" * 60)

    all_ok = True
    check_count = min(num_samples, len(dataset))

    for i in range(check_count):
        try:
            sample = dataset[i]
            pv = sample["pixel_values"]
            cv = sample["conditioning_pixel_values"]
            ids = sample["input_ids"]

            # Shape check
            assert pv.dim() == 3, f"pixel_values should be 3D, got {pv.dim()}D"
            assert cv.dim() == 3, f"conditioning should be 3D, got {cv.dim()}D"
            assert pv.shape[0] == 3, f"pixel_values channels should be 3, got {pv.shape[0]}"
            assert cv.shape[0] == 3, f"conditioning channels should be 3, got {cv.shape[0]}"

            # NaN/Inf check
            assert not torch.isnan(pv).any(), "pixel_values contains NaN!"
            assert not torch.isinf(pv).any(), "pixel_values contains Inf!"
            assert not torch.isnan(cv).any(), "conditioning contains NaN!"
            assert not torch.isinf(cv).any(), "conditioning contains Inf!"

            # Range check
            pv_min, pv_max = pv.min().item(), pv.max().item()
            cv_min, cv_max = cv.min().item(), cv.max().item()

            logger.info(
                f"  Sample {i}: pixel_values shape={list(pv.shape)}, "
                f"range=[{pv_min:.3f}, {pv_max:.3f}] | "
                f"conditioning shape={list(cv.shape)}, "
                f"range=[{cv_min:.3f}, {cv_max:.3f}] | "
                f"input_ids shape={list(ids.shape)} | "
                f"prompt='{sample['prompt'][:60]}...'"
            )

            # 값이 모두 0인지 체크 (빈 이미지)
            if pv.abs().sum() < 1e-6:
                logger.warning(f"  [WARN] Sample {i}: pixel_values is all zeros!")
            if cv.abs().sum() < 1e-6:
                logger.warning(f"  [WARN] Sample {i}: conditioning is all zeros!")

        except FileNotFoundError as e:
            logger.error(f"  [FAIL] Sample {i}: {e}")
            all_ok = False
        except Exception as e:
            logger.error(f"  [FAIL] Sample {i}: {e}")
            all_ok = False

    if all_ok:
        logger.info("  Sanity check PASSED")
    else:
        logger.error("  Sanity check FAILED - fix data issues before training")

    logger.info("=" * 60)
    return all_ok


def run_training_sanity_check(
    controlnet, vae, text_encoder, unet, noise_scheduler,
    dataset, tokenizer, optimizer, scaler, device, weight_dtype, use_amp,
):
    """
    학습 초기 1 step forward/backward 완전 테스트.
    
    데이터 로딩 → VAE 인코딩 → ControlNet forward → UNet forward →
    Loss 계산 → Backward → Gradient 계산까지 전체 파이프라인을 검증합니다.
    
    Returns:
        True if 1-step training completes without NaN, False otherwise
    """
    logger.info("=" * 60)
    logger.info("  Running 1-Step Training Sanity Check")
    logger.info("=" * 60)

    controlnet.train()

    try:
        sample = dataset[0]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device, dtype=weight_dtype)
        conditioning = sample["conditioning_pixel_values"].unsqueeze(0).to(device, dtype=torch.float32)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)

        # Forward pass
        with torch.amp.autocast(device_type=device.type, dtype=weight_dtype, enabled=use_amp):
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (1,), device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0]

        # ControlNet forward (outside autocast)
        down_block_res_samples, mid_block_res_sample = controlnet(
            noisy_latents.to(dtype=torch.float32),
            timesteps,
            encoder_hidden_states=encoder_hidden_states.to(dtype=torch.float32),
            controlnet_cond=conditioning,
            return_dict=False,
        )

        with torch.amp.autocast(device_type=device.type, dtype=weight_dtype, enabled=use_amp):
            model_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    s.to(dtype=weight_dtype) for s in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        logger.info(f"  Forward pass loss: {loss.item():.6f}")

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error("  [FAIL] Loss is NaN/Inf after forward pass!")
            return False

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        logger.info(f"  Gradient norm: {grad_norm:.6f}")

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.error("  [FAIL] Gradient norm is NaN/Inf!")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return False

        # Cleanup (don't actually step the optimizer)
        optimizer.zero_grad(set_to_none=True)
        scaler.update()

        logger.info("  1-Step training sanity check PASSED")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"  [FAIL] 1-Step training sanity check failed: {e}")
        logger.info("=" * 60)
        optimizer.zero_grad(set_to_none=True)
        return False


# =============================================================================
# Validation / Inference
# =============================================================================

def log_validation(
    controlnet, vae, text_encoder, tokenizer, unet, noise_scheduler,
    args, step, device, dataset=None,
):
    """학습 중 검증 이미지를 생성합니다."""
    logger.info("Running validation...")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float32,  # validation은 fp32로
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    output_dir = Path(args.output_dir) / "validation" / f"step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(42)

    # 데이터셋에서 hint 이미지 가져오기
    num_val = min(2, len(dataset)) if dataset else 0

    for i in range(max(num_val, 2)):
        if dataset and i < len(dataset):
            sample = dataset[i]
            prompt = sample["prompt"]
            # conditioning tensor를 PIL Image로 변환
            cond_tensor = sample["conditioning_pixel_values"]  # [C, H, W] in [0, 1]
            cond_np = (cond_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            control_image = Image.fromarray(cond_np)
        else:
            prompt = "a linear scratch on metal surface, steel defect class 3"
            control_image = Image.new("RGB", (args.resolution, args.resolution), (0, 0, 0))

        with torch.autocast(str(device)):
            image = pipeline(
                prompt=prompt,
                image=control_image,
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        image.save(output_dir / f"val_{i:03d}_generated.png")
        control_image.save(output_dir / f"val_{i:03d}_condition.png")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Validation images saved to {output_dir}")


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """메인 학습 루프."""

    device = torch.device(args.device)

    # =====================================================================
    # [FIX #1] Mixed precision은 frozen 모델의 dtype만 결정.
    # ControlNet은 항상 fp32로 학습하고, AMP autocast로 forward만 fp16.
    # =====================================================================
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # =========================================================================
    # 1. 모델 로드
    # =========================================================================
    logger.info("Loading models...")

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # Text encoder (frozen, weight_dtype)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    # VAE (frozen, weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # UNet (frozen, weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    # =====================================================================
    # [FIX #2] ControlNet은 반드시 fp32로 로드 & 학습.
    # fp16으로 로드하면 gradient 계산에서 NaN 발생.
    # =====================================================================
    if args.controlnet_model_name_or_path:
        logger.info(f"Loading ControlNet from: {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            torch_dtype=torch.float32,  # 항상 fp32
        )
    else:
        logger.info("Initializing ControlNet from UNet...")
        controlnet = ControlNetModel.from_unet(unet)

    controlnet.to(device, dtype=torch.float32)  # 항상 fp32
    controlnet.train()

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in controlnet.parameters())
    logger.info(f"ControlNet: {trainable_params:,} trainable / {total_params:,} total params")

    # =========================================================================
    # 2. 데이터셋 로드
    # =========================================================================
    logger.info("Loading dataset...")

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / "train.jsonl"

    if not jsonl_path.exists():
        logger.error(f"train.jsonl not found at {jsonl_path}")
        logger.error("Run 'python scripts/prepare_controlnet_data.py' first.")
        return

    train_dataset = SteelDefectControlNetDataset(
        jsonl_path=str(jsonl_path),
        image_root=args.image_root,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )

    if len(train_dataset) == 0:
        logger.error("No valid training samples found!")
        return

    # =====================================================================
    # [FIX #3] Sanity check: 학습 전 데이터 검증
    # =====================================================================
    if not run_sanity_check(train_dataset, tokenizer, device):
        logger.error("Data sanity check failed. Aborting training.")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False,
    )

    logger.info(f"Dataset: {len(train_dataset)} samples")
    logger.info(f"Batch size: {args.train_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    effective_batch = args.train_batch_size * args.gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch}")

    # =========================================================================
    # 3. Optimizer & Scheduler
    # =========================================================================
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 총 학습 스텝 계산
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # =====================================================================
    # [FIX #4] warmup steps 자동 조정.
    # warmup이 총 step의 5~10%를 초과하지 않도록 클램핑.
    # 하한은 10 step으로 설정하여 최소한의 warmup 보장.
    # =====================================================================
    auto_warmup = max(10, max_train_steps // 20)
    effective_warmup = min(args.lr_warmup_steps, auto_warmup)
    if effective_warmup != args.lr_warmup_steps:
        logger.warning(
            f"Adjusted warmup steps: {args.lr_warmup_steps} -> {effective_warmup} "
            f"(total steps={max_train_steps})"
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=effective_warmup,
        num_training_steps=max_train_steps,
    )

    # =====================================================================
    # [FIX #5] GradScaler for mixed precision training.
    # fp16에서 gradient underflow/overflow 방지.
    # =====================================================================
    use_amp = args.mixed_precision in ("fp16", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # =========================================================================
    # 4. 학습 상태 복원
    # =========================================================================
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.output_dir)
        if args.resume_from_checkpoint == "latest":
            dirs = sorted(
                [d for d in ckpt_dir.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if dirs:
                args.resume_from_checkpoint = str(dirs[-1])
                logger.info(f"Resuming from: {args.resume_from_checkpoint}")
            else:
                logger.warning("No checkpoints found, training from scratch")
                args.resume_from_checkpoint = None

        if args.resume_from_checkpoint:
            ckpt_path = Path(args.resume_from_checkpoint)
            if (ckpt_path / "controlnet").exists():
                controlnet = ControlNetModel.from_pretrained(
                    ckpt_path / "controlnet", torch_dtype=torch.float32
                )
                controlnet.to(device, dtype=torch.float32)
                controlnet.train()
                if args.gradient_checkpointing:
                    controlnet.enable_gradient_checkpointing()
                logger.info(f"Loaded ControlNet from {ckpt_path / 'controlnet'}")

            opt_path = ckpt_path / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=True)
                )

            sched_path = ckpt_path / "lr_scheduler.pt"
            if sched_path.exists():
                lr_scheduler.load_state_dict(
                    torch.load(sched_path, map_location=device, weights_only=True)
                )

            step_file = ckpt_path / "global_step.txt"
            if step_file.exists():
                global_step = int(step_file.read_text().strip())
                resume_step = global_step % num_update_steps_per_epoch
                first_epoch = global_step // num_update_steps_per_epoch
                logger.info(f"Resuming from step {global_step} (epoch {first_epoch})")

    # =========================================================================
    # 4-2. 학습 초기 1-step forward/backward 완전 테스트
    # =========================================================================
    if not args.resume_from_checkpoint:
        if not run_training_sanity_check(
            controlnet=controlnet, vae=vae, text_encoder=text_encoder,
            unet=unet, noise_scheduler=noise_scheduler,
            dataset=train_dataset, tokenizer=tokenizer,
            optimizer=optimizer, scaler=scaler,
            device=device, weight_dtype=weight_dtype, use_amp=use_amp,
        ):
            logger.error(
                "1-step training sanity check failed. "
                "Fix NaN issues before starting full training. Aborting."
            )
            return

    # =========================================================================
    # 5. 학습 루프
    # =========================================================================
    logger.info("=" * 80)
    logger.info("  ControlNet Training for Steel Defect Augmentation")
    logger.info("=" * 80)
    logger.info(f"  Device             : {device}")
    logger.info(f"  Mixed precision    : {args.mixed_precision}")
    logger.info(f"  AMP enabled        : {use_amp}")
    logger.info(f"  Num examples       : {len(train_dataset)}")
    logger.info(f"  Num epochs         : {args.num_train_epochs}")
    logger.info(f"  Batch size         : {args.train_batch_size}")
    logger.info(f"  Grad accum steps   : {args.gradient_accumulation_steps}")
    logger.info(f"  Steps/epoch        : {num_update_steps_per_epoch}")
    logger.info(f"  Total steps        : {max_train_steps}")
    logger.info(f"  Learning rate      : {args.learning_rate}")
    logger.info(f"  LR scheduler       : {args.lr_scheduler}")
    logger.info(f"  Warmup steps       : {effective_warmup}")
    logger.info(f"  Max grad norm      : {args.max_grad_norm}")
    logger.info("=" * 80)

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=False,
    )

    # Loss tracking
    running_loss = 0.0
    running_count = 0
    best_loss = float("inf")
    loss_log = []
    nan_count = 0
    max_nan_tolerance = 10  # 연속 NaN 허용 횟수

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            # Skip steps for resumed training
            if epoch == first_epoch and step < resume_step:
                continue

            # =====================================================
            # Forward Pass
            # =====================================================

            # --- Frozen 모델 (VAE, Text Encoder)은 autocast 내에서 실행 ---
            with torch.amp.autocast(
                device_type=device.type,
                dtype=weight_dtype,
                enabled=use_amp,
            ):
                # 1. Target → VAE latent
                latents = vae.encode(
                    batch["pixel_values"].to(device, dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Forward diffusion (add noise)
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Text encoding
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(device)
                )[0]

            # --- ControlNet forward는 autocast 외부에서 fp32로 실행 ---
            # [FIX #8] Mixed Precision 안정화: ControlNet을 autocast 외부로 분리하여
            # fp32 ↔ fp16 변환 과정에서의 수치 불안정(NaN gradient)을 방지합니다.
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents.to(dtype=torch.float32),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.to(dtype=torch.float32),
                controlnet_cond=batch["conditioning_pixel_values"].to(
                    device, dtype=torch.float32
                ),
                return_dict=False,
            )

            # --- UNet forward는 autocast 내에서 실행 (frozen, fp16) ---
            with torch.amp.autocast(
                device_type=device.type,
                dtype=weight_dtype,
                enabled=use_amp,
            ):
                # 5. UNet forward (ControlNet residual 주입)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        s.to(dtype=weight_dtype) for s in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # 6. Loss 계산
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )
                loss = loss / args.gradient_accumulation_steps

            # =====================================================
            # [FIX #6] NaN loss 검출 및 스킵
            # NaN 카운터는 gradient accumulation 경계와 무관하게 동작
            # =====================================================
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                logger.warning(
                    f"[Step {global_step}] NaN/Inf loss detected "
                    f"(count={nan_count}/{max_nan_tolerance}). Skipping batch."
                )
                optimizer.zero_grad(set_to_none=True)

                # Adaptive LR 감소: NaN 발생 시 learning rate를 절반으로 줄임
                if nan_count > 0 and nan_count % 3 == 0:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        param_group['lr'] = old_lr * 0.5
                        logger.warning(
                            f"  Adaptive LR reduction: {old_lr:.2e} -> {param_group['lr']:.2e}"
                        )

                if nan_count >= max_nan_tolerance:
                    logger.error(
                        f"Too many NaN losses ({nan_count}). "
                        "Check data or reduce learning rate. Aborting."
                    )
                    return
                continue
            else:
                nan_count = 0  # 정상 loss면 카운터 리셋

            # =====================================================
            # Backward with GradScaler
            # =====================================================
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Unscale for gradient clipping
                scaler.unscale_(optimizer)

                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        controlnet.parameters(), args.max_grad_norm
                    )
                else:
                    grad_norm = None

                # =====================================================
                # [FIX #7] Gradient NaN 체크
                # gradient NaN도 nan_count에 반영
                # =====================================================
                if grad_norm is not None and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    nan_count += 1
                    logger.warning(
                        f"[Step {global_step}] NaN/Inf gradient norm "
                        f"(count={nan_count}/{max_nan_tolerance}). "
                        "Skipping optimizer step."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()

                    # Adaptive LR 감소
                    if nan_count > 0 and nan_count % 3 == 0:
                        for param_group in optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = old_lr * 0.5
                            logger.warning(
                                f"  Adaptive LR reduction: {old_lr:.2e} -> {param_group['lr']:.2e}"
                            )

                    if nan_count >= max_nan_tolerance:
                        logger.error(
                            f"Too many NaN gradient norms ({nan_count}). Aborting."
                        )
                        return
                    continue

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                current_loss = loss.item() * args.gradient_accumulation_steps
                running_loss += current_loss
                running_count += 1

                # Progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "epoch": f"{epoch+1}/{args.num_train_epochs}",
                })

                # Logging
                if global_step % args.logging_steps == 0 and running_count > 0:
                    avg_loss = running_loss / running_count
                    logger.info(
                        f"Step {global_step}: loss={avg_loss:.6f}, "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    loss_log.append({
                        "step": global_step,
                        "loss": avg_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    })
                    running_loss = 0.0
                    running_count = 0

                # Checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        controlnet, optimizer, lr_scheduler,
                        global_step, args, loss_log
                    )

                # Validation
                if args.validation_steps and global_step % args.validation_steps == 0:
                    log_validation(
                        controlnet=controlnet, vae=vae,
                        text_encoder=text_encoder, tokenizer=tokenizer,
                        unet=unet, noise_scheduler=noise_scheduler,
                        args=args, step=global_step, device=device,
                        dataset=train_dataset,
                    )
                    controlnet.train()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            if global_step >= max_train_steps:
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            f"Epoch {epoch+1}/{args.num_train_epochs}: "
            f"avg_loss={avg_epoch_loss:.6f}"
        )

        # Save best model (NaN이 아닐 때만)
        if not math.isnan(avg_epoch_loss) and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = Path(args.output_dir) / "best_model"
            controlnet.save_pretrained(save_path)
            logger.info(f"New best model (loss={best_loss:.6f}) -> {save_path}")

    # =========================================================================
    # 6. 최종 저장
    # =========================================================================
    logger.info("Training complete!")

    final_path = Path(args.output_dir) / "final_model"
    controlnet.save_pretrained(final_path)
    logger.info(f"Final model saved to {final_path}")

    log_path = Path(args.output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    logger.info(f"Training log saved to {log_path}")

    save_full_pipeline(controlnet, args, device)

    logger.info("=" * 80)
    logger.info("  Training Complete!")
    logger.info(f"  Best loss: {best_loss:.6f}")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 80)


def save_checkpoint(controlnet, optimizer, lr_scheduler, global_step, args, loss_log):
    """체크포인트를 저장합니다."""
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    controlnet.save_pretrained(ckpt_dir / "controlnet")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")
    (ckpt_dir / "global_step.txt").write_text(str(global_step))

    with open(ckpt_dir / "training_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    logger.info(f"Checkpoint saved: {ckpt_dir}")

    # 오래된 체크포인트 정리
    if args.checkpoints_total_limit:
        ckpt_parent = Path(args.output_dir)
        checkpoints = sorted(
            [d for d in ckpt_parent.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if len(checkpoints) > args.checkpoints_total_limit:
            for old_ckpt in checkpoints[:-args.checkpoints_total_limit]:
                shutil.rmtree(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")


def save_full_pipeline(controlnet, args, device):
    """추론용 전체 파이프라인을 저장합니다."""
    try:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        pipeline_path = Path(args.output_dir) / "pipeline"
        pipeline.save_pretrained(pipeline_path)
        logger.info(f"Full pipeline saved to {pipeline_path}")
    except Exception as e:
        logger.warning(f"Could not save full pipeline: {e}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ControlNet for Steel Defect Augmentation (diffusers)"
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path", type=str,
        default="lllyasviel/sd-controlnet-canny",
    )

    # Training
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default="no",
        choices=["no", "fp16", "bf16"],
    )

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR Scheduler
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=50)

    # Logging & Saving
    parser.add_argument("--output_dir", type=str, default="outputs/controlnet_training")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument(
        "--validation_steps", type=int, default=0,
        help="Run validation every N steps (0=disabled)",
    )

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Device
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.controlnet_model_name_or_path:
        args.controlnet_model_name_or_path = None

    return args


def main():
    args = parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save training config
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Training config saved to {config_path}")

    train(args)


if __name__ == "__main__":
    main()
