#!/usr/bin/env python3
"""
casda_no_blend vs casda_composed 블렌딩 방식 비교 figure 생성.

Usage:
  python scripts/visualize_blend_comparison.py \
    --no-blend-dir data/augmented/casda_no_blend \
    --composed-dir data/augmented/casda_composed \
    --train-images-dir data/raw/train_images \
    --generated-dir outputs/generated \
    --output figures/blend_comparison.png \
    --dpi 300
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {0: "Class 1", 1: "Class 2", 2: "Class 3", 3: "Class 4"}
COL_TITLES = ["Background", "Generated ROI", "No Blending", "Poisson Blending"]
SAMPLES_PER_CLASS = 2
CROP_W = 512


def load_metadata(meta_path: Path) -> List[dict]:
    with open(meta_path) as f:
        return json.load(f)


def build_lookup(entries: List[dict]) -> Dict[str, dict]:
    return {e["source_generated"]: e for e in entries}


def select_samples(
    nb_lookup: Dict[str, dict],
    cp_lookup: Dict[str, dict],
    n: int = SAMPLES_PER_CLASS,
) -> List[Tuple[dict, dict]]:
    common = set(nb_lookup) & set(cp_lookup)
    by_class: Dict[int, List[Tuple[dict, dict]]] = {}
    for key in common:
        nb_e = nb_lookup[key]
        cp_e = cp_lookup[key]
        cid = nb_e["class_id"]
        by_class.setdefault(cid, []).append((nb_e, cp_e))

    selected = []
    for cid in sorted(by_class):
        pairs = sorted(
            by_class[cid],
            key=lambda p: p[0]["suitability_score"],
            reverse=True,
        )
        selected.extend(pairs[:n])
    return selected


def crop_around_defect(
    img: np.ndarray, roi_bbox: list, jitter_x: int, crop_w: int = CROP_W
) -> np.ndarray:
    """roi_bbox + jitter_x 기준으로 crop_w × 256 window crop."""
    x1, _, x2, _ = roi_bbox
    cx = (x1 + x2) // 2 + jitter_x
    half = crop_w // 2
    left = max(0, cx - half)
    right = left + crop_w
    if right > img.shape[1]:
        right = img.shape[1]
        left = max(0, right - crop_w)
    return img[:, left:right]


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 이미지 로드 실패: {path}")
    return img


def make_figure(
    samples: List[Tuple[dict, dict]],
    no_blend_dir: Path,
    composed_dir: Path,
    train_images_dir: Path,
    generated_dir: Path,
    dpi: int = 300,
) -> plt.Figure:
    n_rows = len(samples)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(20, n_rows * 2.8),
        gridspec_kw={"width_ratios": [2, 1, 2, 2], "wspace": 0.04, "hspace": 0.35},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx, title in enumerate(COL_TITLES):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=6)

    prev_class = None
    for row_idx, (nb_e, cp_e) in enumerate(samples):
        cid = nb_e["class_id"]
        roi_bbox = nb_e["roi_bbox"]
        jitter_x = nb_e["jitter_x"]

        bg_img = load_image(train_images_dir / nb_e["source_background"])
        gen_img = load_image(generated_dir / nb_e["source_generated"])
        nb_img = load_image(no_blend_dir / nb_e["image_path"])
        cp_img = load_image(composed_dir / cp_e["image_path"])

        # 1600×256 이미지: roi_bbox + jitter 중심으로 512×256 crop
        for img, col, jit in zip([bg_img, nb_img, cp_img], [0, 2, 3], [0, jitter_x, jitter_x]):
            ax = axes[row_idx, col]
            if img is not None:
                ax.imshow(bgr_to_rgb(crop_around_defect(img, roi_bbox, jit)), aspect="auto")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="red")
            ax.axis("off")

        # Generated ROI: 512×512 원본 표시
        ax_roi = axes[row_idx, 1]
        if gen_img is not None:
            ax_roi.imshow(bgr_to_rgb(gen_img), aspect="auto")
        else:
            ax_roi.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax_roi.transAxes, color="red")
        ax_roi.axis("off")

        # class가 바뀌는 첫 번째 행에만 class 레이블 표시
        if cid != prev_class:
            axes[row_idx, 0].set_ylabel(
                CLASS_NAMES.get(cid, f"Class {cid + 1}"),
                fontsize=10, fontweight="bold", rotation=90, labelpad=6,
            )
            axes[row_idx, 0].yaxis.set_label_position("left")
            prev_class = cid

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="casda_no_blend vs casda_composed 블렌딩 비교 figure 생성"
    )
    parser.add_argument("--no-blend-dir", required=True)
    parser.add_argument("--composed-dir", required=True)
    parser.add_argument("--train-images-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output", default="figures/blend_comparison.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    nb_dir = Path(args.no_blend_dir)
    cp_dir = Path(args.composed_dir)
    train_dir = Path(args.train_images_dir)
    gen_dir = Path(args.generated_dir)
    out_path = Path(args.output)

    nb_meta = load_metadata(nb_dir / "metadata.json")
    cp_meta = load_metadata(cp_dir / "metadata.json")

    nb_lookup = build_lookup(nb_meta)
    cp_lookup = build_lookup(cp_meta)

    samples = select_samples(nb_lookup, cp_lookup)
    print(f"선택된 샘플: {len(samples)}개 (class당 최대 {SAMPLES_PER_CLASS}개)")
    for nb_e, _ in samples:
        print(f"  Class {nb_e['class_id'] + 1}: {nb_e['source_generated']} "
              f"(score={nb_e['suitability_score']:.3f})")

    fig = make_figure(samples, nb_dir, cp_dir, train_dir, gen_dir, dpi=args.dpi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
