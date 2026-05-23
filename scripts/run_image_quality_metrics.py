#!/usr/bin/env python3
"""
Image Quality Metrics: KID + LPIPS (realism / diversity)

Extends the existing FID evaluation (run_fid.py) with:
  - KID (Kernel Inception Distance): unbiased alternative to FID for small samples.
    Reuses InceptionV3 features from FIDCalculator — no duplicate model loading.
  - LPIPS realism: perceptual distance from generated ROI patches to real ROI patches.
    Lower is better (generated patches look like real defects).
  - LPIPS diversity: mean pairwise LPIPS within generated ROI patches.
    Higher is better (diverse synthesis, no mode collapse).

Why KID over FID for CASDA:
  FID bias ∝ 1/n. With ~2,200 generated images, FID estimates are noisy.
  KID uses an unbiased MMD estimator with polynomial kernel on the same
  InceptionV3 features, making it more reliable at this sample scale.

Usage:
  python scripts/run_image_quality_metrics.py \\
    --config        configs/benchmark_experiment.yaml \\
    --data-dir      /path/to/train_images \\
    --csv           /path/to/train.csv \\
    --casda-roi-dir /path/to/augmented_images/generated \\
    --roi-meta      /path/to/roi_metadata.csv \\
    --metrics       kid lpips \\
    --output-dir    /path/to/fid_results
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from src.training.metrics import FIDCalculator
    HAS_FID_CALC = True
except ImportError:
    HAS_FID_CALC = False


CLASS_IDS = [0, 1, 2, 3]  # 0-based
CLASS_NAMES = {0: 'Class1', 1: 'Class2', 2: 'Class3', 3: 'Class4'}


# ============================================================================
# Data Loading
# ============================================================================

def load_gen_by_class(casda_roi_dir: Path) -> Dict[int, List[str]]:
    """Load generated ROI images grouped by class_id (0-based).

    Tries metadata.json first; falls back to filename pattern parsing.
    """
    by_class: Dict[int, List[str]] = {}
    meta_path = casda_roi_dir / "metadata.json"

    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            entries = json.load(f)
        for entry in entries:
            cls_id = entry.get('class_id')
            if cls_id is None:
                continue
            rel = entry.get('image_path', '')
            abs_path = str(casda_roi_dir / rel)
            if os.path.exists(abs_path):
                by_class.setdefault(int(cls_id), []).append(abs_path)
        if by_class:
            return by_class
        logging.warning("metadata.json found but no valid entries — falling back to filename pattern")

    # Fallback: parse {image_id}_class{N}_region{M}_gen{J}.png
    import re
    pattern = re.compile(r'_class(\d+)_')
    for fpath in sorted(casda_roi_dir.glob('*.png')) + sorted(casda_roi_dir.glob('*.jpg')):
        m = pattern.search(fpath.name)
        if m:
            cls_1based = int(m.group(1))
            cls_id = cls_1based - 1  # → 0-based
            by_class.setdefault(cls_id, []).append(str(fpath))

    return by_class


def load_real_roi_by_class(roi_meta_path: Path) -> Dict[int, List[str]]:
    """Load real ROI images grouped by class_id (0-based) from roi_metadata.csv."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required: pip install pandas")
    df = pd.read_csv(roi_meta_path)
    by_class: Dict[int, List[str]] = {}
    for _, row in df.iterrows():
        cls_1based = int(row['class_id'])
        roi_path = str(row['roi_image_path'])
        if os.path.exists(roi_path):
            by_class.setdefault(cls_1based - 1, []).append(roi_path)
    return by_class


# ============================================================================
# KID — Kernel Inception Distance
# ============================================================================

def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial kernel: (X @ Y.T / d + 1)^degree."""
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** degree


def _mmd2_unbiased(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> float:
    """Unbiased MMD² estimator with polynomial kernel."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return float('nan')
    kxx = _poly_kernel(X, X, degree)
    kyy = _poly_kernel(Y, Y, degree)
    kxy = _poly_kernel(X, Y, degree)
    term_xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    term_xy = kxy.mean()
    return float(term_xx + term_yy - 2 * term_xy)


def compute_kid(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    n_subsets: int = 10,
    subset_size: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute KID mean ± std over multiple random subsets.

    Subset-based evaluation reduces computation and provides variance estimate.
    Reported in units ×10⁻³ (multiply by 1000 before reporting).
    """
    rng = np.random.RandomState(seed)
    n_real, n_gen = len(real_feats), len(gen_feats)
    sz = min(subset_size, n_real, n_gen)

    if sz < 4:
        return float('nan'), float('nan')

    mmd2_vals = []
    for _ in range(n_subsets):
        real_sub = real_feats[rng.choice(n_real, sz, replace=False)]
        gen_sub = gen_feats[rng.choice(n_gen, sz, replace=False)]
        val = _mmd2_unbiased(real_sub, gen_sub)
        if not math.isnan(val):
            mmd2_vals.append(val)

    if not mmd2_vals:
        return float('nan'), float('nan')

    return float(np.mean(mmd2_vals)), float(np.std(mmd2_vals))


def run_kid(
    gen_by_class: Dict[int, List[str]],
    real_by_class: Dict[int, List[str]],
    fid_calc: 'FIDCalculator',
    cache_dir: Optional[Path],
    n_subsets: int,
    subset_size: int,
) -> Dict:
    """Compute KID overall + per class. Returns results dict."""
    results: Dict = {'per_class': {}}

    all_real, all_gen = [], []
    for cls_id in CLASS_IDS:
        real_paths = real_by_class.get(cls_id, [])
        gen_paths = gen_by_class.get(cls_id, [])
        cls_name = CLASS_NAMES[cls_id]
        logging.info(f"  KID {cls_name}: {len(real_paths)} real, {len(gen_paths)} gen")

        if len(real_paths) < 4 or len(gen_paths) < 4:
            logging.warning(f"  {cls_name}: too few images, skipping")
            results['per_class'][cls_name] = {'kid_mean': None, 'kid_std': None}
            continue

        real_feats = fid_calc._extract_features(real_paths, cache_dir=cache_dir)
        gen_feats = fid_calc._extract_features(gen_paths, cache_dir=cache_dir)

        kid_mean, kid_std = compute_kid(real_feats, gen_feats, n_subsets, subset_size)
        results['per_class'][cls_name] = {
            'kid_mean': kid_mean * 1000 if not math.isnan(kid_mean) else None,
            'kid_std': kid_std * 1000 if not math.isnan(kid_std) else None,
            'n_real': len(real_paths),
            'n_gen': len(gen_paths),
        }
        all_real.extend(real_paths)
        all_gen.extend(gen_paths)

        if kid_mean is not None and not math.isnan(kid_mean):
            logging.info(f"    KID: {kid_mean*1000:.3f} ± {kid_std*1000:.3f} (×10⁻³)")

    # Overall KID
    if all_real and all_gen:
        real_all_feats = fid_calc._extract_features(all_real, cache_dir=cache_dir)
        gen_all_feats = fid_calc._extract_features(all_gen, cache_dir=cache_dir)
        kid_mean, kid_std = compute_kid(real_all_feats, gen_all_feats, n_subsets, subset_size)
        results['overall'] = {
            'kid_mean': kid_mean * 1000 if not math.isnan(kid_mean) else None,
            'kid_std': kid_std * 1000 if not math.isnan(kid_std) else None,
        }
        logging.info(f"  KID overall: {kid_mean*1000:.3f} ± {kid_std*1000:.3f} (×10⁻³)")
    else:
        results['overall'] = {'kid_mean': None, 'kid_std': None}

    return results


# ============================================================================
# LPIPS
# ============================================================================

def _load_image_tensor(path: str, size: int = 64, device: str = 'cpu') -> Optional['torch.Tensor']:
    """Load image as float tensor in [-1, 1], shape (1, 3, H, W)."""
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size), Image.BILINEAR)
        t = TF.to_tensor(img).unsqueeze(0)  # (1, 3, H, W), [0, 1]
        t = t * 2 - 1  # → [-1, 1]
        return t.to(device)
    except Exception as e:
        logging.debug(f"Failed to load {path}: {e}")
        return None


def compute_lpips_realism(
    real_paths: List[str],
    gen_paths: List[str],
    loss_fn: 'lpips_lib.LPIPS',
    n_pairs: int = 500,
    img_size: int = 64,
    device: str = 'cpu',
    seed: int = 42,
) -> Optional[float]:
    """Mean LPIPS between generated patches and randomly paired real patches."""
    if not gen_paths or not real_paths:
        return None
    rng = random.Random(seed)
    gen_sample = rng.sample(gen_paths, min(n_pairs, len(gen_paths)))
    distances = []
    for gen_path in gen_sample:
        real_path = rng.choice(real_paths)
        t_gen = _load_image_tensor(gen_path, img_size, device)
        t_real = _load_image_tensor(real_path, img_size, device)
        if t_gen is None or t_real is None:
            continue
        with torch.no_grad():
            d = loss_fn(t_gen, t_real).item()
        distances.append(d)
    return float(np.mean(distances)) if distances else None


def compute_lpips_diversity(
    gen_paths: List[str],
    loss_fn: 'lpips_lib.LPIPS',
    n_pairs: int = 500,
    img_size: int = 64,
    device: str = 'cpu',
    seed: int = 42,
) -> Optional[float]:
    """Mean pairwise LPIPS within generated patches (diversity)."""
    if len(gen_paths) < 2:
        return None
    rng = random.Random(seed)
    distances = []
    for _ in range(n_pairs):
        a, b = rng.sample(gen_paths, 2)
        t_a = _load_image_tensor(a, img_size, device)
        t_b = _load_image_tensor(b, img_size, device)
        if t_a is None or t_b is None:
            continue
        with torch.no_grad():
            d = loss_fn(t_a, t_b).item()
        distances.append(d)
    return float(np.mean(distances)) if distances else None


def run_lpips(
    gen_by_class: Dict[int, List[str]],
    real_by_class: Dict[int, List[str]],
    n_pairs: int,
    img_size: int,
    device: str,
) -> Dict:
    """Compute LPIPS realism and diversity per class and overall."""
    results: Dict = {
        'realism': {'per_class': {}},
        'diversity': {'per_class': {}},
    }

    loss_fn = lpips_lib.LPIPS(net='alex').to(device)
    loss_fn.eval()

    all_real, all_gen = [], []
    for cls_id in CLASS_IDS:
        real_paths = real_by_class.get(cls_id, [])
        gen_paths = gen_by_class.get(cls_id, [])
        cls_name = CLASS_NAMES[cls_id]
        logging.info(f"  LPIPS {cls_name}: {len(real_paths)} real, {len(gen_paths)} gen")

        realism = compute_lpips_realism(real_paths, gen_paths, loss_fn, n_pairs, img_size, device, seed=cls_id)
        diversity = compute_lpips_diversity(gen_paths, loss_fn, n_pairs, img_size, device, seed=cls_id + 10)

        results['realism']['per_class'][cls_name] = realism
        results['diversity']['per_class'][cls_name] = diversity

        if realism is not None:
            logging.info(f"    LPIPS realism:   {realism:.4f}")
        if diversity is not None:
            logging.info(f"    LPIPS diversity: {diversity:.4f}")

        all_real.extend(real_paths)
        all_gen.extend(gen_paths)

    # Overall
    overall_realism = compute_lpips_realism(all_real, all_gen, loss_fn, n_pairs, img_size, device, seed=99)
    overall_diversity = compute_lpips_diversity(all_gen, loss_fn, n_pairs, img_size, device, seed=100)
    results['realism']['overall'] = overall_realism
    results['diversity']['overall'] = overall_diversity

    if overall_realism is not None:
        logging.info(f"  LPIPS realism (overall):   {overall_realism:.4f}")
    if overall_diversity is not None:
        logging.info(f"  LPIPS diversity (overall): {overall_diversity:.4f}")

    return results


# ============================================================================
# Table Generation
# ============================================================================

def _fmt(val: Optional[float], scale: float = 1.0, decimals: int = 3) -> str:
    if val is None:
        return "—"
    return f"{val * scale:.{decimals}f}"


def generate_markdown_table(kid_results: Optional[Dict], lpips_results: Optional[Dict]) -> str:
    lines = [
        "## Image Quality Metrics",
        "",
        "| Class | KID↓ (×10⁻³) | KID std | LPIPS Realism↓ | LPIPS Diversity↑ |",
        "|-------|--------------|---------|----------------|-----------------|",
    ]
    all_cls = list(CLASS_NAMES.values()) + ['Overall']
    for cls_name in all_cls:
        is_overall = cls_name == 'Overall'

        kid_mean = kid_std = None
        lpips_r = lpips_d = None

        if kid_results:
            if is_overall:
                ov = kid_results.get('overall', {})
                kid_mean = ov.get('kid_mean')
                kid_std = ov.get('kid_std')
            else:
                pc = kid_results.get('per_class', {}).get(cls_name, {})
                kid_mean = pc.get('kid_mean')
                kid_std = pc.get('kid_std')

        if lpips_results:
            if is_overall:
                lpips_r = lpips_results.get('realism', {}).get('overall')
                lpips_d = lpips_results.get('diversity', {}).get('overall')
            else:
                lpips_r = lpips_results.get('realism', {}).get('per_class', {}).get(cls_name)
                lpips_d = lpips_results.get('diversity', {}).get('per_class', {}).get(cls_name)

        kid_mean_s = _fmt(kid_mean, decimals=3)
        kid_std_s = _fmt(kid_std, decimals=3)
        lpips_r_s = _fmt(lpips_r, decimals=4)
        lpips_d_s = _fmt(lpips_d, decimals=4)

        row = f"| {'**' + cls_name + '**' if is_overall else cls_name} | {kid_mean_s} | {kid_std_s} | {lpips_r_s} | {lpips_d_s} |"
        lines.append(row)

    return "\n".join(lines)


def generate_latex_table(kid_results: Optional[Dict], lpips_results: Optional[Dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Image Quality Metrics: KID and LPIPS Scores}",
        r"\label{tab:image_quality}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & KID$\downarrow$ ($\times 10^{-3}$) & KID std & LPIPS (Realism)$\downarrow$ & LPIPS (Diversity)$\uparrow$ \\",
        r"\midrule",
    ]

    all_cls = list(CLASS_NAMES.values())
    for cls_name in all_cls:
        kid_mean = kid_std = lpips_r = lpips_d = None
        if kid_results:
            pc = kid_results.get('per_class', {}).get(cls_name, {})
            kid_mean, kid_std = pc.get('kid_mean'), pc.get('kid_std')
        if lpips_results:
            lpips_r = lpips_results.get('realism', {}).get('per_class', {}).get(cls_name)
            lpips_d = lpips_results.get('diversity', {}).get('per_class', {}).get(cls_name)

        row = (
            f"{cls_name} & {_fmt(kid_mean)} & {_fmt(kid_std)} & "
            f"{_fmt(lpips_r, decimals=4)} & {_fmt(lpips_d, decimals=4)} \\\\"
        )
        lines.append(row)

    lines.append(r"\midrule")
    # Overall row
    kid_ov = kid_std_ov = lpips_r_ov = lpips_d_ov = None
    if kid_results:
        ov = kid_results.get('overall', {})
        kid_ov, kid_std_ov = ov.get('kid_mean'), ov.get('kid_std')
    if lpips_results:
        lpips_r_ov = lpips_results.get('realism', {}).get('overall')
        lpips_d_ov = lpips_results.get('diversity', {}).get('overall')

    lines.append(
        r"\textbf{Overall} & "
        f"{_fmt(kid_ov)} & {_fmt(kid_std_ov)} & "
        f"{_fmt(lpips_r_ov, decimals=4)} & {_fmt(lpips_d_ov, decimals=4)} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute KID + LPIPS metrics for CASDA generated ROI patches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to benchmark_experiment.yaml (optional, for device config)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Train images directory (currently unused, reserved for future use)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Annotation CSV (currently unused, reserved for future use)')
    parser.add_argument('--casda-roi-dir', type=str, required=True,
                        help='CASDA generated ROI directory (generated/ from compose step). '
                             'Must contain metadata.json or filename pattern '
                             '{id}_class{N}_region{M}_gen{J}.png')
    parser.add_argument('--roi-meta', type=str, required=True,
                        help='Real ROI metadata CSV (roi_metadata.csv). '
                             'Required columns: class_id (1-based), roi_image_path')
    parser.add_argument('--metrics', nargs='+', default=['kid', 'lpips'],
                        choices=['kid', 'lpips'],
                        help='Metrics to compute (default: kid lpips)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--kid-subsets', type=int, default=10,
                        help='Number of random subsets for KID estimation (default: 10)')
    parser.add_argument('--kid-subset-size', type=int, default=1000,
                        help='Subset size per KID estimation (default: 1000)')
    parser.add_argument('--lpips-pairs', type=int, default=500,
                        help='Number of image pairs for LPIPS estimation (default: 500)')
    parser.add_argument('--lpips-img-size', type=int, default=64,
                        help='Image size for LPIPS computation (default: 64)')
    parser.add_argument('--max-per-class', type=int, default=2000,
                        help='Max images per class (default: 2000)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Feature cache directory for KID (InceptionV3 features)')
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif HAS_TORCH and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / 'kid_cache'

    casda_roi_dir = Path(args.casda_roi_dir)
    roi_meta_path = Path(args.roi_meta)

    # ====== Load data ======
    logging.info(f"\nLoading generated ROI images from: {casda_roi_dir}")
    gen_by_class = load_gen_by_class(casda_roi_dir)
    for cls_id, paths in gen_by_class.items():
        cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        if len(paths) > args.max_per_class:
            rng = random.Random(42)
            gen_by_class[cls_id] = rng.sample(paths, args.max_per_class)
        logging.info(f"  {cls_name}: {len(gen_by_class[cls_id])} generated images")

    logging.info(f"\nLoading real ROI images from: {roi_meta_path}")
    real_by_class = load_real_roi_by_class(roi_meta_path)
    for cls_id, paths in real_by_class.items():
        cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        if len(paths) > args.max_per_class:
            rng = random.Random(42)
            real_by_class[cls_id] = rng.sample(paths, args.max_per_class)
        logging.info(f"  {cls_name}: {len(real_by_class[cls_id])} real ROI images")

    kid_results = None
    lpips_results = None

    # ====== KID ======
    if 'kid' in args.metrics:
        logging.info(f"\n{'='*50}")
        logging.info("Computing KID (Kernel Inception Distance)...")
        logging.info(f"{'='*50}")

        if not HAS_FID_CALC:
            logging.error("FIDCalculator not available (src.training.metrics). Cannot compute KID.")
        elif not HAS_TORCH:
            logging.error("torch not available. Cannot compute KID.")
        else:
            fid_calc = FIDCalculator(device=device)
            kid_results = run_kid(
                gen_by_class, real_by_class, fid_calc,
                cache_dir=cache_dir,
                n_subsets=args.kid_subsets,
                subset_size=args.kid_subset_size,
            )
            out_path = output_dir / "kid_results.json"
            with open(out_path, 'w') as f:
                json.dump(kid_results, f, indent=2, default=str)
            logging.info(f"\nKID results saved: {out_path}")

    # ====== LPIPS ======
    if 'lpips' in args.metrics:
        logging.info(f"\n{'='*50}")
        logging.info("Computing LPIPS (realism + diversity)...")
        logging.info(f"{'='*50}")

        if not HAS_LPIPS:
            logging.error("lpips not available. Install: pip install lpips")
        elif not HAS_TORCH:
            logging.error("torch not available. Cannot compute LPIPS.")
        else:
            lpips_results = run_lpips(
                gen_by_class, real_by_class,
                n_pairs=args.lpips_pairs,
                img_size=args.lpips_img_size,
                device=device,
            )
            out_path = output_dir / "lpips_results.json"
            with open(out_path, 'w') as f:
                json.dump(lpips_results, f, indent=2, default=str)
            logging.info(f"\nLPIPS results saved: {out_path}")

    # ====== Tables ======
    if kid_results or lpips_results:
        md = generate_markdown_table(kid_results, lpips_results)
        md_path = output_dir / "quality_metrics_table.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Image Quality Metrics: CASDA Generated ROI Patches\n\n")
            f.write(md)
            f.write("\n\n")
            f.write("Notes:\n")
            f.write("- KID (×10⁻³): Kernel Inception Distance. Unbiased FID alternative. Lower is better.\n")
            f.write("- LPIPS Realism: Perceptual distance from generated to real patches. Lower is better.\n")
            f.write("- LPIPS Diversity: Mean pairwise LPIPS within generated patches. Higher is better.\n")
        logging.info(f"\nMarkdown table saved: {md_path}")

        tex = generate_latex_table(kid_results, lpips_results)
        tex_path = output_dir / "quality_metrics_table.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex)
        logging.info(f"LaTeX table saved: {tex_path}")

        print("\n" + md)
    else:
        logging.warning("No metrics computed. Check dependencies and input paths.")

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
