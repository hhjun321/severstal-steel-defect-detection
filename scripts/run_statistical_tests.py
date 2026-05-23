#!/usr/bin/env python3
"""
Statistical significance tests for CASDA benchmark hypotheses.

Reads multi-seed aggregated results and runs:
  H3 — Architecture Independence (Friedman test across 3 models)
  H4 — Class 2 Improvement      (Wilcoxon signed-rank, paired by seed)
  H5 — LPIPS Realism Superiority (Wilcoxon signed-rank on per-class LPIPS realism)
  H6 — Augmentation Ratio        (Wilcoxon signed-rank, paired by seed)

Note: H5 was originally defined as FID superiority, but FID is structurally biased
toward CopyPaste (which copies real patches). H5 is re-defined as LPIPS realism
superiority — CASDA should have lower LPIPS realism (more perceptually similar to
real patches) due to Poisson Blending removing boundary artifacts.

Multiple-comparison correction: Benjamini-Hochberg FDR (α configurable).
Effect size: Cohen's d.

Usage:
  python scripts/run_statistical_tests.py \
    --aggregated-results     path/aggregated_results.json \
    --lpips-results          path/lpips_results.json \
    --copypaste-lpips-results path/copypaste_lpips_results.json \
    --output-dir             path/statistical_tests \
    --alpha 0.05
"""

import argparse
import json
import sys
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Statistical tests will be skipped.")


# Dataset display names (from benchmark_experiment.yaml)
DATASET_BASELINE = "Baseline (Raw)"
DATASET_CASDA = "CASDA-Composed-Pruning"
DATASET_COPYPASTE = "CopyPaste"


# ============================================================================
# Helpers
# ============================================================================

def get_seed_values(aggregated: Dict, model: str, dataset: str, metric: str) -> List[float]:
    key = f"{model}|{dataset}"
    entry = aggregated.get(key, {})
    return entry.get('seed_values', {}).get(metric, [])


def all_models(aggregated: Dict) -> List[str]:
    return sorted(set(e['model'] for e in aggregated.values()))


def cohens_d(a: List[float], b: List[float]) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float('nan')
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled < 1e-12:
        return float('nan')
    return float((np.mean(a) - np.mean(b)) / pooled)


def interpret_d(d: float) -> str:
    if math.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def bh_correction(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    adjusted = np.zeros(n)
    for rank, (idx, p) in enumerate(zip(sorted_idx, sorted_p), start=1):
        adjusted[idx] = p * n / rank

    # Enforce monotonicity (step-down)
    min_so_far = 1.0
    for i in sorted_idx[::-1]:
        min_so_far = min(adjusted[i], min_so_far)
        adjusted[i] = min_so_far

    return list(np.minimum(adjusted, 1.0))


def result_label(p_adj: float, alpha: float) -> str:
    if math.isnan(p_adj) or p_adj is None:
        return "N/A"
    if p_adj < alpha / 5:
        return "Supported**"
    if p_adj < alpha:
        return "Supported*"
    return "Not Supported"


def significance_stars(p_adj: float) -> str:
    if math.isnan(p_adj):
        return ""
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


# ============================================================================
# Per-class FID extraction
# ============================================================================

def extract_per_class_fid(fid_results: Dict, suffix: str = 'composed') -> Optional[Dict[str, float]]:
    """Try to extract per-class FID values from a fid_results.json dict."""
    per_class = {}
    for cls_num in range(1, 5):
        cls_name = f"Class{cls_num}"
        candidates = [
            f"fid_class{cls_num}_{suffix}",
            f"fid_{cls_name}_{suffix}",
            f"fid_class{cls_num}",
            f"class{cls_num}_fid",
            cls_name,
        ]
        for key in candidates:
            if key in fid_results and fid_results[key] not in (None, float('inf')):
                per_class[cls_name] = float(fid_results[key])
                break
    return per_class if len(per_class) == 4 else None


# ============================================================================
# Hypothesis Tests
# ============================================================================

def test_h3_architecture_independence(
    aggregated: Dict,
    models: List[str],
) -> Dict:
    """Friedman test: does casda-baseline delta vary across models?"""
    test_result = {
        'hypothesis': 'H3',
        'name': 'Architecture Independence',
        'test': 'Friedman',
        'description': 'CASDA improvement (Δ mAP@0.5) is consistent across all architectures',
    }

    if not HAS_SCIPY:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': 'scipy unavailable'})
        return test_result

    deltas_by_model = {}
    for model in models:
        baseline_vals = get_seed_values(aggregated, model, DATASET_BASELINE, 'mAP@0.5')
        casda_vals = get_seed_values(aggregated, model, DATASET_CASDA, 'mAP@0.5')
        if len(baseline_vals) == 0 or len(casda_vals) == 0:
            test_result.update({
                'stat': None, 'p_raw': None, 'effect_d': None,
                'note': f'Missing data for model {model}',
            })
            return test_result
        n = min(len(baseline_vals), len(casda_vals))
        deltas_by_model[model] = [casda_vals[i] - baseline_vals[i] for i in range(n)]

    if len(deltas_by_model) < 2:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None,
                            'note': 'Need at least 2 models'})
        return test_result

    try:
        groups = [deltas_by_model[m] for m in models if m in deltas_by_model]
        stat, p = friedmanchisquare(*groups)
        test_result['stat'] = float(stat)
        test_result['p_raw'] = float(p)
        test_result['effect_d'] = None
        test_result['stat_label'] = f"χ²={stat:.2f}"
        # H3 is "supported" if NOT significant (deltas are consistent across models)
        test_result['interpretation'] = 'supported_if_ns'
        test_result['note'] = (
            f"Models: {models}. "
            f"Δ mAP per seed: {deltas_by_model}. "
            "H3 supported when p > alpha (no significant cross-architecture variation)."
        )
    except Exception as e:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': str(e)})

    return test_result


def test_h4_class2_improvement(aggregated: Dict, models: List[str]) -> Dict:
    """Wilcoxon signed-rank: CASDA improves Class 2 AP vs baseline (per model, aggregated)."""
    test_result = {
        'hypothesis': 'H4',
        'name': 'Class 2 Improvement',
        'test': 'Wilcoxon signed-rank (one-tailed)',
        'description': 'CASDA-Composed-Pruning improves Class 2 AP over Baseline (Raw)',
    }

    if not HAS_SCIPY:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': 'scipy unavailable'})
        return test_result

    casda_all, baseline_all = [], []
    for model in models:
        c = get_seed_values(aggregated, model, DATASET_CASDA, 'class_ap_Class2')
        b = get_seed_values(aggregated, model, DATASET_BASELINE, 'class_ap_Class2')
        n = min(len(c), len(b))
        casda_all.extend(c[:n])
        baseline_all.extend(b[:n])

    if len(casda_all) < 3:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None,
                            'note': f'Too few observations: n={len(casda_all)}'})
        return test_result

    diffs = [a - b for a, b in zip(casda_all, baseline_all)]
    if all(d == 0 for d in diffs):
        test_result.update({'stat': 0.0, 'p_raw': 1.0, 'effect_d': 0.0,
                            'note': 'All differences are zero'})
        return test_result

    try:
        stat, p = wilcoxon(casda_all, baseline_all, alternative='greater')
        d = cohens_d(casda_all, baseline_all)
        test_result.update({
            'stat': float(stat),
            'p_raw': float(p),
            'effect_d': d if not math.isnan(d) else None,
            'stat_label': f"W={stat:.1f}",
            'interpretation': 'supported_if_significant',
            'note': (
                f"n={len(casda_all)} observations (seeds × models). "
                f"CASDA Class2: {[round(v, 4) for v in casda_all]}, "
                f"Baseline Class2: {[round(v, 4) for v in baseline_all]}"
            ),
        })
    except Exception as e:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': str(e)})

    return test_result


def extract_per_class_lpips_realism(lpips_results: Dict) -> Optional[Dict[str, float]]:
    """Extract per-class LPIPS realism values from lpips_results.json."""
    per_class = lpips_results.get('realism', {}).get('per_class', {})
    if not per_class:
        return None
    result = {}
    for cls_name in ['Class1', 'Class2', 'Class3', 'Class4']:
        if cls_name in per_class and per_class[cls_name] is not None:
            result[cls_name] = float(per_class[cls_name])
    return result if result else None


def test_h5_lpips_realism_superiority(
    casda_lpips: Optional[Dict],
    copypaste_lpips: Optional[Dict],
) -> Dict:
    """Wilcoxon signed-rank: CASDA LPIPS realism lower (better) than CopyPaste per class.

    H5 was originally FID superiority, but FID is biased toward CopyPaste (which
    copies real patches, making FID structurally near-zero). LPIPS realism correctly
    captures perceptual quality: Poisson Blending should remove boundary artifacts,
    yielding lower LPIPS realism for CASDA.
    """
    test_result = {
        'hypothesis': 'H5',
        'name': 'LPIPS Realism Superiority (Poisson Blending)',
        'test': 'Wilcoxon signed-rank (one-tailed, class-level)',
        'description': (
            'CASDA has lower LPIPS realism than CopyPaste (better perceptual quality). '
            'H5 redefined from FID superiority: FID is biased toward CopyPaste which '
            'copies real patches, making FID an invalid quality metric here.'
        ),
    }

    if casda_lpips is None or copypaste_lpips is None:
        test_result.update({
            'stat': None, 'p_raw': None, 'effect_d': None,
            'note': (
                'Per-class LPIPS realism data not available. '
                'Provide --lpips-results and --copypaste-lpips-results.'
            ),
        })
        return test_result

    if not HAS_SCIPY:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': 'scipy unavailable'})
        return test_result

    classes = [k for k in ['Class1', 'Class2', 'Class3', 'Class4']
               if k in casda_lpips and k in copypaste_lpips]
    if len(classes) < 3:
        test_result.update({
            'stat': None, 'p_raw': None, 'effect_d': None,
            'note': f'Insufficient class-level LPIPS data: found {len(classes)} classes',
        })
        return test_result

    casda_vals = [casda_lpips[c] for c in classes]
    cp_vals = [copypaste_lpips[c] for c in classes]

    diffs = [a - b for a, b in zip(casda_vals, cp_vals)]
    if all(d == 0 for d in diffs):
        test_result.update({'stat': 0.0, 'p_raw': 1.0, 'effect_d': 0.0,
                            'note': 'All differences are zero'})
        return test_result

    try:
        # CASDA should have LOWER LPIPS realism → alternative='less'
        stat, p = wilcoxon(casda_vals, cp_vals, alternative='less')
        d = cohens_d(cp_vals, casda_vals)  # positive d: CopyPaste > CASDA
        test_result.update({
            'stat': float(stat),
            'p_raw': float(p),
            'effect_d': d if not math.isnan(d) else None,
            'stat_label': f"W={stat:.1f}",
            'interpretation': 'supported_if_significant',
            'note': (
                f"Paired per-class LPIPS realism (n={len(classes)} classes: {classes}). "
                f"CASDA: {[round(v, 4) for v in casda_vals]}, "
                f"CopyPaste: {[round(v, 4) for v in cp_vals]}"
            ),
        })
    except Exception as e:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': str(e)})

    return test_result


def test_h6_augmentation_ratio(aggregated: Dict, models: List[str]) -> Dict:
    """Wilcoxon signed-rank: CASDA mAP better than CopyPaste."""
    test_result = {
        'hypothesis': 'H6',
        'name': 'Augmentation Ratio',
        'test': 'Wilcoxon signed-rank (one-tailed)',
        'description': 'CASDA-Composed-Pruning achieves higher mAP@0.5 than CopyPaste',
    }

    if not HAS_SCIPY:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': 'scipy unavailable'})
        return test_result

    casda_all, cp_all = [], []
    for model in models:
        c = get_seed_values(aggregated, model, DATASET_CASDA, 'mAP@0.5')
        cp = get_seed_values(aggregated, model, DATASET_COPYPASTE, 'mAP@0.5')
        n = min(len(c), len(cp))
        casda_all.extend(c[:n])
        cp_all.extend(cp[:n])

    if len(casda_all) < 3:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None,
                            'note': f'Too few observations: n={len(casda_all)}'})
        return test_result

    diffs = [a - b for a, b in zip(casda_all, cp_all)]
    if all(d == 0 for d in diffs):
        test_result.update({'stat': 0.0, 'p_raw': 1.0, 'effect_d': 0.0,
                            'note': 'All differences are zero'})
        return test_result

    try:
        stat, p = wilcoxon(casda_all, cp_all, alternative='greater')
        d = cohens_d(casda_all, cp_all)
        test_result.update({
            'stat': float(stat),
            'p_raw': float(p),
            'effect_d': d if not math.isnan(d) else None,
            'stat_label': f"W={stat:.1f}",
            'interpretation': 'supported_if_significant',
            'note': (
                f"n={len(casda_all)} observations (seeds × models). "
                f"CASDA mAP: {[round(v, 4) for v in casda_all]}, "
                f"CopyPaste mAP: {[round(v, 4) for v in cp_all]}"
            ),
        })
    except Exception as e:
        test_result.update({'stat': None, 'p_raw': None, 'effect_d': None, 'note': str(e)})

    return test_result


# ============================================================================
# Table generation
# ============================================================================

def make_result_summary(h: Dict, alpha: float) -> Dict:
    p_adj = h.get('p_adj')
    p_raw = h.get('p_raw')
    d = h.get('effect_d')
    stat_label = h.get('stat_label', '—')
    interp = h.get('interpretation', 'supported_if_significant')

    if p_adj is not None and not math.isnan(p_adj):
        p_adj_str = f"p={p_adj:.3f}"
    else:
        p_adj_str = "N/A"

    if p_raw is not None and not math.isnan(p_raw):
        p_raw_str = f"p={p_raw:.3f}"
    else:
        p_raw_str = "N/A"

    if d is not None and not math.isnan(d):
        d_str = f"d={d:.2f} ({interpret_d(d)})"
    else:
        d_str = "—"

    # Determine result label
    if p_adj is None:
        label = "N/A"
    elif interp == 'supported_if_ns':
        # H3: supported when NOT significant
        if p_adj > alpha:
            label = "Supported" + significance_stars(1 - p_adj)
        else:
            label = "Not Supported"
    else:
        label = result_label(p_adj, alpha)

    return {
        'stat_label': stat_label,
        'p_raw_str': p_raw_str,
        'p_adj_str': p_adj_str,
        'd_str': d_str,
        'label': label,
    }


def generate_markdown_significance_table(results: List[Dict], alpha: float) -> str:
    lines = [
        "| Hypothesis | Test | Statistic | p-value (raw) | p-value (BH-adj) | Effect size (d) | Result |",
        "|-----------|------|-----------|---------------|------------------|-----------------|--------|",
    ]
    for h in results:
        s = make_result_summary(h, alpha)
        row = (
            f"| {h['hypothesis']}: {h['name']} | {h['test']} | "
            f"{s['stat_label']} | {s['p_raw_str']} | {s['p_adj_str']} | "
            f"{s['d_str']} | {s['label']} |"
        )
        lines.append(row)
    lines.append("")
    lines.append(f"Significance: * p<{alpha}, ** p<{alpha/5:.3f} (Benjamini-Hochberg FDR, α={alpha})")
    return "\n".join(lines)


def generate_latex_significance_table(results: List[Dict], alpha: float) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Significance Tests for CASDA Hypotheses}",
        r"\label{tab:significance}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Hyp. & Test & Statistic & $p$ (raw) & $p$ (BH-adj) & Effect ($d$) & Result \\",
        r"\midrule",
    ]
    for h in results:
        s = make_result_summary(h, alpha)
        # Escape special chars for LaTeX
        test_str = h['test'].replace('&', r'\&')
        row = (
            f"{h['hypothesis']} & {h['name']} & {test_str} & "
            f"{s['stat_label']} & {s['p_raw_str']} & {s['p_adj_str']} & "
            f"{s['d_str']} & {s['label']} \\\\"
        )
        lines.append(row)
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\small Significance: * $p<" + f"{alpha}" + r"$, ** $p<" + f"{alpha/5:.3f}" + r"$ (Benjamini-Hochberg FDR, $\alpha=" + f"{alpha}" + r"$)} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run statistical tests on multi-seed benchmark results")
    parser.add_argument('--aggregated-results', type=str, required=True,
                        help='Path to aggregated_results.json (from aggregate_multiseed_results.py)')
    parser.add_argument('--lpips-results', type=str, default=None,
                        help='Path to CASDA lpips_results.json (from run_image_quality_metrics.py) — required for H5')
    parser.add_argument('--copypaste-lpips-results', type=str, default=None,
                        help='Path to CopyPaste lpips_results.json — required for H5')
    # Legacy FID args kept for backward compatibility (no longer used for H5)
    parser.add_argument('--fid-results', type=str, default=None,
                        help='[unused for H5] Path to CASDA fid_results.json')
    parser.add_argument('--copypaste-fid-results', type=str, default=None,
                        help='[unused for H5] Path to CopyPaste fid_results.json')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for test results')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load aggregated results
    with open(args.aggregated_results) as f:
        aggregated = json.load(f)
    print(f"Loaded {len(aggregated)} (model, dataset) combinations from {args.aggregated_results}")

    models = all_models(aggregated)
    print(f"Models: {models}")

    # Load LPIPS results for H5 (optional)
    casda_lpips_per_class = None
    copypaste_lpips_per_class = None

    if args.lpips_results:
        with open(args.lpips_results) as f:
            casda_lpips_data = json.load(f)
        casda_lpips_per_class = extract_per_class_lpips_realism(casda_lpips_data)
        if casda_lpips_per_class:
            print(f"CASDA per-class LPIPS realism: {casda_lpips_per_class}")
        else:
            print("WARNING: Could not extract per-class LPIPS realism from CASDA lpips_results.json")

    if args.copypaste_lpips_results:
        with open(args.copypaste_lpips_results) as f:
            cp_lpips_data = json.load(f)
        copypaste_lpips_per_class = extract_per_class_lpips_realism(cp_lpips_data)
        if copypaste_lpips_per_class:
            print(f"CopyPaste per-class LPIPS realism: {copypaste_lpips_per_class}")
        else:
            print("WARNING: Could not extract per-class LPIPS realism from CopyPaste lpips_results.json")

    # ====== Run Hypothesis Tests ======
    print(f"\nRunning hypothesis tests (α = {args.alpha})...")

    h3 = test_h3_architecture_independence(aggregated, models)
    h4 = test_h4_class2_improvement(aggregated, models)
    h5 = test_h5_lpips_realism_superiority(casda_lpips_per_class, copypaste_lpips_per_class)
    h6 = test_h6_augmentation_ratio(aggregated, models)

    all_tests = [h3, h4, h5, h6]

    # ====== BH-FDR Correction ======
    raw_p = [h.get('p_raw') for h in all_tests]
    valid_mask = [p is not None and not math.isnan(p) for p in raw_p]
    valid_p = [raw_p[i] for i in range(len(raw_p)) if valid_mask[i]]

    adjusted = bh_correction(valid_p)

    adj_iter = iter(adjusted)
    for h, is_valid in zip(all_tests, valid_mask):
        if is_valid:
            h['p_adj'] = next(adj_iter)
        else:
            h['p_adj'] = None

    # ====== Print Summary ======
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 70)
    for h in all_tests:
        p_raw = h.get('p_raw')
        p_adj = h.get('p_adj')
        d = h.get('effect_d')
        p_raw_s = f"{p_raw:.4f}" if p_raw is not None else "N/A"
        p_adj_s = f"{p_adj:.4f}" if p_adj is not None else "N/A"
        d_s = f"{d:.3f} ({interpret_d(d)})" if d is not None else "N/A"
        s = make_result_summary(h, args.alpha)
        print(f"\n{h['hypothesis']}: {h['name']}")
        print(f"  Test:       {h['test']}")
        print(f"  Statistic:  {h.get('stat_label', '—')}")
        print(f"  p (raw):    {p_raw_s}")
        print(f"  p (BH-adj): {p_adj_s}")
        print(f"  Cohen's d:  {d_s}")
        print(f"  Result:     {s['label']}")
        if h.get('note'):
            print(f"  Note:       {h['note'][:120]}")
    print("=" * 70)

    # ====== Save JSON ======
    out_json = output_dir / "hypothesis_test_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_tests, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")

    # ====== Save Tables ======
    md_table = generate_markdown_significance_table(all_tests, args.alpha)
    md_path = output_dir / "significance_table.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Statistical Significance Tests\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"Saved: {md_path}")

    tex_table = generate_latex_significance_table(all_tests, args.alpha)
    tex_path = output_dir / "significance_table.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_table)
    print(f"Saved: {tex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
