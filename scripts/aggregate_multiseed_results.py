#!/usr/bin/env python3
"""
Aggregate multi-seed benchmark results.

Reads benchmark_results.json from multiple seed directories and computes
mean ± std across seeds for each (model, dataset) combination.

Usage:
  python scripts/aggregate_multiseed_results.py \
    --results-dirs path/seed_42 path/seed_123 path/seed_456 \
    --output-dir path/multiseed_aggregated
"""

import argparse
import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


SCALAR_METRICS = ['mAP@0.5', 'dice_mean']
CLASS_KEYS = ['Class1', 'Class2', 'Class3', 'Class4']


def load_seed_results(results_dir: Path) -> List[Dict]:
    results_path = results_dir / "benchmark_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"benchmark_results.json not found in {results_dir}")
    with open(results_path) as f:
        return json.load(f)


def extract_metrics(entry: Dict) -> Dict[str, float]:
    m = entry['metrics']
    metrics = {}
    for k in SCALAR_METRICS:
        metrics[k] = float(m.get(k, 0.0))
    class_ap = m.get('class_ap', {})
    for cls in CLASS_KEYS:
        metrics[f'class_ap_{cls}'] = float(class_ap.get(cls, 0.0))
    return metrics


def aggregate(results_dirs: List[Path]) -> Dict:
    grouped = defaultdict(lambda: {
        'model': None,
        'dataset': None,
        'seed_values': defaultdict(list),
    })

    for seed_dir in results_dirs:
        seed_results = load_seed_results(seed_dir)
        for entry in seed_results:
            key = f"{entry['model']}|{entry['dataset']}"
            grouped[key]['model'] = entry['model']
            grouped[key]['dataset'] = entry['dataset']
            for metric, value in extract_metrics(entry).items():
                grouped[key]['seed_values'][metric].append(value)

    aggregated = {}
    for key, data in grouped.items():
        seed_vals = data['seed_values']
        mean_vals, std_vals = {}, {}
        for metric, values in seed_vals.items():
            arr = np.array(values, dtype=float)
            mean_vals[metric] = float(np.mean(arr))
            std_vals[metric] = float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)

        aggregated[key] = {
            'model': data['model'],
            'dataset': data['dataset'],
            'n_seeds': len(next(iter(seed_vals.values()))) if seed_vals else 0,
            'seed_values': {k: list(v) for k, v in seed_vals.items()},
            'mean': mean_vals,
            'std': std_vals,
        }

    return aggregated


def _fmt(mean: float, std: float, fmt: str = 'md') -> str:
    if fmt == 'tex':
        return f"${mean:.4f} \\pm {std:.4f}$"
    return f"{mean:.4f} ± {std:.4f}"


def _bold_tex(s: str) -> str:
    return r"\textbf{" + s.strip('$') + "}"


def generate_markdown_table(aggregated: Dict) -> str:
    entries = sorted(aggregated.values(), key=lambda e: (e['model'], e['dataset']))
    lines = [
        "| Model | Dataset | mAP@0.5 | Dice | C1 AP | C2 AP | C3 AP | C4 AP |",
        "|-------|---------|---------|------|-------|-------|-------|-------|",
    ]
    for e in entries:
        m, s = e['mean'], e['std']
        row = (
            f"| {e['model']:<12} | {e['dataset']:<25} | "
            f"{_fmt(m.get('mAP@0.5', 0), s.get('mAP@0.5', 0))} | "
            f"{_fmt(m.get('dice_mean', 0), s.get('dice_mean', 0))} | "
            f"{_fmt(m.get('class_ap_Class1', 0), s.get('class_ap_Class1', 0))} | "
            f"{_fmt(m.get('class_ap_Class2', 0), s.get('class_ap_Class2', 0))} | "
            f"{_fmt(m.get('class_ap_Class3', 0), s.get('class_ap_Class3', 0))} | "
            f"{_fmt(m.get('class_ap_Class4', 0), s.get('class_ap_Class4', 0))} |"
        )
        lines.append(row)
    return "\n".join(lines)


def generate_latex_table(aggregated: Dict) -> str:
    entries = sorted(aggregated.values(), key=lambda e: (e['model'], e['dataset']))

    # Find per-column best mean for boldface
    col_metrics = ['mAP@0.5', 'dice_mean'] + [f'class_ap_{c}' for c in CLASS_KEYS]
    best = {col: max(e['mean'].get(col, 0) for e in entries) for col in col_metrics}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CASDA Multi-Seed Benchmark Results (Mean $\pm$ Std, $n=3$ seeds)}",
        r"\label{tab:multiseed}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Model & Dataset & mAP@0.5 & Dice & C1 AP & C2 AP & C3 AP & C4 AP \\",
        r"\midrule",
    ]

    prev_model = None
    for e in entries:
        m, s = e['mean'], e['std']
        model_str = e['model'] if e['model'] != prev_model else ""
        if e['model'] != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = e['model']

        def cell(metric):
            val = _fmt(m.get(metric, 0), s.get(metric, 0), 'tex')
            if abs(m.get(metric, 0) - best[metric]) < 1e-6:
                return r"\textbf{" + val.strip('$') + r"}"
            return val

        row = (
            f"{model_str} & {e['dataset']} & "
            f"{cell('mAP@0.5')} & {cell('dice_mean')} & "
            f"{cell('class_ap_Class1')} & {cell('class_ap_Class2')} & "
            f"{cell('class_ap_Class3')} & {cell('class_ap_Class4')} \\\\"
        )
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed benchmark results")
    parser.add_argument('--results-dirs', nargs='+', required=True,
                        help='Seed result directories (each must contain benchmark_results.json)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {len(results_dirs)} seed directories:")
    for d in results_dirs:
        print(f"  {d}")

    aggregated = aggregate(results_dirs)

    print(f"\nCombinations found ({len(aggregated)}):")
    for key, e in sorted(aggregated.items()):
        print(f"  {e['model']} | {e['dataset']} (n={e['n_seeds']} seeds)")

    out_json = output_dir / "aggregated_results.json"
    with open(out_json, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved: {out_json}")

    md_table = generate_markdown_table(aggregated)
    md_path = output_dir / "table_mean_std.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Seed Benchmark Results (Mean ± Std)\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"Saved: {md_path}")
    print("\n" + md_table)

    tex_table = generate_latex_table(aggregated)
    tex_path = output_dir / "table_mean_std.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_table)
    print(f"Saved: {tex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
