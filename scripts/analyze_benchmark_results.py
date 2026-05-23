#!/usr/bin/env python3
"""
Benchmark Results Analysis and Visualization

Analyzes the output of run_benchmark.py, generating:
  - Comparison tables (LaTeX and Markdown)
  - Performance bar charts per model and dataset
  - Precision-Recall curves
  - Training loss curves
  - Statistical significance tests between groups
  - Hypothesis validation summary

Usage:
  python scripts/analyze_benchmark_results.py --results-dir outputs/benchmark_results/<timestamp>
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Data Loading
# ============================================================================

def load_benchmark_results(results_dir: str) -> Dict:
    """Load all benchmark results from a completed experiment."""
    results_dir = Path(results_dir)

    data = {
        'results': [],
        'fid': {},
        'hypotheses': {},
        'config': {},
    }

    # Load main results
    results_path = results_dir / "benchmark_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data['results'] = json.load(f)

    # Load FID
    fid_path = results_dir / "fid_results.json"
    if fid_path.exists():
        with open(fid_path) as f:
            data['fid'] = json.load(f)

    # Load hypotheses
    hyp_path = results_dir / "hypothesis_results.json"
    if hyp_path.exists():
        with open(hyp_path) as f:
            data['hypotheses'] = json.load(f)

    # Load config
    config_path = results_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            data['config'] = yaml.safe_load(f)

    # Load per-experiment training histories
    data['histories'] = {}
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            meta_path = subdir / "experiment_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                key = f"{meta.get('model_key', '')}_{meta.get('dataset_group_key', '')}"
                data['histories'][key] = meta

    return data


# ============================================================================
# Table Generation
# ============================================================================

def generate_markdown_table(results: List[Dict]) -> str:
    """Generate Markdown comparison table matching experiment.md format."""
    lines = []
    lines.append("| Model | Dataset | mAP@0.5 | Dice | Class1 AP | Class2 AP | Class3 AP | Class4 AP |")
    lines.append("|-------|---------|---------|------|-----------|-----------|-----------|-----------|")

    for r in results:
        m = r['metrics']
        cap = m.get('class_ap', {})
        line = (
            f"| {r['model']:<12} | {r['dataset']:<18} | "
            f"{m.get('mAP@0.5', 0):.4f}  | {m.get('dice_mean', 0):.4f} | "
            f"{cap.get('Class1', 0):.4f}    | {cap.get('Class2', 0):.4f}    | "
            f"{cap.get('Class3', 0):.4f}    | {cap.get('Class4', 0):.4f}    |"
        )
        lines.append(line)

    return "\n".join(lines)


def generate_latex_table(results: List[Dict]) -> str:
    """Generate LaTeX comparison table for paper."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{CASDA Benchmark Results: Detection and Segmentation Performance}")
    lines.append(r"\label{tab:benchmark}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Dataset & mAP@0.5 & Dice & C1 AP & C2 AP & C3 AP & C4 AP \\")
    lines.append(r"\midrule")

    prev_model = None
    for r in results:
        m = r['metrics']
        cap = m.get('class_ap', {})
        model_str = r['model'] if r['model'] != prev_model else ""
        if r['model'] != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = r['model']

        line = (
            f"{model_str} & {r['dataset']} & "
            f"{m.get('mAP@0.5', 0):.4f} & {m.get('dice_mean', 0):.4f} & "
            f"{cap.get('Class1', 0):.4f} & {cap.get('Class2', 0):.4f} & "
            f"{cap.get('Class3', 0):.4f} & {cap.get('Class4', 0):.4f} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================================
# Visualization
# ============================================================================

def plot_map_comparison(results: List[Dict], output_dir: Path):
    """Bar chart comparing mAP@0.5 across models and datasets."""
    if not HAS_MATPLOTLIB:
        return

    # Organize data
    models = sorted(set(r['model'] for r in results))
    datasets = sorted(set(r['dataset'] for r in results))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(models))
    width = 0.2
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

    for i, ds in enumerate(datasets):
        values = []
        for model in models:
            val = 0.0
            for r in results:
                if r['model'] == model and r['dataset'] == ds:
                    val = r['metrics'].get('mAP@0.5', 0.0)
                    break
            values.append(val)

        bars = ax.bar(x + i * width, values, width, label=ds, color=colors[i % len(colors)])
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Detection Performance (mAP@0.5) by Model and Dataset')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / "map_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_ap_comparison(results: List[Dict], output_dir: Path):
    """Per-class AP comparison grouped by dataset."""
    if not HAS_MATPLOTLIB:
        return

    datasets = sorted(set(r['dataset'] for r in results))
    num_ds = len(datasets)

    fig, axes = plt.subplots(1, num_ds, figsize=(5 * num_ds, 5), sharey=True)
    if num_ds == 1:
        axes = [axes]

    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

    for ax, ds in zip(axes, datasets):
        ds_results = [r for r in results if r['dataset'] == ds]
        models = [r['model'] for r in ds_results]
        x = np.arange(4)  # 4 classes

        for i, r in enumerate(ds_results):
            cap = r['metrics'].get('class_ap', {})
            values = [cap.get(f'Class{c}', 0.0) for c in range(1, 5)]
            ax.bar(x + i * 0.2, values, 0.2, label=r['model'],
                   color=colors[i % len(colors)])

        ax.set_title(ds, fontsize=10)
        ax.set_xlabel('Defect Class')
        ax.set_ylabel('AP')
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels(['C1', 'C2', 'C3', 'C4'])
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Class AP by Dataset Group', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "class_ap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(histories: Dict, output_dir: Path):
    """Plot training loss and validation metric curves."""
    if not HAS_MATPLOTLIB or not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, meta in histories.items():
        history = meta.get('training_history', {})
        train_loss = history.get('train_loss', [])
        val_metric = history.get('val_metric', [])
        label = f"{meta.get('model', '')} / {meta.get('dataset_group', '')}"

        if train_loss:
            axes[0].plot(train_loss, label=label, alpha=0.7)
        if val_metric:
            axes[1].plot(val_metric, label=label, alpha=0.7)

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=6, loc='upper right')
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Validation Metric')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP / Dice')
    axes[1].legend(fontsize=6, loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curves(results_dir: Path, output_dir: Path):
    """Plot Precision-Recall curves from saved JSON data."""
    if not HAS_MATPLOTLIB:
        return

    pr_files = sorted(results_dir.glob("pr_curve_*.json"))
    if not pr_files:
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

    for pr_file in pr_files:
        with open(pr_file) as f:
            pr_data = json.load(f)

        label = f"{pr_data['model']} / {pr_data['dataset']}"

        for cls_idx in range(4):
            cls_key = f"Class{cls_idx + 1}"
            prec = pr_data.get('precisions', {}).get(cls_key, [])
            rec = pr_data.get('recalls', {}).get(cls_key, [])

            if prec and rec:
                axes[cls_idx].plot(rec, prec, label=label, alpha=0.7)

    for i in range(4):
        axes[i].set_title(f'Class {i+1}')
        axes[i].set_xlabel('Recall')
        if i == 0:
            axes[i].set_ylabel('Precision')
        axes[i].legend(fontsize=5)
        axes[i].grid(alpha=0.3)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

    plt.suptitle('Precision-Recall Curves by Class', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Statistical Tests
# ============================================================================

def compute_improvement_summary(results: List[Dict]) -> Dict:
    """Compute improvement percentages between dataset groups."""
    summary = {}

    models = sorted(set(r['model'] for r in results))

    for model in models:
        model_results = {r['dataset']: r['metrics'] for r in results if r['model'] == model}

        baseline_raw = model_results.get('Baseline (Raw)', {})
        baseline_trad = model_results.get('Baseline (Trad)', {})
        casda_full = model_results.get('CASDA-Full', {})
        casda_pruning = model_results.get('CASDA-Pruning', {})

        raw_map = baseline_raw.get('mAP@0.5', 0)

        model_summary = {}
        for name, group in [('Trad', baseline_trad), ('CASDA-Full', casda_full), ('CASDA-Pruning', casda_pruning)]:
            group_map = group.get('mAP@0.5', 0)
            if raw_map > 0:
                improvement = (group_map - raw_map) / raw_map * 100
            else:
                improvement = 0
            model_summary[name] = {
                'mAP': group_map,
                'improvement_pct': improvement,
            }

        summary[model] = model_summary

    return summary


# ============================================================================
# Main
# ============================================================================

def generate_markdown_table_mean_std(aggregated: Dict) -> str:
    """Generate mean±std Markdown table from aggregated multi-seed results."""
    entries = sorted(aggregated.values(), key=lambda e: (e['model'], e['dataset']))
    lines = [
        "| Model | Dataset | mAP@0.5 | Dice | C1 AP | C2 AP | C3 AP | C4 AP |",
        "|-------|---------|---------|------|-------|-------|-------|-------|",
    ]
    for e in entries:
        m, s = e['mean'], e['std']

        def fmt(metric: str) -> str:
            return f"{m.get(metric, 0):.4f} ± {s.get(metric, 0):.4f}"

        row = (
            f"| {e['model']:<12} | {e['dataset']:<25} | "
            f"{fmt('mAP@0.5')} | {fmt('dice_mean')} | "
            f"{fmt('class_ap_Class1')} | {fmt('class_ap_Class2')} | "
            f"{fmt('class_ap_Class3')} | {fmt('class_ap_Class4')} |"
        )
        lines.append(row)
    return "\n".join(lines)


def generate_latex_table_mean_std(aggregated: Dict) -> str:
    """Generate mean±std LaTeX table from aggregated multi-seed results."""
    entries = sorted(aggregated.values(), key=lambda e: (e['model'], e['dataset']))
    col_metrics = ['mAP@0.5', 'dice_mean'] + [f'class_ap_Class{i}' for i in range(1, 5)]
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

        def cell(metric: str) -> str:
            val_str = f"${m.get(metric, 0):.4f} \\pm {s.get(metric, 0):.4f}$"
            if abs(m.get(metric, 0) - best[metric]) < 1e-6:
                return r"\textbf{" + val_str.strip('$') + r"}"
            return val_str

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
    parser = argparse.ArgumentParser(description="Analyze benchmark experiment results")
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to benchmark results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for analysis (defaults to results-dir/analysis)')
    parser.add_argument('--aggregated-results', type=str, default=None,
                        help='Path to aggregated_results.json for mean±std table output '
                             '(from scripts/aggregate_multiseed_results.py)')
    args = parser.parse_args()

    if args.aggregated_results:
        agg_path = Path(args.aggregated_results)
        if not agg_path.exists():
            print(f"Error: aggregated results not found: {agg_path}")
            sys.exit(1)
        with open(agg_path) as f:
            aggregated = json.load(f)

        output_dir = Path(args.output_dir) if args.output_dir else agg_path.parent / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        md_table = generate_markdown_table_mean_std(aggregated)
        print("\n" + md_table)
        with open(output_dir / "comparison_table_mean_std.md", 'w', encoding='utf-8') as f:
            f.write("# Multi-Seed Benchmark Results (Mean ± Std)\n\n")
            f.write(md_table)
            f.write("\n")

        tex_table = generate_latex_table_mean_std(aggregated)
        with open(output_dir / "comparison_table_mean_std.tex", 'w', encoding='utf-8') as f:
            f.write(tex_table)

        print(f"\nMean±std tables saved to: {output_dir}")
        return

    if not args.results_dir:
        print("Error: --results-dir is required when --aggregated-results is not specified")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    data = load_benchmark_results(str(results_dir))

    if not data['results']:
        print("No benchmark results found!")
        sys.exit(1)

    print(f"Found {len(data['results'])} experiment results")

    # ====== Tables ======
    md_table = generate_markdown_table(data['results'])
    print("\n" + md_table)

    with open(output_dir / "comparison_table.md", 'w') as f:
        f.write(md_table)

    latex_table = generate_latex_table(data['results'])
    with open(output_dir / "comparison_table.tex", 'w') as f:
        f.write(latex_table)

    # ====== Improvement Summary ======
    improvement = compute_improvement_summary(data['results'])
    print("\nImprovement over Baseline (Raw):")
    for model, groups in improvement.items():
        print(f"\n  {model}:")
        for group_name, vals in groups.items():
            print(f"    {group_name}: mAP={vals['mAP']:.4f} ({vals['improvement_pct']:+.1f}%)")

    with open(output_dir / "improvement_summary.json", 'w') as f:
        json.dump(improvement, f, indent=2)

    # ====== Visualizations ======
    if HAS_MATPLOTLIB:
        print("\nGenerating visualizations...")
        plot_map_comparison(data['results'], output_dir)
        plot_class_ap_comparison(data['results'], output_dir)
        plot_training_curves(data.get('histories', {}), output_dir)
        plot_pr_curves(results_dir, output_dir)
        print(f"Visualizations saved to: {output_dir}")
    else:
        print("matplotlib not available, skipping visualizations")

    # ====== Hypothesis Summary ======
    if data['hypotheses']:
        print("\nHypothesis Results:")
        for h_name, h_result in data['hypotheses'].items():
            status = h_result.get('supported', h_result.get('status', 'unknown'))
            print(f"  {h_name}: {h_result['description']}")
            print(f"    -> {status}")

    # ====== FID ======
    if data['fid']:
        print(f"\nFID Score: {data['fid'].get('fid_overall', 'N/A')}")

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
