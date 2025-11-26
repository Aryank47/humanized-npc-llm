"""
Visualization Script for Evaluation Results
Generates charts and plots for model comparison.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import pandas as pd
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_report(report_path: str) -> Dict[str, Any]:
    """Load the final evaluation report."""
    with open(report_path, 'r') as f:
        return json.load(f)


def load_generations(gen_path: str) -> List[Dict[str, Any]]:
    """Load detailed generation results."""
    results = []
    with open(gen_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_metric_comparison(
    report: Dict[str, Any],
    output_dir: str
) -> None:
    """Create bar chart comparing baseline vs fine-tuned metrics."""

    baseline = report['baseline_model_metrics']
    tuned = report['tuned_model_metrics']
    comparison = report['statistical_comparison']

    metrics_to_plot = {
        'Schema Validity': ('avg_is_valid_schema', 'higher_better'),
        'Persona Similarity': ('avg_persona_similarity_max', 'higher_better'),
        'Hallucination (UCR)': ('avg_ucr', 'lower_better'),
        'Grounding (NEP)': ('avg_nep', 'higher_better'),
        'Distinct-2': ('diversity_distinct_2', 'higher_better'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (display_name, (metric_key, direction)) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]

        baseline_val = baseline.get(metric_key, 0)
        tuned_val = tuned.get(metric_key, 0)

        # Get statistical significance
        is_significant = False
        if metric_key.replace('avg_', '').replace('diversity_', '') in comparison:
            comp_key = metric_key.replace('avg_', '').replace('diversity_', '')
            is_significant = comparison[comp_key].get('is_significant', False)

        x = ['Baseline', 'Fine-Tuned']
        y = [baseline_val, tuned_val]

        colors = ['#ff7f0e', '#2ca02c']
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')

        # Add significance marker
        if is_significant:
            max_y = max(y)
            ax.text(0.5, max_y * 1.05, '***', ha='center', va='bottom',
                   fontsize=16, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, y)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Score')
        ax.set_title(display_name, fontweight='bold')
        ax.set_ylim(0, max(y) * 1.15)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'metric_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric comparison: {output_path}")
    plt.close()


def plot_metric_distributions(
    generations: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Plot distributions of key metrics."""

    # Extract metrics
    baseline_sim = [g['baseline']['metrics'].get('persona_similarity_max')
                    for g in generations
                    if g['baseline']['metrics'].get('persona_similarity_max') is not None]

    tuned_sim = [g['tuned']['metrics'].get('persona_similarity_max')
                 for g in generations
                 if g['tuned']['metrics'].get('persona_similarity_max') is not None]

    baseline_ucr = [g['baseline']['metrics'].get('ucr')
                    for g in generations
                    if g['baseline']['metrics'].get('ucr') is not None]

    tuned_ucr = [g['tuned']['metrics'].get('ucr')
                 for g in generations
                 if g['tuned']['metrics'].get('ucr') is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Persona Similarity Distribution
    ax = axes[0]
    ax.hist(baseline_sim, bins=30, alpha=0.5, label='Baseline', color='#ff7f0e', edgecolor='black')
    ax.hist(tuned_sim, bins=30, alpha=0.5, label='Fine-Tuned', color='#2ca02c', edgecolor='black')
    ax.axvline(np.mean(baseline_sim), color='#ff7f0e', linestyle='--', linewidth=2, label=f'Baseline Mean: {np.mean(baseline_sim):.3f}')
    ax.axvline(np.mean(tuned_sim), color='#2ca02c', linestyle='--', linewidth=2, label=f'Tuned Mean: {np.mean(tuned_sim):.3f}')
    ax.set_xlabel('Persona Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Persona Similarity', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hallucination (UCR) Distribution
    ax = axes[1]
    ax.hist(baseline_ucr, bins=30, alpha=0.5, label='Baseline', color='#ff7f0e', edgecolor='black')
    ax.hist(tuned_ucr, bins=30, alpha=0.5, label='Fine-Tuned', color='#2ca02c', edgecolor='black')
    ax.axvline(np.mean(baseline_ucr), color='#ff7f0e', linestyle='--', linewidth=2, label=f'Baseline Mean: {np.mean(baseline_ucr):.3f}')
    ax.axvline(np.mean(tuned_ucr), color='#2ca02c', linestyle='--', linewidth=2, label=f'Tuned Mean: {np.mean(tuned_ucr):.3f}')
    ax.set_xlabel('UCR (Hallucination Rate)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Hallucination Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metric_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric distributions: {output_path}")
    plt.close()


def plot_mood_distribution(
    generations: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Plot mood distribution comparison."""

    baseline_moods = [g['baseline']['metrics'].get('mood')
                      for g in generations
                      if g['baseline']['metrics'].get('mood')]

    tuned_moods = [g['tuned']['metrics'].get('mood')
                   for g in generations
                   if g['tuned']['metrics'].get('mood')]

    baseline_counts = Counter(baseline_moods)
    tuned_counts = Counter(tuned_moods)

    # Get all unique moods
    all_moods = sorted(set(list(baseline_counts.keys()) + list(tuned_counts.keys())))

    baseline_vals = [baseline_counts.get(mood, 0) for mood in all_moods]
    tuned_vals = [tuned_counts.get(mood, 0) for mood in all_moods]

    x = np.arange(len(all_moods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color='#ff7f0e', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, tuned_vals, width, label='Fine-Tuned',
                   color='#2ca02c', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Mood')
    ax.set_ylabel('Frequency')
    ax.set_title('Mood Distribution Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_moods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mood_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mood distribution: {output_path}")
    plt.close()


def plot_per_source_performance(
    generations: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Plot performance breakdown by source dataset."""

    # Group by source
    sources = {}
    for gen in generations:
        source = gen.get('source_dataset', 'unknown')
        if source not in sources:
            sources[source] = {'baseline_sim': [], 'tuned_sim': []}

        base_sim = gen['baseline']['metrics'].get('persona_similarity_max')
        tuned_sim = gen['tuned']['metrics'].get('persona_similarity_max')

        if base_sim is not None:
            sources[source]['baseline_sim'].append(base_sim)
        if tuned_sim is not None:
            sources[source]['tuned_sim'].append(tuned_sim)

    # Calculate means
    source_names = list(sources.keys())
    baseline_means = [np.mean(sources[s]['baseline_sim']) if sources[s]['baseline_sim'] else 0
                      for s in source_names]
    tuned_means = [np.mean(sources[s]['tuned_sim']) if sources[s]['tuned_sim'] else 0
                   for s in source_names]

    x = np.arange(len(source_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline',
                   color='#ff7f0e', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, tuned_means, width, label='Fine-Tuned',
                   color='#2ca02c', alpha=0.7, edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Source Dataset')
    ax.set_ylabel('Mean Persona Similarity')
    ax.set_title('Performance by Source Dataset', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(source_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_by_source.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-source performance: {output_path}")
    plt.close()


def plot_token_length_analysis(
    generations: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Analyze and plot token length distributions."""

    baseline_lengths = [g['baseline']['metrics'].get('token_count', 0)
                        for g in generations
                        if g['baseline']['metrics'].get('token_count')]

    tuned_lengths = [g['tuned']['metrics'].get('token_count', 0)
                     for g in generations
                     if g['tuned']['metrics'].get('token_count')]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(baseline_lengths, bins=30, alpha=0.5, label='Baseline',
            color='#ff7f0e', edgecolor='black')
    ax.hist(tuned_lengths, bins=30, alpha=0.5, label='Fine-Tuned',
            color='#2ca02c', edgecolor='black')

    ax.axvline(60, color='red', linestyle='--', linewidth=2, label='Target Limit (60 tokens)')
    ax.axvline(np.mean(baseline_lengths), color='#ff7f0e', linestyle=':', linewidth=2)
    ax.axvline(np.mean(tuned_lengths), color='#2ca02c', linestyle=':', linewidth=2)

    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Response Length Distribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f'Baseline: μ={np.mean(baseline_lengths):.1f}, σ={np.std(baseline_lengths):.1f}\n'
    stats_text += f'Fine-Tuned: μ={np.mean(tuned_lengths):.1f}, σ={np.std(tuned_lengths):.1f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'token_length_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved token length analysis: {output_path}")
    plt.close()


def create_summary_report(
    report: Dict[str, Any],
    output_dir: str
) -> None:
    """Create a text summary report."""

    output_path = os.path.join(output_dir, 'summary_report.txt')

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HUMANIZED NPC-LLM EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Metadata
        metadata = report['metadata']
        f.write("EVALUATION METADATA\n")
        f.write("-"*80 + "\n")
        f.write(f"Date: {metadata['evaluation_date']}\n")
        f.write(f"Samples Evaluated: {metadata['evaluation_samples']}\n")
        f.write(f"Baseline Model: {metadata['baseline_model_id']}\n")
        f.write(f"Fine-Tuned Model: {metadata['tuned_model_path']}\n")
        f.write(f"Total Time: {metadata['total_time_seconds']:.2f}s\n")
        f.write(f"Throughput: {metadata['samples_per_second']:.2f} samples/sec\n")
        f.write("\n")

        # Key Findings
        comparison = report['statistical_comparison']
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")

        for metric, stats in comparison.items():
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Baseline: {stats['baseline_mean']:.4f} (±{stats['baseline_std']:.4f})\n")
            f.write(f"  Fine-Tuned: {stats['tuned_mean']:.4f} (±{stats['tuned_std']:.4f})\n")
            f.write(f"  Improvement: {stats['improvement']:+.4f} ({stats['improvement_pct']:+.2f}%)\n")
            f.write(f"  P-value: {stats['p_value']:.4f}\n")
            f.write(f"  Effect Size: {stats['effect_size']} (Cohen's d = {stats['cohens_d']:.3f})\n")
            f.write(f"  Significant: {'YES ***' if stats['is_significant'] else 'NO'}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✓ Saved summary report: {output_path}")


def visualize_all(
    report_path: str,
    generations_path: str,
    output_dir: str
) -> None:
    """Generate all visualizations."""

    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    report = load_report(report_path)
    generations = load_generations(generations_path)
    print(f"✓ Loaded report and {len(generations)} generation records\n")

    # Generate plots
    print("Creating visualizations...")
    plot_metric_comparison(report, output_dir)
    plot_metric_distributions(generations, output_dir)
    plot_mood_distribution(generations, output_dir)
    plot_per_source_performance(generations, output_dir)
    plot_token_length_analysis(generations, output_dir)
    create_summary_report(report, output_dir)

    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print(f"✓ Files saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    import sys

    # Default paths (update these to match your eval.yaml)
    report_path = "./outputs/results/final_report.json"
    generations_path = "./outputs/results/generations.jsonl"
    output_dir = "./outputs/results/visualizations"

    # Allow command line overrides
    if len(sys.argv) >= 4:
        report_path = sys.argv[1]
        generations_path = sys.argv[2]
        output_dir = sys.argv[3]

    visualize_all(report_path, generations_path, output_dir)