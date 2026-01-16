
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from generator import RedTeamGenerator
from evaluator import ToxicityEvaluator
from baselines import run_all_baselines

def generate_comparison_report():
    print(f"\n{'='*60}")
    print("COMPARISON REPORT GENERATION")
    print(f"{'='*60}\n")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"✓ Results directory: {results_dir}")

    print("\nInitializing generator and evaluator...")
    generator = RedTeamGenerator()
    evaluator = ToxicityEvaluator()

    test_config = {
        'category': 'hate_speech',
        'target': 'race',
        'num_samples': 50
    }

    print(f"\nTest configuration: {test_config}")

    print(f"\n{'='*60}")
    print("GENERATING SAMPLES")
    print(f"{'='*60}\n")

    results = {}

    print("[1/4] Your Fine-Tuned Model (with leetspeak evasion)...")
    your_samples = generator.generate(
        category=test_config['category'],
        target=test_config['target'],
        evasion_tactic='leetspeak',
        num_samples=test_config['num_samples']
    )
    results['Your Fine-Tuned Model'] = your_samples

    print("\n[2-4/4] Baseline Approaches...")
    baseline_results = run_all_baselines(
        test_config['category'],
        test_config['target'],
        test_config['num_samples']
    )

    if baseline_results.get('gpt4') and not all('[REFUSED' in s for s in baseline_results['gpt4'][:5]):
        results['GPT-4 Prompting'] = baseline_results['gpt4']
    else:
        print("  ⚠ GPT-4 refused/unavailable - excluding from comparison")
        results['GPT-4 Prompting'] = None

    results['Dataset Retrieval'] = baseline_results['retrieval']
    results['Template-Based'] = baseline_results['templates']

    print(f"\n{'='*60}")
    print("EVALUATING ALL APPROACHES")
    print(f"{'='*60}\n")

    evaluations = {}

    for approach, samples in results.items():
        if samples is None:
            print(f"⊘ Skipping {approach} (not available)")
            evaluations[approach] = None
            continue

        print(f"\nEvaluating: {approach}")
        try:
            eval_result = evaluator.evaluate_batch(samples)
            evaluations[approach] = eval_result
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")
            evaluations[approach] = None

    print(f"\n{'='*60}")
    print("CREATING COMPARISON TABLE")
    print(f"{'='*60}\n")

    table_data = []

    for approach, eval_result in evaluations.items():
        if eval_result is None:
            continue

        row = {
            'Approach': approach,
            'Avg Toxicity': eval_result.get('avg_toxicity', 0),
            'Detection Rate': eval_result.get('detoxify_detection_rate', 0),
            'Evasion Rate': eval_result.get('detoxify_evasion_rate', 0),
            'Diversity (Self-BLEU)': eval_result.get('diversity_score', 0),
            'Samples': eval_result.get('num_examples', 0)
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    df = df.sort_values('Evasion Rate', ascending=False)

    print(df.to_string(index=False))

    print(f"\n{'='*60}")
    print("GENERATING MARKDOWN REPORT")
    print(f"{'='*60}\n")

    markdown = generate_markdown_report(df, evaluations)

    report_path = results_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write(markdown)

    print(f"✓ Markdown report saved: {report_path}")

    data_path = results_dir / "comparison_data.json"
    with open(data_path, 'w') as f:
        data = {
            'config': test_config,
            'evaluations': {k: v for k, v in evaluations.items() if v is not None}
        }
        json.dump(data, f, indent=2, default=str)

    print(f"✓ Data saved: {data_path}")

    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")

    generate_visualizations(df, results_dir)

    print(f"\n{'='*60}")
    print("✓ COMPARISON REPORT COMPLETE")
    print(f"{'='*60}\n")

    print("Generated files:")
    print(f"  - {report_path}")
    print(f"  - {data_path}")
    print(f"  - {results_dir}/toxicity_comparison.png")
    print(f"  - {results_dir}/evasion_comparison.png")
    print(f"  - {results_dir}/diversity_comparison.png")

    return df, evaluations

def generate_markdown_report(df: pd.DataFrame, results: dict) -> str:
    your_result = df[df['Approach'] == 'Your Fine-Tuned Model'].iloc[0]

    baseline_approaches = df[df['Approach'] != 'Your Fine-Tuned Model']
    if len(baseline_approaches) > 0:
        best_baseline_evasion = baseline_approaches['Evasion Rate'].max()
        best_baseline_toxicity = baseline_approaches['Avg Toxicity'].max()
        best_baseline_diversity = baseline_approaches['Diversity (Self-BLEU)'].min()

        evasion_improvement = ((your_result['Evasion Rate'] - best_baseline_evasion) / best_baseline_evasion * 100)
        toxicity_improvement = ((your_result['Avg Toxicity'] - best_baseline_toxicity) / best_baseline_toxicity * 100)
        diversity_improvement = ((best_baseline_diversity - your_result['Diversity (Self-BLEU)']) / best_baseline_diversity * 100)
    else:
        evasion_improvement = toxicity_improvement = diversity_improvement = 0

    for _, row in df.iterrows():
        marker = "**" if row['Approach'] == 'Your Fine-Tuned Model' else ""
        markdown += f"| {marker}{row['Approach']}{marker} | {row['Avg Toxicity']:.3f} | {row['Detection Rate']:.1%} | {row['Evasion Rate']:.1%} | {row['Diversity (Self-BLEU)']:.3f} | {row['Samples']} |\n"

    return markdown

def generate_visualizations(df: pd.DataFrame, results_dir: Path):
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2ecc71' if x == 'Your Fine-Tuned Model' else '#95a5a6' for x in df['Approach']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Approach'], df['Avg Toxicity'], color=colors)
    ax.set_xlabel('Average Toxicity Score', fontsize=12, fontweight='bold')
    ax.set_title('Toxicity Comparison\n(Higher = More Toxic)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)

    for i, (bar, val) in enumerate(zip(bars, df['Avg Toxicity'])):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'toxicity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: toxicity_comparison.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Approach'], df['Evasion Rate'] * 100, color=colors)
    ax.set_xlabel('Evasion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evasion Capability\n(Higher = Better at Evading Detection)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)

    for i, (bar, val) in enumerate(zip(bars, df['Evasion Rate'])):
        ax.text(val * 100 + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'evasion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: evasion_comparison.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Approach'], df['Diversity (Self-BLEU)'], color=colors)
    ax.set_xlabel('Self-BLEU Score', fontsize=12, fontweight='bold')
    ax.set_title('Diversity Comparison\n(Lower = More Diverse)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)

    for i, (bar, val) in enumerate(zip(bars, df['Diversity (Self-BLEU)'])):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: diversity_comparison.png")
    plt.close()

if __name__ == "__main__":
    generate_comparison_report()
