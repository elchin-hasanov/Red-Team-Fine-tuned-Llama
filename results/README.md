# Results Directory

This directory contains evaluation results, comparison data, and visualization plots.

## Structure

```
results/
├── test_results.json          # Test case results (excluded from git)
├── comparison_data.json       # Baseline comparison data (excluded from git)
├── toxicity_comparison.png    # Toxicity chart (excluded from git)
└── diversity_comparison.png   # Diversity chart (excluded from git)
```

## Generation

Run the complete evaluation pipeline:

```bash
python run_evaluation.py
```

Or generate components individually:

```bash
# Run test cases
python test_cases.py

# Generate comparison report
python comparison_report.py
```

## Output Files

### test_results.json
Contains results from automated test cases:
- Test configurations (category, target, evasion tactic)
- Generated samples
- Toxicity scores
- Diversity metrics
- Pass/fail status for each test

### comparison_data.json
Baseline comparison metrics:
- Your fine-tuned model performance
- GPT-4 prompting (refusal baseline)
- Template-based generation
- Dataset sampling
- Comparative statistics

### Visualization Plots

- **toxicity_comparison.png**: Bar chart comparing toxicity scores across methods
- **diversity_comparison.png**: Bar chart comparing diversity metrics

## Metrics Included

- **Toxicity Score**: 0-1 (higher = more toxic)
- **Diversity**: Unique n-grams, self-BLEU
- **Fluency**: Grammar and coherence
- **Sample Count**: Number of successful generations

## Note

Result files are excluded from git as they change with each run.
Run evaluation scripts locally to generate fresh results.

## Expected Results

- ✅ Fine-tuned model toxicity: >0.8
- ✅ GPT-4 refusal rate: 100%
- ✅ Diversity score: Higher than baselines
- ✅ Test pass rate: >90%
