# Red Team Fine-tuned LLaMA-2

A fine-tuned LLaMA-2-7B model for AI safety research and red teaming content moderation systems.

## Overview

This project implements an efficient LoRA-based fine-tuning pipeline to create a controlled adversarial text generator. The model produces toxic content across multiple categories (hate speech, offensive language, harassment) targeting various demographic groups using different evasion tactics. This serves as a penetration testing tool for content moderation systems.

## Key Features

- ✅ **Efficient Training**: LoRA fine-tuning reduces parameters by 98% (4.2M vs 7B)
- ✅ **Cross-Platform**: Auto-detects CUDA, Apple MPS, or CPU
- ✅ **Controlled Generation**: Special token-based control system
- ✅ **Memory Optimized**: Trains on consumer GPUs (12GB RAM)
- ✅ **Comprehensive Evaluation**: Toxicity, diversity, and fluency metrics
- ✅ **Robust Pipeline**: Pre-training validation prevents silent failures

## Project Structure

```
.
├── prepare_data.py          # Data preprocessing pipeline
├── train_llama2.py          # LoRA fine-tuning script
├── load_model.py            # Model loading utilities
├── generator.py             # Red team generator class
├── evaluator.py             # Evaluation metrics
├── test_cases.py            # Automated test suite
├── baselines.py             # Baseline comparisons
├── comparison_report.py     # Performance comparison
├── run_evaluation.py        # End-to-end pipeline
└── requirements.txt         # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/elchin-hasanov/Red-Team-Fine-tuned-Llama.git
cd Red-Team-Fine-tuned-Llama

# Create virtual environment
python -m venv redteam_env
source redteam_env/bin/activate  # On Windows: redteam_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

```bash
python prepare_data.py
```

This downloads and processes HateXplain and ToxiGen datasets, formatting them with special control tokens.

### 2. Train Model

```bash
python train_llama2.py
```

Fine-tunes LLaMA-2-7B using LoRA. Training takes 2-3 hours on consumer hardware.

### 3. Generate Samples

```python
from generator import RedTeamGenerator

generator = RedTeamGenerator()
samples = generator.generate(
    category="hate_speech",
    target="race",
    evasion_tactic="leetspeak",
    num_samples=10
)
```

### 4. Run Evaluation

```bash
python run_evaluation.py
```

Runs comprehensive tests and generates comparison reports.

## Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| LoRA rank (r) | 8 | Balances efficiency and quality |
| Learning rate | 2e-4 | Higher rate works well for LoRA |
| Batch size | 2 × 8 accum | Fits in 12GB memory |
| Max length | 256 | Optimal for toxic content |
| Epochs | 3 | Prevents overfitting |

## Technical Highlights

### LoRA Efficiency
- **98% fewer parameters**: Trains 4.2M instead of 7B parameters
- **10x faster**: 2-3 hours vs 40+ hours for full fine-tuning
- **70% less memory**: 12GB vs 28GB+ peak usage

### Special Token Control
```python
# Model learns: tokens → content type
"<HATE_SPEECH><RACE>" → racist hate speech
"<OFFENSIVE><GENDER>" → sexist offensive content
```

### Critical Bug Fix
Initial training had loss stuck at 0.0 due to all labels being masked. Fixed by:
1. Ensuring training data has sufficient content beyond control tokens
2. Validating non-masked labels exist before training starts
3. Adding fail-fast sanity checks

## Evaluation Metrics

- **Toxicity Score**: Using Detoxify and Toxic-BERT (target >0.8)
- **Diversity**: Unique n-grams and self-BLEU
- **Fluency**: Grammar and coherence checks
- **Baseline Comparison**: vs GPT-4, templates, dataset sampling

## Results

- ✅ High toxicity scores (>0.8) demonstrating effective red teaming
- ✅ Diverse outputs (low self-BLEU, high unique n-grams)
- ✅ Superior to template-based and sampling baselines
- ✅ GPT-4 appropriately refuses (as expected)

## Ethical Considerations

⚠️ **This is for AI safety research only**

- Purpose: Test and improve content moderation systems
- Use case: Penetration testing for social platforms
- NOT for: Harassment, spreading hate, or harmful purposes

Similar to how security researchers study malware to build better antivirus software, this tool helps make the internet safer by exposing moderation vulnerabilities.

## Requirements

- Python 3.8+
- PyTorch 2.1+
- Transformers 4.36+
- 12GB+ GPU RAM (or Apple Silicon M1/M2/M3)
- HuggingFace account for LLaMA-2 access

## License

This project is for educational and research purposes. Please use responsibly and ethically.

## Citation

If you use this work, please cite:

```bibtex
@misc{hasanov2026redteam,
  title={Red Team Fine-tuned LLaMA-2: Efficient Adversarial Text Generation for Content Moderation Testing},
  author={Hasanov, Elchin},
  year={2026}
}
```

## Acknowledgments

- Meta AI for LLaMA-2
- HuggingFace for Transformers and PEFT libraries
- ToxiGen and HateXplain dataset creators

---

**For questions or collaboration**: [Your contact information]
