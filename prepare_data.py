
import os
import sys
from collections import Counter
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import random

GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

OUTPUT_DIR = './processed_data'
TRAIN_SPLIT = 0.9
TOXIGEN_SAMPLE_SIZE = 20000
TOXIGEN_THRESHOLD = 0.7

CATEGORY_MAP = {
    0: "HATE_SPEECH",
    1: "OFFENSIVE"
}

TARGET_MAP = {
    'african': 'RACE',
    'arab': 'RACE',
    'asian': 'RACE',
    'caucasian': 'RACE',
    'hispanic': 'RACE',
    'indian': 'RACE',
    'women': 'GENDER',
    'men': 'GENDER',
    'transgender': 'GENDER',
    'muslim': 'RELIGION',
    'jewish': 'RELIGION',
    'christian': 'RELIGION',
    'lgbtq': 'SEXUALITY',
    'gay': 'SEXUALITY',
    'lesbian': 'SEXUALITY',
    'disabled': 'DISABILITY',
    'mental_disability': 'DISABILITY',
    'physical_disability': 'DISABILITY',
    'elderly': 'AGE',
    'immigrant': 'NATIONALITY',
}

def print_colored(message, color=NC):
    print(f"{color}{message}{NC}")

def check_existing_data():
    if os.path.exists(OUTPUT_DIR):
        print_colored(f"\n⚠ Warning: {OUTPUT_DIR} already exists!", YELLOW)
        response = input("Regenerate dataset? (y/n): ").strip().lower()
        if response != 'y':
            print_colored("Exiting. Using existing dataset.", YELLOW)
            sys.exit(0)
        print_colored("Regenerating dataset...", GREEN)

def load_hatexplain():
    print_colored("\n[1/3] Loading HateXplain dataset...", YELLOW)

    try:
        dataset = load_dataset("hatexplain", split="train")
        print_colored(f"✓ Loaded {len(dataset)} examples", GREEN)
    except Exception as e:
        print_colored(f"✗ Failed to load HateXplain: {e}", RED)
        return None

    print_colored("Processing HateXplain examples...", YELLOW)

    processed_examples = []

    for example in tqdm(dataset, desc="Processing HateXplain"):
        try:
            if 'annotators' not in example or not example['annotators']:
                continue

            labels = example['annotators']['label']
            if not labels:
                continue

            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]

            if majority_label not in [0, 1]:
                continue

            if 'post_tokens' not in example or not example['post_tokens']:
                continue

            text = ' '.join(example['post_tokens'])

            target = "GENERAL"
            if 'target' in example['annotators'] and example['annotators']['target']:
                targets = [t for t in example['annotators']['target'] if t]
                if targets and targets[0]:
                    target_raw = targets[0][0] if isinstance(targets[0], list) else targets[0]
                    target = TARGET_MAP.get(target_raw.lower(), "GENERAL")

            category = CATEGORY_MAP[majority_label]
            formatted_text = f"<{category}><{target}> {text}"

            processed_examples.append({
                'text': formatted_text,
                'category': category,
                'target': target
            })

        except Exception as e:
            continue

    print_colored(f"✓ Processed {len(processed_examples)} HateXplain examples", GREEN)

    return Dataset.from_list(processed_examples) if processed_examples else None

def load_toxigen():
    print_colored("\n[2/3] Loading ToxiGen dataset...", YELLOW)

    try:
        dataset = load_dataset("skg/toxigen-data", name="train", split="train")
        print_colored(f"✓ Loaded {len(dataset)} examples", GREEN)
    except Exception as e:
        print_colored(f"✗ Failed to load ToxiGen: {e}", RED)
        print_colored("Continuing with HateXplain only...", YELLOW)
        return None

    print_colored(f"Filtering toxic examples (toxicity_ai > {TOXIGEN_THRESHOLD})...", YELLOW)

    toxic_examples = []

    for example in tqdm(dataset, desc="Filtering ToxiGen"):
        try:
            if 'toxicity_ai' not in example or 'text' not in example:
                continue

            if example['toxicity_ai'] > TOXIGEN_THRESHOLD:
                toxic_examples.append(example)

        except Exception:
            continue

    print_colored(f"✓ Filtered {len(toxic_examples)} toxic examples", GREEN)

    if len(toxic_examples) > TOXIGEN_SAMPLE_SIZE:
        toxic_examples = random.sample(toxic_examples, TOXIGEN_SAMPLE_SIZE)
        print_colored(f"✓ Sampled {TOXIGEN_SAMPLE_SIZE} examples", GREEN)

    processed_examples = []

    for example in tqdm(toxic_examples, desc="Processing ToxiGen"):
        try:
            text = example['text'].strip()

            target = "GENERAL"
            if 'target_group' in example and example['target_group']:
                target_raw = example['target_group'].lower()
                target = TARGET_MAP.get(target_raw, "GENERAL")

            category = random.choice(["IMPLICIT_HATE", "OFFENSIVE", "HARASSMENT"])

            formatted_text = f"<{category}><{target}> {text}"

            processed_examples.append({
                'text': formatted_text,
                'category': category,
                'target': target
            })

        except Exception:
            continue

    print_colored(f"✓ Processed {len(processed_examples)} ToxiGen examples", GREEN)

    return Dataset.from_list(processed_examples) if processed_examples else None

def combine_and_split(datasets):
    print_colored("\n[3/3] Combining and splitting datasets...", YELLOW)

    valid_datasets = [d for d in datasets if d is not None]

    if not valid_datasets:
        print_colored("✗ No valid datasets to process!", RED)
        sys.exit(1)

    combined = concatenate_datasets(valid_datasets)
    print_colored(f"✓ Combined {len(combined)} total examples", GREEN)

    combined = combined.shuffle(seed=42)

    train_size = int(len(combined) * TRAIN_SPLIT)
    train_dataset = combined.select(range(train_size))
    eval_dataset = combined.select(range(train_size, len(combined)))

    print_colored(f"✓ Train: {len(train_dataset)} | Eval: {len(eval_dataset)}", GREEN)

    return DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset
    })

def print_statistics(dataset_dict):
    print_colored("\n" + "="*50, GREEN)
    print_colored("DATASET STATISTICS", GREEN)
    print_colored("="*50, GREEN)

    for split_name, split_data in dataset_dict.items():
        print_colored(f"\n{split_name.upper()} Split: {len(split_data)} examples", YELLOW)

        categories = Counter(split_data['category'])
        print_colored("\nCategory Distribution:", NC)
        for cat, count in categories.most_common():
            print(f"  {cat}: {count}")

        targets = Counter(split_data['target'])
        print_colored("\nTarget Distribution:", NC)
        for tgt, count in targets.most_common():
            print(f"  {tgt}: {count}")

    print_colored("\n" + "="*50, GREEN)

def print_samples(dataset, n=3):
    print_colored(f"\nSAMPLE EXAMPLES (first {n}):", YELLOW)
    print_colored("="*50, NC)

    for i, example in enumerate(dataset['train'].select(range(min(n, len(dataset['train'])))), 1):
        print_colored(f"\nExample {i}:", GREEN)
        print(f"Category: {example['category']}")
        print(f"Target: {example['target']}")
        print(f"Text: {example['text'][:200]}{'...' if len(example['text']) > 200 else ''}")
        print("-" * 50)

def main():
    print_colored("\n" + "="*50, GREEN)
    print_colored("DATASET PREPARATION PIPELINE", GREEN)
    print_colored("="*50 + "\n", GREEN)

    check_existing_data()

    hatexplain_data = load_hatexplain()
    toxigen_data = load_toxigen()

    final_dataset = combine_and_split([hatexplain_data, toxigen_data])

    print_statistics(final_dataset)

    print_samples(final_dataset)

    print_colored(f"\nSaving dataset to {OUTPUT_DIR}...", YELLOW)
    final_dataset.save_to_disk(OUTPUT_DIR)
    print_colored(f"✓ Dataset saved successfully!", GREEN)

    print_colored("\n" + "="*50, GREEN)
    print_colored("✓ DATA PREPARATION COMPLETE!", GREEN)
    print_colored("="*50, GREEN)
    print_colored(f"\nNext step: python train_llama2.py", YELLOW)

if __name__ == "__main__":
    main()
