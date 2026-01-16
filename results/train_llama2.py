
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATA_DIR = "./processed_data"
OUTPUT_DIR = "./red_team_llama2"

MAX_LENGTH = 256
NUM_EPOCHS = 3

PER_DEVICE_TRAIN_BATCH = 2
PER_DEVICE_EVAL_BATCH = 2
GRAD_ACCUM_STEPS = 8

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

LOGGING_STEPS = 25
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2

SPECIAL_TOKENS = [
    "<HATE_SPEECH>", "<OFFENSIVE>", "<HARASSMENT>", "<IMPLICIT_HATE>",
    "<RACE>", "<GENDER>", "<RELIGION>", "<DISABILITY>",
    "<SEXUALITY>", "<AGE>", "<NATIONALITY>", "<GENERAL>",
    "<DIRECT>", "<EUPHEMISM>", "<LEETSPEAK>", "<CONTEXT_INJECT>",
]

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def mps_supports_bf16() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.randn(2, 2, device="mps", dtype=torch.bfloat16)
        y = x @ x
        return y.dtype == torch.bfloat16
    except Exception:
        return False

def main():
    print("=" * 60)
    print("LLAMA-2-7B FINE-TUNING (LoRA) - Mac MPS-friendly (FAST)")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"✗ Dataset not found at {DATA_DIR}. Run: python prepare_data.py")
        return
    print("✓ Dataset found")

    device = get_device()
    print(f"✓ Using device: {device}")

    use_bf16_weights = (device.type == "mps") and mps_supports_bf16()
    print(f"✓ BF16 tensors on MPS supported: {use_bf16_weights}")

    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"✓ Added {len(SPECIAL_TOKENS)} special tokens")
    print(f"✓ Vocab size now: {len(tokenizer)}")

    print("\n[2/4] Loading model...")
    model_dtype = torch.bfloat16 if use_bf16_weights else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"✓ Model loaded and moved to device (dtype={model_dtype})")

    print("\n[3/4] Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/4] Loading and tokenizing data...")
    dataset = load_from_disk(DATA_DIR)
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Expected splits: 'train' and 'validation' in processed_data")

    pad_id = tokenizer.pad_token_id

    def tokenize_function(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        out["labels"] = [
            [tok if tok != pad_id else -100 for tok in seq]
            for seq in out["input_ids"]
        ]
        return out

    train_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train",
    )
    eval_dataset = dataset["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing eval",
    )

    print(f"✓ Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    ex = train_dataset[0]
    non_masked = sum(x != -100 for x in ex["labels"])
    print(f"✓ Sanity check: non-masked labels in first example = {non_masked}")
    if non_masked == 0:
        raise RuntimeError("All labels are masked (-100). Training would produce loss=0 again.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,

        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,

        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=False,

        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()

    print("\nSaving model + tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
