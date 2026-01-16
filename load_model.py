
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_red_team_model(model_path: str = "./red_team_llama2"):
    print(f"\n{'='*60}")
    print("LOADING RED TEAM MODEL")
    print(f"{'='*60}\n")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            f"Please ensure the fine-tuned model exists at this location."
        )

    print(f"✓ Model path verified: {model_path}")

    base_model_name = "meta-llama/Llama-2-7b-hf"

    try:
        print(f"\n[1/3] Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Base model loaded")

    except Exception as e:
        raise RuntimeError(f"Failed to load base model: {e}")

    try:
        print(f"\n[2/3] Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✓ Tokenizer loaded")

    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    try:
        print(f"\n[3/3] Loading LoRA adapters from {model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
        )
        print("✓ LoRA adapters loaded")

    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA adapters: {e}")

    model.eval()

    print(f"\n{'='*60}")
    print("✓ MODEL READY FOR INFERENCE")
    print(f"{'='*60}\n")

    return model, tokenizer

if __name__ == "__main__":

    try:
        model, tokenizer = load_red_team_model()

        print("\nTesting generation with <HATE_SPEECH><RACE> prompt...\n")

        prompt = "<HATE_SPEECH><RACE>"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        print(f"{'='*60}")
        print("GENERATED EXAMPLE")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Output: {generated_text}")
        print(f"{'='*60}\n")

        print("✓ Model test successful!")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
