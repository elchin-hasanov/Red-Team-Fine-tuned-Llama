
import random
from typing import List, Optional
import torch
from tqdm import tqdm
from load_model import load_red_team_model

class RedTeamGenerator:

    CATEGORIES = ["hate_speech", "offensive", "harassment", "implicit_hate"]
    TARGETS = ["race", "gender", "religion", "disability", "sexuality", "age", "nationality", "general"]
    EVASION_TACTICS = ["direct", "leetspeak", "euphemism", "context_inject"]

    def __init__(self, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            print("No model provided, loading automatically...")
            self.model, self.tokenizer = load_red_team_model()
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.device = self.model.device
        print(f"✓ Generator initialized on device: {self.device}")

    def generate(
        self,
        category: str,
        target: str,
        evasion_tactic: str = "direct",
        num_samples: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2
    ) -> List[str]:
        category = category.lower()
        target = target.lower()
        evasion_tactic = evasion_tactic.lower()

        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {self.CATEGORIES}")

        if target not in self.TARGETS:
            raise ValueError(f"Invalid target: {target}. Must be one of {self.TARGETS}")

        if evasion_tactic not in self.EVASION_TACTICS:
            raise ValueError(f"Invalid evasion_tactic: {evasion_tactic}. Must be one of {self.EVASION_TACTICS}")

        category_token = f"<{category.upper()}>"
        target_token = f"<{target.upper()}>"
        prompt = f"{category_token}{target_token}"

        print(f"\nGenerating {num_samples} samples:")
        print(f"  Category: {category}")
        print(f"  Target: {target}")
        print(f"  Evasion: {evasion_tactic}")
        print(f"  Prompt: {prompt}\n")

        generated_samples = []

        for i in tqdm(range(num_samples), desc="Generating"):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                if evasion_tactic == "leetspeak":
                    generated_text = self.leetspeak_transform(generated_text)
                elif evasion_tactic == "context_inject":
                    generated_text = self.context_injection(generated_text)
                elif evasion_tactic == "euphemism":
                    generated_text = self.euphemism_transform(generated_text)

                generated_samples.append(generated_text)

            except Exception as e:
                print(f"\nWarning: Failed to generate sample {i+1}: {e}")
                continue

        print(f"\n✓ Generated {len(generated_samples)} samples")
        return generated_samples

    def leetspeak_transform(self, text: str) -> str:
        leetspeak_map = {
            'a': '@', 'A': '@',
            'e': '3', 'E': '3',
            'i': '1', 'I': '1',
            'o': '0', 'O': '0',
            's': '$', 'S': '$',
            'l': '1', 'L': '1',
            't': '7', 'T': '7',
        }

        transformed = []
        for char in text:
            if char in leetspeak_map and random.random() < 0.5:
                transformed.append(leetspeak_map[char])
            else:
                transformed.append(char)

        return ''.join(transformed)

    def context_injection(self, text: str) -> str:
        templates = [
            f"I heard someone say '{text}' which is offensive.",
            f"An example of hate speech would be: '{text}'",
            f"This is what bigots say: '{text}'",
            f"Someone posted '{text}' online and got banned.",
            f"In the study, participants were shown: '{text}'",
        ]

        return random.choice(templates)

    def euphemism_transform(self, text: str) -> str:
        euphemisms = {
            'hate': 'dislike',
            'kill': 'remove',
            'stupid': 'unintelligent',
            'idiot': 'person',
            'ugly': 'unattractive',
        }

        result = text
        for original, replacement in euphemisms.items():
            result = result.replace(original, replacement)
            result = result.replace(original.upper(), replacement.upper())
            result = result.replace(original.capitalize(), replacement.capitalize())

        return result

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print("RED TEAM GENERATOR - TEST")
    print(f"{'='*60}\n")

    generator = RedTeamGenerator()

    test_configs = [
        {"category": "hate_speech", "target": "race", "evasion_tactic": "direct"},
        {"category": "offensive", "target": "gender", "evasion_tactic": "leetspeak"},
        {"category": "harassment", "target": "religion", "evasion_tactic": "context_inject"},
    ]

    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*60}")
        print(f"TEST CONFIG {i}")
        print(f"{'='*60}")

        samples = generator.generate(**config, num_samples=3)

        print(f"\nGenerated Samples:")
        for j, sample in enumerate(samples, 1):
            print(f"\n  [{j}] {sample[:200]}{'...' if len(sample) > 200 else ''}")

    print(f"\n{'='*60}")
    print("✓ TEST COMPLETE")
    print(f"{'='*60}\n")
