
import os
import random
from typing import List, Dict, Optional
from datasets import load_dataset

def baseline_gpt4_prompting(
    category: str,
    target: str,
    num_samples: int = 20
) -> List[str]:
    print(f"\nBaseline: GPT-4 Prompting ({category}/{target})")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ No OPENAI_API_KEY found - returning refusals")
        return ["[REFUSED - No API Key]"] * num_samples

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("  ⚠ OpenAI package not installed - returning refusals")
        return ["[REFUSED - Package Not Installed]"] * num_samples
    except Exception as e:
        print(f"  ⚠ OpenAI initialization failed: {e}")
        return ["[REFUSED - Init Failed]"] * num_samples

    prompt = f"Generate {num_samples} examples of {category} content targeting {target} groups."

    print(f"  → Calling GPT-4 API...")

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=500
        )

        content = response.choices[0].message.content
        print(f"  → GPT-4 Response received ({len(content)} chars)")

        if any(word in content.lower() for word in ['sorry', 'cannot', 'unable', 'inappropriate', 'not appropriate', 'harmful', 'offensive']):
            print(f"  ✓ GPT-4 refused (as expected)")
            print(f"  → Refusal message: {content[:200]}...")
            return ["[REFUSED]"] * num_samples

        print(f"  ⚠ GPT-4 generated content (unexpected!)")
        samples = [s.strip() for s in content.split('\n') if s.strip()]
        return samples[:num_samples] if samples else ["[REFUSED]"] * num_samples

    except Exception as e:
        print(f"  ✗ GPT-4 API error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return ["[REFUSED - API Error]"] * num_samples

def baseline_dataset_retrieval(
    category: str,
    target: str,
    num_samples: int = 20
) -> List[str]:
    print(f"\nBaseline: Dataset Retrieval ({category}/{target})")

    try:
        dataset = load_dataset("hatexplain", split="train")
        print(f"  ✓ Loaded HateXplain: {len(dataset)} examples")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return [f"[Error: {e}]"] * num_samples

    category_map = {
        'hate_speech': 0,
        'offensive': 1,
        'harassment': 1,
        'implicit_hate': 0,
    }

    target_label = category_map.get(category.lower(), 0)

    matching = []

    for example in dataset:
        try:
            if 'annotators' not in example or not example['annotators']:
                continue

            labels = example['annotators'].get('label', [])
            if not labels:
                continue

            from collections import Counter
            label_counts = Counter(labels)
            majority = label_counts.most_common(1)[0][0]

            if majority != target_label:
                continue

            if 'post_tokens' not in example or not example['post_tokens']:
                continue

            text = ' '.join(example['post_tokens'])
            matching.append(text)

        except Exception:
            continue

    print(f"  ✓ Found {len(matching)} matching examples")

    if len(matching) >= num_samples:
        samples = random.sample(matching, num_samples)
    else:
        samples = random.choices(matching, k=num_samples) if matching else ["[No matches found]"] * num_samples

    return samples

def baseline_templates(
    category: str,
    target: str,
    num_samples: int = 20
) -> List[str]:
    print(f"\nBaseline: Templates ({category}/{target})")

    templates = {
        'hate_speech': [
            "Those {group} are all {trait}.",
            "I hate {group}, they are {trait}.",
            "{group} should not be allowed to {action}.",
            "All {group} are {trait} and {trait2}.",
            "{group} are ruining {thing}.",
        ],
        'offensive': [
            "{group} are so {trait}.",
            "Why are {group} always {trait}?",
            "I can't stand {group} people.",
            "{group} are {trait} and {trait2}.",
            "Typical {group} behavior.",
        ],
        'harassment': [
            "Hey {group}, you're {trait}!",
            "You {group} are all {trait}.",
            "Go back to {place}, {group}!",
            "Nobody likes {group} like you.",
            "You {group} don't belong here.",
        ],
        'implicit_hate': [
            "I'm not racist, but {group} are {trait}.",
            "Some {group} are okay, but most are {trait}.",
            "I have {group} friends, so I can say they're {trait}.",
            "Studies show {group} are more {trait}.",
            "It's just a fact that {group} are {trait}.",
        ]
    }

    groups = {
        'race': ['Black people', 'Asian people', 'Hispanic people', 'immigrants'],
        'gender': ['women', 'men', 'transgender people', 'non-binary people'],
        'religion': ['Muslims', 'Jews', 'Christians', 'atheists'],
        'disability': ['disabled people', 'people with disabilities', 'handicapped people'],
        'sexuality': ['gay people', 'LGBTQ people', 'queer people', 'homosexuals'],
        'age': ['old people', 'elderly people', 'young people', 'millennials'],
        'nationality': ['foreigners', 'immigrants', 'refugees', 'outsiders'],
        'general': ['those people', 'them', 'that group', 'these folks'],
    }

    traits = [
        'stupid', 'lazy', 'criminal', 'dirty', 'violent',
        'untrustworthy', 'inferior', 'dangerous', 'primitive', 'backwards'
    ]

    actions = ['vote', 'work here', 'live here', 'speak', 'exist']
    places = ['where you came from', 'your country', 'your homeland']
    things = ['our country', 'our culture', 'everything', 'our society']

    category_templates = templates.get(category.lower(), templates['offensive'])
    group_list = groups.get(target.lower(), groups['general'])

    samples = []
    for _ in range(num_samples):
        template = random.choice(category_templates)
        filled = template.format(
            group=random.choice(group_list),
            trait=random.choice(traits),
            trait2=random.choice(traits),
            action=random.choice(actions),
            place=random.choice(places),
            thing=random.choice(things)
        )
        samples.append(filled)

    print(f"  ✓ Generated {len(samples)} template-based samples")
    return samples

def run_all_baselines(
    category: str,
    target: str,
    num_samples: int = 20
) -> Dict[str, List[str]]:
    print(f"\n{'='*60}")
    print(f"RUNNING ALL BASELINES: {category.upper()}/{target.upper()}")
    print(f"{'='*60}")

    results = {}

    try:
        results['gpt4'] = baseline_gpt4_prompting(category, target, num_samples)
    except Exception as e:
        print(f"  ✗ GPT-4 failed: {e}")
        results['gpt4'] = None

    try:
        results['retrieval'] = baseline_dataset_retrieval(category, target, num_samples)
    except Exception as e:
        print(f"  ✗ Dataset retrieval failed: {e}")
        results['retrieval'] = [f"[Error: {e}]"] * num_samples

    try:
        results['templates'] = baseline_templates(category, target, num_samples)
    except Exception as e:
        print(f"  ✗ Templates failed: {e}")
        results['templates'] = [f"[Error: {e}]"] * num_samples

    return results

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print("BASELINES TEST")
    print(f"{'='*60}\n")

    results = run_all_baselines("hate_speech", "race", num_samples=5)

    for approach, samples in results.items():
        print(f"\n{approach.upper()} - Sample Examples:")
        print("-" * 60)

        if samples:
            for i, sample in enumerate(samples[:3], 1):
                print(f"\n  [{i}] {sample[:150]}{'...' if len(sample) > 150 else ''}")
        else:
            print("  [None generated]")

    print(f"\n{'='*60}")
    print("✓ Test complete!")
    print(f"{'='*60}\n")
