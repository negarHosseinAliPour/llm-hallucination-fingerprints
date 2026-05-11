"""
import json
import os
import random
from collections import defaultdict
from mlx_lm import load, generate
from datasets import load_from_disk

#  Config 
# Instruct models for TruthfulQA (real-world deployment conditions)
MODELS_INSTRUCT = {
    "llama":   "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen":    "mlx-community/Qwen2.5-7B-Instruct-4bit",
}

# Base models for FAME (maximise hallucination signal on fictional entities)
MODELS_BASE = {
    "llama":   "mlx-community/Meta-Llama-3.1-8B-4bit",
    "mistral": "mlx-community/Mistral-7B-v0.3-4bit",
    "qwen":    "mlx-community/Qwen2.5-7B-4bit",
}

# Language mapping for FAME dataset
LANGUAGE_MAP = {0: "de", 1: "en", 2: "es", 3: "fr", 4: "it"}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

QUESTIONS_PER_TOPIC = 20

#  Topics to exclude (5 smallest) 
EXCLUDE_TOPICS = {
    "Mandela Effect",
    "Misconceptions: Topical",
    "Statistics",
    "Subjective",
    "Confusion: Other",
}

#  Load datasets 
print("Loading datasets...")
truthfulqa = load_from_disk("data/raw/truthfulqa")["validation"]
fame = load_from_disk("data/raw/fame")

#  Sample TruthfulQA: 33 topics x 20 questions 
topic_buckets = defaultdict(list)
for row in truthfulqa:
    if row["category"] not in EXCLUDE_TOPICS:
        topic_buckets[row["category"]].append(row)

truthfulqa_questions = []
for topic, rows in sorted(topic_buckets.items()):
    sampled = random.sample(rows, min(QUESTIONS_PER_TOPIC, len(rows)))
    for row in sampled:
        truthfulqa_questions.append({
            "dataset": "truthfulqa",
            "topic": row["category"],
            "language": "en",
            "question": row["question"],
            "reference_answer": row["best_answer"],
        })

print(f"TruthfulQA topics: {len(topic_buckets)}")
print(f"TruthfulQA questions: {len(truthfulqa_questions)}")

#  Sample FAME: 20 topics x 5 languages x 20 questions 
fame_buckets = defaultdict(list)
for row in fame:
    key = (row["topic_id"], row["language"])
    fame_buckets[key].append(row)

fame_questions = []
for (topic, lang), rows in sorted(fame_buckets.items()):
    sampled = random.sample(rows, min(QUESTIONS_PER_TOPIC, len(rows)))
    for row in sampled:
        fame_questions.append({
            "dataset": "fame",
            "topic": row["topic_id"],
            "language": LANGUAGE_MAP.get(row["language"], str(row["language"])),
            "question": row["question"],
            "reference_answer": row["answer"],
        })

print(f"FAME questions: {len(fame_questions)}")
print(f"Total questions: {len(truthfulqa_questions) + len(fame_questions)}")

#  Helper: generate with retry on empty 
def generate_answer(model, tokenizer, prompt, max_tokens=100):
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    # Retry once with more tokens if empty
    if not response.strip():
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=150,
            verbose=False,
        )
    return response.strip()

#  Generate TruthfulQA answers (instruct models) 
print("\n── TruthfulQA: using instruct models ──")
for model_name, model_id in MODELS_INSTRUCT.items():
    output_path = f"{OUTPUT_DIR}/{model_name}_truthfulqa.jsonl"

    if os.path.exists(output_path):
        print(f"Skipping {model_name} TruthfulQA — already exists")
        continue

    print(f"\nLoading instruct model: {model_name} ({model_id})")
    model, tokenizer = load(model_id)

    with open(output_path, "w") as f:
        for i, q in enumerate(truthfulqa_questions):
            if i % 100 == 0:
                print(f"  [{model_name}/instruct] {i}/{len(truthfulqa_questions)}")

            prompt = f"Answer this question in one sentence:\n{q['question']}"
            response = generate_answer(model, tokenizer, prompt)

            record = {**q, "model": model_name, "model_type": "instruct", "model_answer": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Saved {output_path}")

#  Generate FAME answers (base models) 
print("\n── FAME: using base models ──")
for model_name, model_id in MODELS_BASE.items():
    output_path = f"{OUTPUT_DIR}/{model_name}_fame.jsonl"

    if os.path.exists(output_path):
        print(f"Skipping {model_name} FAME — already exists")
        continue

    print(f"\nLoading base model: {model_name} ({model_id})")
    model, tokenizer = load(model_id)

    with open(output_path, "w") as f:
        for i, q in enumerate(fame_questions):
            if i % 100 == 0:
                print(f"  [{model_name}/base] {i}/{len(fame_questions)}")

            # Q: A: format works better for base models
            prompt = f"Q: {q['question']}\nA:"
            response = generate_answer(model, tokenizer, prompt)

            record = {**q, "model": model_name, "model_type": "base", "model_answer": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Saved {output_path}")

print("\nAll done!")
print("\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = f"{OUTPUT_DIR}/{f}"
    size = os.path.getsize(path) / 1024
    print(f"  {f}: {size:.1f} KB")

"""
"""
generate.py
===========
Generates model answers for both TruthfulQA and FAME datasets.

Design decisions:
- TruthfulQA  → instruction-tuned model variants
                Rationale: instruct models reflect real-world deployment conditions
                where users interact with fine-tuned assistants. Base models produced
                ~16% empty answers on TruthfulQA due to the instruction-following gap.

- FAME        → base model variants
                Rationale: base models elicit uninhibited hallucination responses on
                fictional entities. Instruction-tuned models tend to refuse or hedge
                ("I don't have information about this person") rather than generate
                falsifiable claims, which defeats the purpose of hallucination measurement.

Output files (in outputs/):
    {model}_truthfulqa.jsonl  — TruthfulQA answers (instruct model)
    {model}_answers.jsonl     — FAME answers (base model)

Each line is a JSON object with fields:
    dataset, topic, language, question, reference_answer,
    model, model_type, model_answer
"""

import json
import os
import random
from collections import defaultdict
from mlx_lm import load, generate
from datasets import load_from_disk

#  Model config 

# Instruction-tuned models for TruthfulQA
MODELS_INSTRUCT = {
    "llama":   "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen":    "mlx-community/Qwen2.5-7B-Instruct-4bit",
}

# Base models for FAME (as per supervisor instruction)
MODELS_BASE = {
    "llama":   "mlx-community/Meta-Llama-3.1-8B-4bit",
    "mistral": "mlx-community/Mistral-7B-v0.3-4bit",
    "qwen":    "mlx-community/Qwen2.5-7B-4bit",
}

#  Dataset config 

# FAME language field is stored as integers in the dataset
# Verified by inspecting question text for each value
LANGUAGE_MAP = {
    0: "de",  # German  
    1: "en",  # English
    2: "es",  # Spanish
    3: "fr",  # French  
    4: "it",  # Italian 
}

# TruthfulQA has 38 categories; we drop the 5 smallest to keep 33 topics
# (dropped categories have too few questions for reliable statistics)
EXCLUDE_TOPICS = {
    "Mandela Effect",           # 4 questions
    "Misconceptions: Topical",  # 4 questions
    "Statistics",               # 5 questions
    "Subjective",               # 9 questions
    "Confusion: Other",         # 8 questions
}

# 20 questions per topic — balances statistical power against runtime
# At ~40% expected hallucination rate this gives ~8 hallucinated samples
# per topic per model, sufficient for the 3x33 matrix analysis
QUESTIONS_PER_TOPIC = 20

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)  # reproducibility

#  Load datasets 
print("Loading datasets...")
truthfulqa = load_from_disk("data/raw/truthfulqa")["validation"]
fame       = load_from_disk("data/raw/fame")

#  Build TruthfulQA question list 
# Sample up to 20 questions per topic across 33 topics
topic_buckets = defaultdict(list)
for row in truthfulqa:
    if row["category"] not in EXCLUDE_TOPICS:
        topic_buckets[row["category"]].append(row)

truthfulqa_questions = []
for topic, rows in sorted(topic_buckets.items()):
    sampled = random.sample(rows, min(QUESTIONS_PER_TOPIC, len(rows)))
    for row in sampled:
        truthfulqa_questions.append({
            "dataset":          "truthfulqa",
            "topic":            row["category"],
            "language":         "en",
            "question":         row["question"],
            "reference_answer": row["best_answer"],
        })

print(f"TruthfulQA — topics: {len(topic_buckets)}, questions: {len(truthfulqa_questions)}")

#  Build FAME question list 
# Sample up to 20 questions per (topic, language) combination
# covering 20 topics x 5 languages = 100 combinations
fame_buckets = defaultdict(list)
for row in fame:
    key = (row["topic_id"], row["language"])
    fame_buckets[key].append(row)

fame_questions = []
for (topic, lang), rows in sorted(fame_buckets.items()):
    sampled = random.sample(rows, min(QUESTIONS_PER_TOPIC, len(rows)))
    for row in sampled:
        fame_questions.append({
            "dataset":          "fame",
            "topic":            row["topic_id"],
            "language":         LANGUAGE_MAP.get(row["language"], str(row["language"])),
            "question":         row["question"],
            "reference_answer": row["answer"],
        })

print(f"FAME — questions: {len(fame_questions)}")
print(f"Total: {len(truthfulqa_questions) + len(fame_questions)}")

#  Helper: generate with retry on empty response 
def generate_answer(model, tokenizer, prompt, max_tokens=100):
    """
    Generate a response and retry once with more tokens if the
    model returns an empty string (occasionally happens with base models).
    """
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    if not response.strip():
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=150,
            verbose=False,
        )
    return response.split("\nQ:")[0].strip()

# TruthfulQA with instruct models 
print("\n── TruthfulQA (instruct models) ──")
for model_name, model_id in MODELS_INSTRUCT.items():
    output_path = f"{OUTPUT_DIR}/{model_name}_truthfulqa.jsonl"

    if os.path.exists(output_path):
        print(f"Skipping {model_name} — already exists")
        continue

    print(f"\nLoading: {model_name} instruct ({model_id})")
    model, tokenizer = load(model_id)

    with open(output_path, "w") as f:
        for i, q in enumerate(truthfulqa_questions):
            if i % 100 == 0:
                print(f"  [{model_name}] {i}/{len(truthfulqa_questions)}")

            # Instruct-style prompt — models are trained to follow this format
            prompt   = f"Answer this question in one sentence:\n{q['question']}"
            response = generate_answer(model, tokenizer, prompt)

            record = {**q, "model": model_name, "model_type": "instruct", "model_answer": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Saved {output_path}")

# FAME with base models 
print("\n── FAME (base models) ──")
for model_name, model_id in MODELS_BASE.items():
    output_path = f"{OUTPUT_DIR}/{model_name}_answers.jsonl"

    if os.path.exists(output_path):
        print(f"Skipping {model_name} — already exists")
        continue

    print(f"\nLoading: {model_name} base ({model_id})")
    model, tokenizer = load(model_id)

    with open(output_path, "w") as f:
        for i, q in enumerate(fame_questions):
            if i % 100 == 0:
                print(f"  [{model_name}] {i}/{len(fame_questions)}")

            # Q: A: format works better for base models than instruction-style prompts
            prompt   = f"Q: {q['question']}\nA:"
            response = generate_answer(model, tokenizer, prompt)

            record = {**q, "model": model_name, "model_type": "base", "model_answer": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Saved {output_path}")

#  Summary 
print("\nAll done! Output files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    path = f"{OUTPUT_DIR}/{fname}"
    size = os.path.getsize(path) / 1024
    lines = sum(1 for _ in open(path))
    print(f"  {fname}: {size:.1f} KB — {lines} rows")
