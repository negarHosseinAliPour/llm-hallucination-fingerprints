import json
import os
import random
from collections import defaultdict
from mlx_lm import load, generate
from datasets import load_from_disk

# ── Config 
MODELS = {
    "llama": "mlx-community/Meta-Llama-3.1-8B-4bit",
    "mistral": "mlx-community/Mistral-7B-v0.3-4bit",
    "qwen": "mlx-community/Qwen2.5-7B-4bit",
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

# ── Topics to exclude (5 smallest) 
EXCLUDE_TOPICS = {
    "Mandela Effect",        # 4
    "Misconceptions: Topical", # 4
    "Statistics",            # 5
    "Subjective",            # 9
    "Confusion: Other",      # 8
}

QUESTIONS_PER_TOPIC = 20

# ── Load datasets 
print("Loading datasets...")
truthfulqa = load_from_disk("data/raw/truthfulqa")["validation"]
fame = load_from_disk("data/raw/fame")

# ── Sample TruthfulQA: 33 topics x 20 questions ──────────────
topic_buckets = defaultdict(list)
for row in truthfulqa:
    if row["category"] not in EXCLUDE_TOPICS:
        topic_buckets[row["category"]].append(row)

questions = []
for topic, rows in sorted(topic_buckets.items()):
    sampled = random.sample(rows, min(QUESTIONS_PER_TOPIC, len(rows)))
    for row in sampled:
        questions.append({
            "dataset": "truthfulqa",
            "topic": row["category"],
            "language": "en",
            "question": row["question"],
            "reference_answer": row["best_answer"],
        })

print(f"TruthfulQA topics: {len(topic_buckets)}")
print(f"TruthfulQA questions: {len(questions)}")

# ── Sample FAME: 20 topics x 5 languages x 20 questions 
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
            "language": row["language"],
            "question": row["question"],
            "reference_answer": row["answer"],
        })

print(f"FAME questions: {len(fame_questions)}")

all_questions = questions + fame_questions
print(f"Total questions: {len(all_questions)}")

# ── Generate answers 
for model_name, model_id in MODELS.items():
    output_path = f"{OUTPUT_DIR}/{model_name}_answers.jsonl"

    if os.path.exists(output_path):
        print(f"\nSkipping {model_name} — already exists")
        continue

    print(f"\nLoading model: {model_name} ({model_id})")
    model, tokenizer = load(model_id)

    with open(output_path, "w") as f:
        for i, q in enumerate(all_questions):
            if i % 100 == 0:
                print(f"  [{model_name}] {i}/{len(all_questions)}")

            prompt = f"Answer this question in one sentence:\n{q['question']}"

            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=100,
                verbose=False,
            )

            record = {**q, "model": model_name, "model_answer": response}
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved {output_path}")

print("\nAll done!")