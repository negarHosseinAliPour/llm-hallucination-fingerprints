"""
Output : 3×3×33 miss-rate tensor + Mirroring Coefficient

Expected input format (one JSON per line):
{
    "question":         "...",
    "reference_answer": "...",
    "model_answer":     "...",
    "model":            "llama" | "mistral" | "qwen",
    "topic":            "...",
    "language":         "en" | "it" | ...,
    "label":            "HALLUCINATED" | "CORRECT"
}

Usage
-----
    python evaluate.py \
        --input  phase1_output.jsonl \
        --output results/ \
        [--model_host mlx|ollama]
"""

import argparse
import json
import os
import time
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

#  Configuration 

GENERATORS = ["llama", "mistral", "qwen"]
JUDGES     = ["llama", "mistral", "qwen"]

MLX_MODEL_IDS = {
    "llama":   "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen":    "mlx-community/Qwen2.5-7B-Instruct-4bit",
}

OLLAMA_MODEL_IDS = {
    "llama":   "llama3.1:8b",
    "mistral": "mistral:7b",
    "qwen":    "qwen2.5:7b",
}


JUDGE_PROMPT = """You are a factual accuracy evaluator.

Question: {question}
Reference answer: {reference_answer}
Model answer: {model_answer}

Focus only on whether the key fact in the model answer matches the reference answer.
Ignore style, length, or phrasing differences.

Reply with exactly one word: CORRECT or HALLUCINATED"""

#  Argument parsing 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True,  help="Path to phase1_output.jsonl")
    p.add_argument("--output",      default="results/", help="Folder for output files")
    p.add_argument("--model_host",  default="mlx", choices=["mlx", "ollama"])
    p.add_argument("--sleep",       default=0.5,   type=float)
    p.add_argument("--dry_run",     action="store_true")
    return p.parse_args()

#  Data loading 

def load_hallucinated_samples(input_path: Path) -> list[dict]:
    """
    Load Phase 1 output and keep only HALLUCINATED samples.
    These are the ~264 samples Phase 2 operates on.
    """
    all_records = []
    hallucinated = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            all_records.append(record)
            if record.get("label") == "HALLUCINATED":
                hallucinated.append(record)

    print(f"Total Phase 1 records  : {len(all_records)}")
    print(f"Hallucinated samples   : {len(hallucinated)}  (~264 expected)")
    print(f"Correct samples        : {len(all_records) - len(hallucinated)}")

    if not hallucinated:
        raise ValueError(
            "No HALLUCINATED samples found. "
            "Check that your teammate's output uses label='HALLUCINATED'."
        )

    return hallucinated

#  Model calling 

def call_mlx(model_name: str, prompt: str, max_tokens: int = 8) -> str:
    cmd = [
        "python", "-m", "mlx_lm.generate",
        "--model", MLX_MODEL_IDS[model_name],
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else "UNKNOWN"


def call_ollama(model_name: str, prompt: str, max_tokens: int = 8) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL_IDS[model_name],
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("response", "UNKNOWN").strip()


def get_verdict(judge: str, prompt: str, args) -> str:
    """Returns CORRECT | HALLUCINATED | UNKNOWN"""
    if args.dry_run:
        return "HALLUCINATED"  #  for testing

    try:
        raw = call_mlx(judge, prompt) if args.model_host == "mlx" else call_ollama(judge, prompt)
    except Exception as e:
        print(f"  ⚠️  {judge} error: {e}")
        return "UNKNOWN"

    upper = raw.upper()
    if "HALLUCINATED" in upper:
        return "HALLUCINATED"
    elif "CORRECT" in upper:
        return "CORRECT"
    else:
        print(f"  ⚠️  Unexpected reply from {judge}: {repr(raw)}")
        return "UNKNOWN"

#  Position-swap double run 


def get_verdict_with_swap(judge: str, record: dict, args) -> str:
    """
    Run judge twice with reference/model answer swapped.
    Only return a verdict if both runs agree → reduces position bias.
    If they disagree → INCONSISTENT.
    """
    prompt_normal = JUDGE_PROMPT.format(
        question         = record["question"],
        reference_answer = record["reference_answer"],
        model_answer     = record["model_answer"],
    )
    prompt_swapped = JUDGE_PROMPT.format(
        question         = record["question"],
        reference_answer = record["model_answer"],    # swapped
        model_answer     = record["reference_answer"], # swapped
    )

    verdict_1 = get_verdict(judge, prompt_normal, args)
    time.sleep(args.sleep)
    verdict_2_raw = get_verdict(judge, prompt_swapped, args)

    # In swapped prompt, CORRECT means the model_answer IS the reference
    # so we need to flip the swapped verdict
    if verdict_2_raw == "CORRECT":
        verdict_2 = "HALLUCINATED"
    elif verdict_2_raw == "HALLUCINATED":
        verdict_2 = "CORRECT"
    else:
        verdict_2 = "UNKNOWN"

    if verdict_1 == verdict_2 and verdict_1 != "UNKNOWN":
        return verdict_1
    else:
        return "INCONSISTENT"

#  Main judging loop 

def run_judges(hallucinated: list[dict], args) -> list[dict]:
    """
    For each hallucinated sample, get a verdict from each of the 3 judges.
    Uses position-swap double run for reliability.
    """
    results = []
    total = len(hallucinated) * len(JUDGES)
    done  = 0

    for rec in hallucinated:
        row = {
            "question":         rec.get("question", ""),
            "reference_answer": rec.get("reference_answer", ""),
            "model_answer":     rec.get("model_answer", ""),
            "generator":        rec.get("model", "unknown"),
            "topic":            rec.get("topic", "unknown"),
            "language":         rec.get("language", "unknown"),
            "ground_truth":     "HALLUCINATED",  # all inputs are hallucinated
        }

        for judge in JUDGES:
            done += 1
            print(f"[{done}/{total}] generator={row['generator']} judge={judge} topic={row['topic'][:20]}")

            verdict = get_verdict_with_swap(judge, rec, args)
            row[f"judge_{judge}"] = verdict
            time.sleep(args.sleep)

        results.append(row)

    return results

#  Miss-rate tensor 

def compute_miss_rate_tensor(results: list[dict]) -> tuple[dict, list[str]]:
    """
    Miss-rate = P(judge says CORRECT | answer is HALLUCINATED)
    = fraction of hallucinated answers the judge missed.

    Only CORRECT and HALLUCINATED verdicts count.
    INCONSISTENT verdicts are excluded from miss-rate calculation
    but reported separately.

    Returns tensor[generator][judge][topic] = miss_rate
    """
    topics = sorted({r["topic"] for r in results})

    # Count structure: {generator: {judge: {topic: {missed, total}}}}
    counts = {
        gen: {
            jdg: {
                topic: {"missed": 0, "total": 0, "inconsistent": 0}
                for topic in topics
            }
            for jdg in JUDGES
        }
        for gen in GENERATORS
    }

    for r in results:
        gen   = r["generator"]
        topic = r["topic"]
        if gen not in counts:
            continue

        for jdg in JUDGES:
            verdict = r[f"judge_{jdg}"]

            if verdict == "INCONSISTENT":
                counts[gen][jdg][topic]["inconsistent"] += 1

            elif verdict in ("CORRECT", "HALLUCINATED"):
                counts[gen][jdg][topic]["total"] += 1
                if verdict == "CORRECT":  # judge missed the hallucination
                    counts[gen][jdg][topic]["missed"] += 1

    # Convert to miss-rates
    tensor = {}
    for gen in GENERATORS:
        tensor[gen] = {}
        for jdg in JUDGES:
            tensor[gen][jdg] = {}
            for topic in topics:
                c = counts[gen][jdg][topic]
                tensor[gen][jdg][topic] = {
                    "miss_rate":    round(c["missed"] / c["total"], 4) if c["total"] > 0 else None,
                    "missed":       c["missed"],
                    "total":        c["total"],
                    "inconsistent": c["inconsistent"],
                }

    return tensor, topics

#  Mirroring Coefficient 

def compute_mirroring_coefficient(tensor: dict, topics: list[str]) -> dict:
    """
    MC(m, t) = miss_rate(judge=m, generator=m, topic=t)
             − mean(miss_rate(judge=m, generator≠m, topic=t))

    Positive MC → judge misses more hallucinations from same-family generator
                → self-preference bias confirmed for topic t
    """
    mc = {}
    for m in GENERATORS:
        mc[m] = {}
        for topic in topics:
            self_rate = tensor[m][m][topic]["miss_rate"]

            other_rates = [
                tensor[gen][m][topic]["miss_rate"]
                for gen in GENERATORS
                if gen != m and tensor[gen][m][topic]["miss_rate"] is not None
            ]

            if self_rate is None or not other_rates:
                mc[m][topic] = None
            else:
                mc[m][topic] = round(self_rate - np.mean(other_rates), 4)

    return mc

#  Save results 

def save_results(results: list[dict], tensor: dict, mc: dict,
                 topics: list[str], output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw judge verdicts
    raw_path = output_dir / "judge_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Raw verdicts      → {raw_path}")

    # 2. Full tensor as JSON
    tensor_path = output_dir / "miss_rate_tensor.json"
    with open(tensor_path, "w", encoding="utf-8") as f:
        json.dump({
            "tensor": tensor,
            "mirroring_coefficient": mc,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Tensor JSON       → {tensor_path}")

    # 3. CSV pivot: judge × generator (averaged over topics)
    rows = []
    for gen in GENERATORS:
        for jdg in JUDGES:
            vals = [
                tensor[gen][jdg][t]["miss_rate"]
                for t in topics
                if tensor[gen][jdg][t]["miss_rate"] is not None
            ]
            rows.append({
                "generator": gen,
                "judge":     jdg,
                "avg_miss_rate": round(np.mean(vals), 4) if vals else None,
            })

    df = pd.DataFrame(rows).pivot(
        index="judge", columns="generator", values="avg_miss_rate"
    )
    csv_path = output_dir / "miss_rate_tensor.csv"
    df.to_csv(csv_path)
    print(f" Tensor CSV        → {csv_path}")

    print("\n── Miss-rate matrix (judge rows × generator cols) ──")
    print(df.to_string())

    # 4. Mirroring Coefficient summary
    mc_rows = []
    for m in GENERATORS:
        vals = [v for v in mc[m].values() if v is not None]
        mc_rows.append({
            "model":   m,
            "mean_MC": round(np.mean(vals), 4) if vals else None,
            "std_MC":  round(np.std(vals),  4) if vals else None,
            "positive_topics": sum(1 for v in vals if v > 0),
            "negative_topics": sum(1 for v in vals if v < 0),
        })

    mc_df = pd.DataFrame(mc_rows)
    mc_path = output_dir / "mirroring_coefficient.csv"
    mc_df.to_csv(mc_path, index=False)
    print(f"\n  Mirroring Coeff   → {mc_path}")

    print("\n── Mirroring Coefficient (positive = self-preference bias) ──")
    print(mc_df.to_string(index=False))

    # 5. Inconsistency report (position bias indicator)
    inc_rows = []
    for gen in GENERATORS:
        for jdg in JUDGES:
            total_inc = sum(tensor[gen][jdg][t]["inconsistent"] for t in topics)
            total_all = sum(
                tensor[gen][jdg][t]["total"] + tensor[gen][jdg][t]["inconsistent"]
                for t in topics
            )
            inc_rows.append({
                "generator":        gen,
                "judge":            jdg,
                "inconsistent":     total_inc,
                "total":            total_all,
                "inconsistency_rate": round(total_inc / total_all, 4) if total_all > 0 else None,
            })

    inc_df = pd.DataFrame(inc_rows)
    inc_path = output_dir / "inconsistency_report.csv"
    inc_df.to_csv(inc_path, index=False)
    print(f"\n  Inconsistency     → {inc_path}")
    print("\n── Inconsistency rate (position bias indicator) ──")
    print(inc_df.to_string(index=False))

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_path  = Path(args.input)
    output_dir  = Path(args.output)

    print("=" * 60)
    print("Phase 2 — Miss-Rate Tensor")
    print(f"  Input      : {input_path}")
    print(f"  Generators : {GENERATORS}")
    print(f"  Judges     : {JUDGES}")
    print(f"  Model host : {args.model_host}")
    print(f"  Dry run    : {args.dry_run}")
    print("=" * 60)

    # Step 1 — load only hallucinated samples (~264)
    hallucinated = load_hallucinated_samples(input_path)

    # Step 2 — run 3 judges on each sample (with position-swap)
    results = run_judges(hallucinated, args)

    # Step 3 — compute 3×3×33 miss-rate tensor
    tensor, topics = compute_miss_rate_tensor(results)

    # Step 4 — compute Mirroring Coefficient
    mc = compute_mirroring_coefficient(tensor, topics)

    # Step 5 — save everything
    save_results(results, tensor, mc, topics, output_dir)

    print("\n complete.")

if __name__ == "__main__":
    main()