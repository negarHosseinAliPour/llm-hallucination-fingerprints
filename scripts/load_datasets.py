from datasets import load_dataset

# TruthfulQA - 817 questions across 38 categories
print("Loading TruthfulQA...")
truthfulqa = load_dataset("truthfulqa/truthful_qa", "generation")
truthfulqa.save_to_disk("data/raw/truthfulqa")
print("✓ TruthfulQA saved")

# FAME - 21k QA pairs about fictional actors across 5 languages
print("Loading FAME...")
fame = load_dataset("ClaudioSavelli/FAME", split="retain")
fame.save_to_disk("data/raw/fame")
print("✓ FAME saved")

print("\nDone! Check data/raw/ for both datasets.")