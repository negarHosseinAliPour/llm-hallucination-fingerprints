from datasets import load_dataset
import os

#  TruthfulQA 
print("Loading TruthfulQA...")
truthfulqa = load_dataset("truthful_qa", "generation")
truthfulqa.save_to_disk("data/raw/truthfulqa")
print("✓ TruthfulQA saved")

# FAME 
print("Loading FAME...")
fame = load_dataset("anonymousub/FAME")
fame.save_to_disk("data/raw/fame")
print("✓ FAME saved")

print("\nDone! Check data/raw/ for both datasets.")