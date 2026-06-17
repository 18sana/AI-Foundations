import json
import csv
import os
from collections import defaultdict
from datetime import datetime

# -------------------------
# Load Golden Dataset
# -------------------------

with open("evals/golden_set.json", "r") as f:
    golden_set = json.load(f)

# -------------------------
# Simulated Evaluation
# Replace this later with:
# RAG -> Claude -> Judge
# -------------------------

passed = 0

category_total = defaultdict(int)
category_passed = defaultdict(int)

details = []

for item in golden_set:

    question = item["question"]
    expected = item["expected"]
    category = item["category"]

    # Simulated answer
    answer = expected

    success = expected.lower() in answer.lower()

    if success:
        passed += 1
        category_passed[category] += 1

    category_total[category] += 1

    details.append({
        "id": item["id"],
        "question": question,
        "category": category,
        "status": "PASS" if success else "FAIL"
    })

# -------------------------
# Scores
# -------------------------

overall_score = passed / len(golden_set)

category_scores = {}

for category in category_total:

    category_scores[category] = round(
        category_passed[category] / category_total[category],
        2
    )

# -------------------------
# Console Output
# -------------------------

print("\n===== Evaluation Report =====")
print(f"Overall Score: {overall_score:.2f}")

for category, score in category_scores.items():
    print(f"{category}: {score:.2f}")

# -------------------------
# CSV Tracking
# -------------------------

csv_file = "evals/eval_results.csv"

file_exists = os.path.exists(csv_file)

with open(csv_file, "a", newline="") as f:

    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "overall_score"
        ])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overall_score
    ])

# -------------------------
# JSON Report
# -------------------------

os.makedirs(
    "evals/reports",
    exist_ok=True
)

report = {
    "timestamp": datetime.now().isoformat(),
    "overall_score": overall_score,
    "passed": passed,
    "total": len(golden_set),
    "category_scores": category_scores,
    "details": details
}

with open(
    "evals/reports/latest.json",
    "w"
) as f:
    json.dump(
        report,
        f,
        indent=2
    )

# -------------------------
# CI Gate
# -------------------------

MIN_SCORE = 0.80

if overall_score < MIN_SCORE:

    raise Exception(
        f"Evaluation Failed. Score={overall_score:.2f}"
    )

print("\nEvaluation Passed")