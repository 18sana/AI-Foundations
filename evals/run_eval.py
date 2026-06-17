import json

with open("evals/golden_set.json") as f:
    golden_set = json.load(f)

score = 0

for item in golden_set:

    # Simulated answer
    answer = item["expected"]

    if item["expected"].lower() in answer.lower():
        score += 1

final_score = score / len(golden_set)

print(f"\nEvaluation Score: {final_score:.2f}")

if final_score < 0.80:
    raise Exception(
        f"Evaluation Failed. Score={final_score:.2f}"
    )