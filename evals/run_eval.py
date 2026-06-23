import os
import csv
import json
import time
from datetime import datetime
from collections import defaultdict
from anthropic import Anthropic
from dotenv import load_dotenv

from src.agent import RAGAgent
from src.memory import SemanticMemory

def run_citation_judge(client: Anthropic, query: str, answer: str, citations: list, expected: str) -> int:
    """
    LLM-as-a-judge to evaluate citation quality on a 0-2 scale.
    """
    judge_prompt = (
        "You are an expert AI evaluator assessing RAG citation quality.\n"
        "Assess whether the answer includes appropriate citations and doesn't hallucinate.\n\n"
        f"Query: {query}\n"
        f"Answer: {answer}\n"
        f"Citations list: {citations}\n"
        f"Expected keywords: {expected}\n\n"
        "SCORING RUBRIC:\n"
        "2: Factual claims are correctly cited and backed by documents/tools. No hallucinations.\n"
        "1: Answer is mostly correct, but citations are partially missing, incomplete, or slightly mismatched.\n"
        "0: Answer misses citations entirely, has incorrect citations, or has major factual errors.\n\n"
        "Output ONLY a single integer digit (0, 1, or 2) with no other text."
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=5,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )
        score_str = response.content[0].text.strip()
        # Extract first digit found
        for char in score_str:
            if char in ["0", "1", "2"]:
                return int(char)
        return 1
    except Exception as e:
        print(f"Judge invocation failed: {e}")
        return 1

def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    agent = RAGAgent()
    memory = SemanticMemory()
    
    # Reset memories before evaluation
    memory.reset_memories()

    print("Loading Golden Set...")
    with open("evals/golden_set.json", "r") as f:
        golden_set = json.load(f)

    results = []
    
    tier_totals = defaultdict(int)
    tier_correct = defaultdict(int)
    schema_valid_count = 0
    total_cost = 0.0

    print(f"Running evaluation on {len(golden_set)} items...")
    
    for item in golden_set:
        item_id = item["id"]
        tier = item["tier"]
        question = item["question"]
        expected = item["expected"]
        session_id = item["session_id"]
        history = item.get("multi_turn_history", [])

        # 1. Populate memory for multi-turn history if present
        if history:
            print(f"\n[Item {item_id}] Pre-populating memory for session: {session_id}")
            for turn in history:
                memory.save_exchange(session_id, turn["user"], turn["assistant"])

        # 2. Run agent
        print(f"\n[Item {item_id}][Tier {tier}] Query: '{question}'")
        start_time = time.time()
        
        # Capture raw response
        try:
            res_dict = agent.run_agent(question, session_id)
            latency = time.time() - start_time
            schema_valid = True
            schema_valid_count += 1
        except Exception as e:
            print(f"Error running agent: {e}")
            res_dict = {
                "answer": "Error during execution",
                "citations": [],
                "confidence": 0.0,
                "follow_up_questions": []
            }
            latency = time.time() - start_time
            schema_valid = False

        answer = res_dict.get("answer", "")
        citations = res_dict.get("citations", [])
        confidence = res_dict.get("confidence", 0.0)

        # 3. Score correctness
        correct = expected.lower() in answer.lower()
        if correct:
            tier_correct[tier] += 1
        tier_totals[tier] += 1

        # 4. Score citation quality via LLM-as-a-judge
        citation_score = run_citation_judge(client, question, answer, citations, expected)
        
        item_cost = agent.total_cost
        total_cost += item_cost
        
        print(f"Result: Correct={correct} | SchemaValid={schema_valid} | CitationScore={citation_score} | Cost=${item_cost:.4f} | Latency={latency:.2f}s")
        
        results.append({
            "id": item_id,
            "tier": tier,
            "question": question,
            "answer": answer,
            "expected": expected,
            "correct": correct,
            "schema_valid": schema_valid,
            "citation_score": citation_score,
            "cost": item_cost,
            "latency": latency
        })

    # Summary Statistics
    total_items = len(golden_set)
    overall_accuracy = sum(1 for r in results if r["correct"]) / total_items
    overall_schema_validity = schema_valid_count / total_items
    avg_citation_score = sum(r["citation_score"] for r in results) / total_items
    avg_latency = sum(r["latency"] for r in results) / total_items

    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}% ({sum(1 for r in results if r['correct'])}/{total_items})")
    print(f"Schema Validity:  {overall_schema_validity*100:.1f}% ({schema_valid_count}/{total_items})")
    print(f"Avg Citation Score (0-2): {avg_citation_score:.2f}")
    print(f"Total Evals Cost: ${total_cost:.4f}")
    print(f"Avg Query Latency: {avg_latency:.2f}s")
    
    print("\n--- Tier Scores ---")
    for t in sorted(tier_totals.keys()):
        acc = tier_correct[t] / tier_totals[t]
        print(f"Tier {t}: {tier_correct[t]}/{tier_totals[t]} ({acc*100:.1f}%)")

    # Enforce pass/fail gates
    # Minimum bar: Tier 1 >= 4/5 · Tier 2 >= 3/5 · Tier 3 >= 2/5 · Schema validity >= 85%
    tier1_pass = (tier_correct[1] >= 4)
    tier2_pass = (tier_correct[2] >= 3)
    tier3_pass = (tier_correct[3] >= 2)
    schema_pass = (overall_schema_validity >= 0.85)

    print("\n--- Pass/Fail Gates ---")
    print(f"Tier 1 Gate (>= 4/5): {'PASS' if tier1_pass else 'FAIL'} (Score: {tier_correct[1]}/5)")
    print(f"Tier 2 Gate (>= 3/5): {'PASS' if tier2_pass else 'FAIL'} (Score: {tier_correct[2]}/5)")
    print(f"Tier 3 Gate (>= 2/5): {'PASS' if tier3_pass else 'FAIL'} (Score: {tier_correct[3]}/5)")
    print(f"Schema Gate (>= 85%): {'PASS' if schema_pass else 'FAIL'} (Score: {overall_schema_validity*100:.1f}%)")

    # Write to CSV
    csv_file = "evals/eval_results.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "overall_accuracy", "schema_validity", 
                "avg_citation_score", "tier1_score", "tier2_score", 
                "tier3_score", "total_cost"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(overall_accuracy, 4),
            round(overall_schema_validity, 4),
            round(avg_citation_score, 4),
            f"{tier_correct[1]}/5",
            f"{tier_correct[2]}/5",
            f"{tier_correct[3]}/5",
            round(total_cost, 4)
        ])

    # Write JSON report
    os.makedirs("evals/reports", exist_ok=True)
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": overall_accuracy,
        "schema_validity": overall_schema_validity,
        "avg_citation_score": avg_citation_score,
        "total_cost": total_cost,
        "avg_latency": avg_latency,
        "details": results
    }
    with open("evals/reports/latest.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Throw error if gates fail to block CI pipeline if regression occurs
    if not (tier1_pass and tier2_pass and tier3_pass and schema_pass):
        raise RuntimeError("CI Gate failed: Regressions detected under required threshold.")
    print("\nEvaluation Passed Successfully!")

if __name__ == "__main__":
    main()