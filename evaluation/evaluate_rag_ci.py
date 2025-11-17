import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, Faithfulness
from langchain_openai import ChatOpenAI
from src.helper import download_embeddings
from pipelines.rag_pipeline import get_rag_chain


def load_test_set():
    return [
        {"question": "What is gigantism?"},
        {"question": "What are symptoms of hypothyroidism?"},
        {"question": "How to treat diabetes type 2?"},
    ]


# evaluation/evaluate_rag_ci.py (Final Fix)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, Faithfulness
from langchain_openai import ChatOpenAI
from src.helper import download_embeddings
from pipelines.rag_pipeline import get_rag_chain
import numpy as np  # Required for np.mean

# Remove normalize_result since we are fixing the logic in main

def load_test_set():
    return [
        {"question": "What is gigantism?"},
        {"question": "What are symptoms of hypothyroidism?"},
        {"question": "How to treat diabetes type 2?"},
    ]


def main():
    print("=== DEBUG: Starting evaluate_rag_ci ===")

    try:
        rag_chain = get_rag_chain()
        print("DEBUG: RAG chain loaded")

        ragas_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        ragas_embeddings = download_embeddings()
        print("DEBUG: LLM + embeddings loaded")

        samples = []
        for item in load_test_set():
            q = item["question"]
            print(f"DEBUG: Query -> {q}")
            response = rag_chain.invoke({"input": q})
            samples.append({
                "question": q,
                "answer": response["answer"],
                "contexts": [d.page_content for d in response["context"]],
            })

        print("DEBUG: Samples prepared")
        dataset = Dataset.from_list(samples)
        print("DEBUG: Dataset created")

        score = evaluate(
            dataset,
            metrics=[ResponseRelevancy(), Faithfulness()],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        # ------------------- REAL FINAL FIX -------------------
        # 1. Access the scores dictionary (where keys point to lists of scores)
        # We'll use the .scores attribute, which is usually correct.
        # If the direct scores dict access fails, fall back to the attribute access.
        try:
            raw_scores = score.scores
        except:
            # Fallback for when 'score' object acts like a dictionary itself
            raw_scores = score

        print("DEBUG raw scores (before aggregation):", raw_scores)

        # --- REAL WORKING FIX ---
        # Ragas returns list[dict], so aggregate manually

        results = {}
        metric_names = raw_scores[0].keys()  # e.g. "answer_relevancy", "faithfulness"

        for metric in metric_names:
            metric_values = []
            for row in raw_scores:
                val = row.get(metric, None)
                if val is not None:
                    metric_values.append(float(val))
                    
                    results[metric] = np.mean(metric_values) if metric_values else 0.0
# --- END FIX ---

        # --------------------------------------------------------

        print("=== CI Metrics ===")
        for k, v in results.items():
            print(f"{k}: {v:.4f}") # Print formatted float

        # Quality gates now compare floats to floats
        if results.get("faithfulness", 0) < 0.20:
            print("❌ CI fail: faithfulness below threshold")
            sys.exit(1)

        if results.get("answer_relevancy", results.get("response_relevancy", 0)) < 0.30:
            # Using both answer_relevancy and response_relevancy for safety
            print("❌ CI fail: relevance below threshold")
            sys.exit(1)

        print("✅ CI PASS")
        sys.exit(0)

    except Exception:
        import traceback
        print("❌ EXCEPTION THROWN IN CI:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()