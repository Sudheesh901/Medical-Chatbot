# evaluation/evaluate_rag_ci.py  (FINAL VERSION – overwrite fully)
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


def normalize_result(result):
    """
    Ragas returns different formats depending on environment.
    This function normalizes into a clean {metric: float} dict.
    """
    # Case 1: local run → EvaluationResult with .scores dict
    if hasattr(result, "scores") and isinstance(result.scores, dict):
        return {k: float(v) for k, v in result.scores.items()}

    # Case 2: CI run → list of per-sample dicts
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        agg = {}
        n = len(result)
        for item in result:
            for k, v in item.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        return {k: v / n for k, v in agg.items()}

    # Case 3: Unexpected fallback
    return {}


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

        print("DEBUG raw score:", score)

        results = normalize_result(score)

        print("=== CI Metrics ===")
        for k, v in results.items():
            print(f"{k}: {v}")

        # Quality gates
        if results.get("faithfulness", 0) < 0.20:
            print("❌ CI fail: faithfulness below threshold")
            sys.exit(1)

        if results.get("answer_relevancy", results.get("response_relevancy", 0)) < 0.30:
            print("❌ CI fail: relevance below threshold")
            sys.exit(1)

        print("✅ CI PASS")
        sys.exit(0)

    except Exception as e:
        print("❌ EXCEPTION THROWN IN CI:")
        print(str(e))
        sys.exit(1)



if __name__ == "__main__":
    main()
