import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from ragas import evaluate
from datasets import Dataset
from evaluation.log_rag_mlflow import log_rag_to_mlflow
from evaluation.log_rag_artifacts import log_json_artifacts
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall
)

from langchain_openai import ChatOpenAI
from src.helper import download_embeddings
from pipelines.rag_pipeline import get_rag_chain


# -----------------------------
# Test Questions + REFERENCES
# -----------------------------
def load_test_set():
    return [
        {
            "question": "What is gigantism?",
            "reference": (
                "Gigantism is a rare condition that causes excessive growth and "
                "height due to excess growth hormone before growth plates close."
            )
        },
        {
            "question": "What are symptoms of hypothyroidism?",
            "reference": (
                "Symptoms include fatigue, cold sensitivity, constipation, dry skin, "
                "weight gain, puffy face, and thinning hair."
            )
        },
        {
            "question": "How to treat diabetes type 2?",
            "reference": (
                "Treatment includes diet, exercise, oral medications, and sometimes insulin."
            )
        }
    ]


# -----------------------------
# Normalize ANY Ragas Output
# -----------------------------
def normalize_result(result):
    """
    Normalize Ragas result into:  {metric: float}
    Handles:
    - Local EvaluationResult (result.scores dict)
    - GHA/CI list of per-sample dicts
    - Ragas weird fallback formats
    """

    # Case 1 — Standard EvaluationResult (local run)
    if hasattr(result, "scores") and isinstance(result.scores, dict):
        return {k: float(v) for k, v in result.scores.items()}

    # Case 2 — GitHub Actions returns a list of dictionaries
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        agg = {}
        n = len(result)
        for row in result:
            for k, v in row.items():
                try:
                    agg[k] = agg.get(k, 0.0) + float(v)
                except:
                    pass
        return {k: v / n for k, v in agg.items()}

    # Case 3 — Unexpected, prevent crash
    return {
        "answer_relevancy": 0.0,
        "faithfulness": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }


# -----------------------------
# Main Evaluation Pipeline
# -----------------------------
def main():

    # MLflow settings (local only)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("medical_rag_experiment")

    with mlflow.start_run():

        ragas_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        ragas_embeddings = download_embeddings()

        rag_chain = get_rag_chain()
        dataset = load_test_set()

        data_for_ragas = []
        rag_outputs = []

        # --------------------------
        # Run RAG on each question
        # --------------------------
        for item in dataset:
            q = item["question"]
            ref = item["reference"]

            print(f"\nEvaluating: {q}")
            response = rag_chain.invoke({"input": q})

            answer = response["answer"]
            contexts = [doc.page_content for doc in response["context"]]

            rag_outputs.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "reference": ref
            })

            data_for_ragas.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "reference": ref
            })

        # Save artifacts
        log_json_artifacts(dataset, "evaluation_dataset.json")
        log_json_artifacts(rag_outputs, "rag_outputs.json")

        # --------------------------
        # Ragas Evaluation
        # --------------------------
        ragas_dataset = Dataset.from_list(data_for_ragas)

        print("\nStarting Ragas evaluation...")
        score = evaluate(
            ragas_dataset,
            metrics=[
                ResponseRelevancy(),
                Faithfulness(),
                ContextPrecision(),
                ContextRecall(),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        # Normalize results for MLflow
        score_dict = normalize_result(score)

        # Log to MLflow
        log_rag_to_mlflow(score_dict)

        print("\n=== FINAL RAG Evaluation Metrics ===")
        print(score_dict)


if __name__ == "__main__":
    main()
