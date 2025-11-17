import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall
)
from langchain_openai import ChatOpenAI
from src.helper import download_embeddings
from pipelines.rag_pipeline import get_rag_chain


def load_test_set():
    return [
        {"question": "What is gigantism?"},
        {"question": "What are symptoms of hypothyroidism?"},
        {"question": "How to treat diabetes type 2?"},
    ]


def average_scores(score_list):
    """Aggregate list of metric dicts into a single averaged dict."""
    aggregated = {}
    n = len(score_list)

    for result in score_list:
        for k, v in result.items():
            aggregated[k] = aggregated.get(k, 0) + float(v)

    # Average values
    for k in aggregated:
        aggregated[k] /= n

    return aggregated


def main():
    rag_chain = get_rag_chain()

    ragas_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    ragas_embeddings = download_embeddings()

    samples = []
    for item in load_test_set():
        q = item["question"]
        response = rag_chain.invoke({"input": q})

        samples.append({
            "question": q,
            "answer": response["answer"],
            "contexts": [d.page_content for d in response["context"]],
        })

    dataset = Dataset.from_list(samples)

    print("Running RAG evaluation...")
    score = evaluate(
        dataset,
        metrics=[
            ResponseRelevancy(),
            Faithfulness(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    # Fix: Aggregate if Ragas returns a LIST
    if isinstance(score, list):
        final_scores = average_scores(score)
    else:
        final_scores = score.scores  # normal local mode

    print("\n=== RAG Evaluation Scores ===")
    for metric, value in final_scores.items():
        print(f"{metric}: {value}")

    # ---- Quality Gate ----
    if final_scores["faithfulness"] < 0.70:
        print("❌ Failing pipeline: Faithfulness below threshold")
        sys.exit(1)

    print("✅ RAG evaluation passed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
