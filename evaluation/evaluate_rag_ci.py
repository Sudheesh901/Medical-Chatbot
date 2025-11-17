import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness
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


def main():
    rag_chain = get_rag_chain()

    # RAGAS models
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

    # ONLY two metrics (NO reference required)
    score = evaluate(
        dataset,
        metrics=[
            ResponseRelevancy(),
            Faithfulness(),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    # Print metrics
    print("=== CI Metrics ===")
    print(score.scores)

    # Quality Gate
    if score.scores["faithfulness"] < 0.50:
        print("❌ CI FAILED: Faithfulness too low")
        sys.exit(1)

    if score.scores["response_relevancy"] < 0.40:
        print("❌ CI FAILED: Relevancy too low")
        sys.exit(1)

    print("✅ CI PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
