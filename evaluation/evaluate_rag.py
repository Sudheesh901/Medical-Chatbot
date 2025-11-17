import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow
from ragas import evaluate
from datasets import Dataset # Corrected import from previous issue
from evaluation.log_rag_mlflow import log_rag_to_mlflow
from evaluation.log_rag_artifacts import log_json_artifacts
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall
)


# ➡️ Import your required components for Ragas evaluation
from langchain_openai import ChatOpenAI
from src.helper import download_embeddings

from pipelines.rag_pipeline import get_rag_chain


def load_test_set():
    return [
        {"question": "What is gigantism?",
         "ground_truth": "Gigantism is a rare condition that causes excessive growth and height, significantly above average. It's typically caused by overproduction of growth hormone (GH) by the pituitary gland before the growth plates close."},
        {"question": "What are symptoms of hypothyroidism?",
         "ground_truth": "Symptoms include fatigue, increased sensitivity to cold, constipation, dry skin, weight gain, a puffy face, and thinning hair."},
        {"question": "How to treat diabetes type 2?",
         "ground_truth": "Treatment includes lifestyle changes such as diet and exercise, oral medications, and sometimes insulin therapy."},
    ]


def main():

    # Load MLflow config
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("medical_rag_experiment")

    with mlflow.start_run():
        # Instantiate the LLM and Embedding Model for Ragas
        # The LLM must be a powerful model for accurate critique (GPT-4o)
        ragas_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
        # Using same embedding model as indexing
        ragas_embeddings = download_embeddings()

        # 1. Run RAG pipeline to generate answers/contexts
        rag_chain = get_rag_chain()
        dataset = load_test_set()
        data_for_ragas = [] 
        rag_outputs = []   # store model outputs for logging

        for item in dataset:
            q = item["question"]
            gt=item["ground_truth"]
            print("\nEvaluating:", q)
            response = rag_chain.invoke({"input": q})
            answer = response["answer"]
            contexts = [doc.page_content for doc in response["context"]]
            
            rag_outputs.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": gt
            })


            data_for_ragas.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt
            })
        #log the RAG outputs as artifacts (json file)
        log_json_artifacts(dataset, "evaluation_dataset.json")
        log_json_artifacts(rag_outputs, "rag_output.json")
        

        # 2. Create the Ragas Dataset
        ragas_dataset = Dataset.from_list(data_for_ragas)
        
        # 3. Evaluate RAG using RAGAS, passing the models explicitly
        print("\nStarting Ragas Evaluation...")
        score = evaluate(
            ragas_dataset,
            metrics=[
                ResponseRelevancy(),
                Faithfulness(),
                ContextPrecision(),
                ContextRecall(),
            ],
            # ➡️ Explicitly pass your models here!
            llm=ragas_llm, 
            embeddings=ragas_embeddings
        )

        # Convert RAGAS scores list → proper dict
        score_dict = {k: v for d in score.scores for k, v in d.items()}
        

        #call the new logging function
        log_rag_to_mlflow(score_dict)
        print("\n=== RAG Evaluation Metrics ===")
        print(score_dict)

if __name__ == "__main__":
    main()