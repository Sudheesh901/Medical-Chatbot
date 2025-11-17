# evaluation/log_rag_mlflow.py (Verify this function)
import mlflow
from typing import Any, Dict

# The Ragas score is now a standard dictionary when passed here
def log_rag_to_mlflow(ragas_scores: Dict[str, Any]): 
    """
    Logs RAG evaluation results to the currently active MLflow.
    """
    try:
        # This now works because ragas_scores is guaranteed to be a dict with .to_dict()
        ragas_metrics = { k: float(v) for k,v in ragas_scores.items() } 
        mlflow.log_metrics(ragas_metrics)
        print("✅ RAG evaluation results logged to MLflow.")
        
    except Exception as e:
        # This will now catch true MLflow errors (like no active run)
        print(f"❌ An error occurred while logging to MLflow: {e}")