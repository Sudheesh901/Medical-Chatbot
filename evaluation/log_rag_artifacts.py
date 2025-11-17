import json
import mlflow
import os


def log_json_artifacts(data, filename: str):
    """
    Saves a python list/dictionary as a JSON file and logs it as an MLflow artifacts
    """
    os.makedirs("tmp_artifacts", exist_ok=True)
    filepath=os.path.join("tmp_artifacts", filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    mlflow.log_artifact(filepath)

    print(f"logged artifact to MLflow: {filename}")