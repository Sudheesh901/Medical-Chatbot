import subprocess
import sys
import os

def main():
    print("Running RAG evaluation pipeline...")

    result=subprocess.run(
        [sys.executable, "evaluation/evaluate_rag.py"],
        capture_output=True,
        text=True
    )

    print(" ==== OUTPUT OF EVALUATION SCRIPT ==== ")
    print(result.stdout)
    print(" ==== END OF OUTPUT ==== ")

    if result.stderr:
        print("\n Errors/Warnings during evaluation: ")
        print(result.stderr)

    if result.returncode ==0 :
        print("\n Evlaluation completed successfully.")
    
    else:
        print("\n Evaluation script failed with return code:", result.returncode)
        sys.exit(result.returncode)

if __name__=="__main__":
    main()
