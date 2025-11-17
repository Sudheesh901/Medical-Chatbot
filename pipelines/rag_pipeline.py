import os
import yaml
import mlflow
from mlflow import log_metric,log_param
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from src.prompt import system_prompt
from src.helper import download_embeddings

from dotenv import load_dotenv

load_dotenv()

def load_mlflow_config():
    print(">>> Looking for mlflow.yaml at:", os.path.abspath("configs/mlflow.yaml"))
    with open("configs/mlflow.yaml") as f:
        config = yaml.safe_load(f)
        print(">>> Loaded config:", config)
        return config

def get_rag_chain():
    """
    config=load_mlflow_config()

    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    """
    embeddings=download_embeddings()

    docsearch=PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    retriever=docsearch.as_retriever(search_kwargs={'k':3})
    llm=ChatOpenAI(model="gpt-4o")

    prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
            
       

    qa_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(retriever,qa_chain)
    """
    mlflow.log_param("model", "gpt-4o")
    mlflow.log_param("retriever_k", 3)
    """
    return rag_chain
    