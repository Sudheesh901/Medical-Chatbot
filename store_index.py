from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#set as environment variable
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY


extracted_data=load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk=text_split(minimal_docs)


embedding=download_embeddings()

#create a database
pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, #Dimensions of embeddings
        metric="cosine", #Consine Similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index=pc.Index(index_name)

#Store all the vector embeddings in database

docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)


