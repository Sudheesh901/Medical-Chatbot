import langchain_community
from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

#Extract text from pdf

def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PDFLoader
    )

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of Document objects
    containing only 'sourc' in metadata and the original page_content
    """

    minimal_docs: List[Document] =[]

    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source" : src}
            )
        )
    return minimal_docs

# Split the document into smaller chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk=text_splitter.split_documents(minimal_docs)

    return text_chunk

#Download embedding model


def download_embeddings():
    """
    Download and return the Huggingface embedding model
    """
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embedding=HuggingFaceEmbeddings(
        model_name=model_name,
        #model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    return embedding
