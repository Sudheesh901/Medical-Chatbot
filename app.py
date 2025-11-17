from flask import Flask, render_template,jsonify, request
from dotenv import load_dotenv
import os

#Import rag pipeline with MLflow tracking

from pipelines.rag_pipeline import get_rag_chain

app = Flask(__name__)

load_dotenv()

print("RUNTIME OPENAI KEY PREFIX:", os.getenv("OPENAI_API_KEY")[:20])

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#set as environment variable
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

#Build the rag chain with ML flow

rag_chain=get_rag_chain()

"""
embedding=download_embeddings()
index_name="medical-chatbot"


#embed each chunk and upsert the embeddings into your Pinecone index
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={'k':3})
chatModel=ChatOpenAI(model="gpt-4o")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)
"""

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg=request.form["msg"]

    print("USER", msg)
    
    response =rag_chain.invoke({"input":msg})
    bot_answer=response["answer"]

    print("BOT:", bot_answer)
    return str(bot_answer)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)