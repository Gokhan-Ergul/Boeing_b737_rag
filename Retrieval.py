from multiprocessing import context
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

collection_name = "my_rag_collection"
qdrant_path = "./qdrant_db_data"

client = QdrantClient(path = qdrant_path)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}   
)

db = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embedding_model
)

def retrieve_similar_documents(query, k=3):
    results = db.similarity_search(query, k=k)
    
    formatted_context = ""
    for i, doc in enumerate(results):
        page = doc.metadata.get('page')
        content = doc.page_content
        formatted_context += f"\n--- Source {i+1} (Page {page}) ---\n{content}\n"
    return formatted_context


import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

system_prompt = """
You are a helpful and strict assistant for a technical manual. 
Your task is to answer the user's question using ONLY the provided context snippets.

Rules:
1. Analyze the provided context snippets to find the best answer.
2. If the text describes a standard procedure related to the user's topic (e.g., "CDU entries"), provide that standard procedure, even if the user's question implies a different scenario (e.g., wrong role assignment).
3. ALWAYS cite the specific page number(s) where you found the information (e.g., [Source: Page 12]).
4. If the provided context truly contains no relevant information to the topic, strictly say: "I cannot answer this based on the provided documents."
5. Do not make up information or use outside knowledge.
"""



def format_context_for_llm(user_question, k=5):
    context_text = retrieve_similar_documents(user_question, k=k)

    messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {user_question}")
    ]   
    response = llm.invoke(messages)
    return response.content



if __name__ == "__main__":
    pass