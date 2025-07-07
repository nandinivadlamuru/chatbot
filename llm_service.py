from fastapi import FastAPI
from pydantic import BaseModel
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.schema import Document
from langchain_community.llms import Ollama

app = FastAPI()

# Define input model
class QuestionInput(BaseModel):
    question: str

# Initialize Haystack in-memory document store with BM25
doc_store = InMemoryDocumentStore(use_bm25=True)

# Sample documents to populate store
docs = [
    Document(content="AI is the simulation of human intelligence in machines."),
    Document(content="Mistral is a light" \
    "" \
    "weight language model for local inference."),
]

# Write documents to store
doc_store.write_documents(docs)

# Initialize BM25 retriever
retriever = BM25Retriever(document_store=doc_store)

# Initialize Ollama LLM
llm = Ollama(model="mistral", temperature=0.2)

# Preload model on startup
@app.on_event("startup")
async def preload_model():
    try:
        llm.invoke("Hello")
        print("[INFO] Ollama model preloaded.")
    except Exception as e:
        print(f"[ERROR] Failed to preload model: {e}")

# Endpoint to query Haystack + Ollama
@app.post("/query-haystack")
async def query_haystack(data: QuestionInput):
    try:
        query = data.question
        retrieved = retriever.retrieve(query, top_k=2)
        context = " ".join([d.content for d in retrieved])
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        answer = llm.invoke(prompt)
        return {
            "context": [d.content for d in retrieved],
            "question": query,
            "answer": answer
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to process the question."
        }
