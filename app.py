import os 
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

Pinecone_api_key = os.getenv("PINECONE_API_KEY")
Pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=Pinecone_api_key)
index = pc.Index(Pinecone_index_name)
model = SentenceTransformer(        
'BAAI/bge-small-en-v1.5',
device='cpu'
)     

app = FastAPI()

class searchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def read_root():
    return {"message": "Api is working"}

@app.post("/search")
def search(request: searchRequest):
    query_embedding = model.encode(request.query).tolist()

    results = index.query(
        vector= query_embedding,
        top_k= request.top_k,
        include_metadata=True
    )

    matches = []
    for match in results['matches']:
        matches.append({
            "id": match["id"],
            "score": match["score"],
            "title": match["metadata"].get("title"),
            "metadata": match["metadata"].get("class")
        })

    return {
        "query": request.query,
        "matches": matches
    }