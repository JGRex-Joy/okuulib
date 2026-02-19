from fastapi import FastAPI
from openai import OpenAI

from src.shared.embedder import embedder
from src.shared.vector_store import vector_store

from dotenv import load_dotenv
load_dotenv()
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/ask")
async def ask_rag(query: str) -> str:
    
    embed_query = embedder.embed(query)
    found = vector_store.search(embed_query)
    
    contexts = [
        point.payload["text"]
        for point in found.points
    ]
    
    context_text = "\n\n".join(contexts)
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system", "content": f"Сен адабий жардамчысын. Ушул контекстке таянып, жооп бер: \n\n{context_text}"},
            {"role":"user", "content": query}
        ],
    )
    
    return response.choices[0].message.content