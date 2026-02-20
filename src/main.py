from fastapi import FastAPI

app = FastAPI()

@app.get("/ask")
async def ask(prompt: str) -> str:
    answer = call_rag(prompt)
    return answer