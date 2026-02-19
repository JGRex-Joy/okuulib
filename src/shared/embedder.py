from typing import List, Iterable
from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

class Embedder:
    def __init__(self,  
                model: str = "text-embedding-3-small", 
                batch_size: int = 100,
    ):
        api_key: str = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text.strip(),
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        for batch in self._batchify(texts):
            response = self.client.embeddings.create(
                model=self.model,
                input=[text.strip() for text in batch],
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _batchify(self, texts: List[str]) -> Iterable[List[str]]:
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

embedder = Embedder()