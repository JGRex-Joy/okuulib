from typing import List, Iterable
from openai import OpenAI

from src.config import settings

class DenseEmbedder:
    def __init__(self):
        if settings.OPENAI_API_KEY is None:
            raise ValueError("API_KEY not found")
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.DENSE_EMBEDDING_MODEL
        self.batch_size = settings.DENSE_EMBEDDING_BATCH_SIZE

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

dense_embedder = DenseEmbedder()