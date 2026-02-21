from openai import OpenAI

from src.retrieval.prompts.prompt_loader import load_prompt
from src.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

class LLMService:
    def generate_answer(self, query: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(contexts)
        
        system_prompt = load_prompt("system_prompt.txt")
        rag_template = load_prompt("rag_prompt.txt")
        
        query_prompt = rag_template.format(
            context=context_text,
            question=query
        )
        
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_prompt}
            ],
        )

        return response.choices[0].message.content
    
llm_service = LLMService()