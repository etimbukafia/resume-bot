import os
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks

load_dotenv()

# Cache the LLM instance
_fireworks_llm_instance = None

async def get_fireworks_llm():
    global _fireworks_llm_instance
    if _fireworks_llm_instance is None:
        fireworks_api_key = os.environ['FIREWORKS_API_KEY']
        MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"
        
        _fireworks_llm_instance = ChatFireworks(
            model=MODEL_ID,
            temperature = 0.7,
            max_tokens = 2048,
            model_kwargs={
                "top_p": 1,
                },
                cache=None,
        )
    
    return _fireworks_llm_instance
