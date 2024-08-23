from config import Configs
import asyncio
from langchain_core.runnables import RunnableLambda


configs = Configs()

async def main():

    await configs.initialize()

    vector_Store = configs.get_vector_store()
    print(type(vector_Store))
    
    """llm = configs.get_llm()

    messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
    ]

    response = await llm.ainvoke(messages)
    print(response.content)"""

if __name__ == "__main__":
    asyncio.run(main())