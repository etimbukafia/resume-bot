from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import asyncio
import io
from config import Configs

configs = Configs()

async def retrieval_chain(vector_store, llm):
    try:
        retriever = vector_store.as_retriever()

        try:
            with io.open("prompt.txt","r",encoding="utf-8")   as f1:
                prompt=f1.read()
        except IOError as e:
            print(f"Error reading prompt file: {e}")
            return
        
        custom_rag_prompt = PromptTemplate.from_template(prompt)

    
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        query = "Which of the applicants worked as a Data Entry Clerk"
        answer = await rag_chain.ainvoke(query)
        print(answer)
    
    except Exception as e:
        print(f"Error in retrieval_chain: {e}")

async def main():
    try:
        # Initialize asynchronously
        await configs.initialize()
    
        vector_store = configs.get_vector_store()
        llm = configs.get_llm()

        # Call the synchronous function
        await retrieval_chain(vector_store, llm)
    
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())


