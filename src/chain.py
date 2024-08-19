from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import io

async def retrieval_chain(vector_store, llm, query):
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

        answer = await rag_chain.ainvoke(query)
        return answer
    
    except Exception as e:
        print(f"Error in retrieval_chain: {e}")

