from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import io
from bson import ObjectId
import os
import pprint


async def get_retriever(vector_store, k):
    try:
        return vector_store.as_retriever(
            search_type = "similarity", search_kwargs={"k": k}
        )
    except Exception as e:
        print(f"Error in retrieval_chain: {e}")
        return None

async def doc_Retriever(vector_store, query, resumeCollection):
    retriever = await get_retriever(vector_store, 5)
    if not retriever:
        print("Failed to initialize retriver")
        return None
    
    try:
        documents = await retriever.ainvoke(query)
    #pprint.pprint(documents)

    except Exception as e:
        print(f"Error invoking retriever: {e}")
        return None

    processed_ids = set()  # To track processed original_id's
    #print(processed_ids)

    # Directory to save the PDF resumes
    output_dir = "src/resume"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the retrieved documents to fetch the original resume
    for doc in documents:
        # Print the document and its metadata
        original_id = doc.metadata.get("metadata", {}).get("original_id")
        print(f"Original ID: {original_id}")

        """
        Extract the original_id from the document's metadata
        If "metadata" exists as a key, it will return the value associated with it (which is expected to be another dictionary).
        If "metadata" does not exist as a key, the second argument {} (an empty dictionary) is returned as a default.
        """
        
        if original_id and original_id not in processed_ids:
            # Query the resumeCollection to retrieve the full original resume
            original_resume = await resumeCollection.find_one({"_id": ObjectId(original_id)})

            
            if original_resume:
                #original_resumes.append(original_resume)
                
                # Get the binary PDF data
                pdf_data = original_resume.get("resume")
                
                if pdf_data:
                    # Save the binary PDF data to a file
                    pdf_filename = f"{original_resume['firstName']}_{original_resume['lastName']}_resume.pdf"
                    pdf_filepath = os.path.join(output_dir, pdf_filename)
                    
                    try:
                        with open(pdf_filepath, "wb") as f:
                            f.write(pdf_data)
                        print (f"Saved PDF resume to {pdf_filepath}")
                    except IOError as e:
                        print (f"Failed to save PDF resume: {e}")

                    # Add the original_id to the set of processed IDs to avoid duplicates
                    processed_ids.add(original_id)
                else: 
                    print(f"No PDF data found in resume for original_id: {original_id}")
            else:
                print(f"No original resume found for original_id: {original_id}")
        else:
            print(f"Skipping duplicate or missing original_id")


async def retrieval_chain(vector_store, llm, query):
    try:
        retriever = await get_retriever(vector_store)
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



async def pdf_retriever(vector_store, query, resumeCollection):
        result = await doc_Retriever(vector_store, query, resumeCollection)
        if result is None:
            print("Failed to retrieve documents.")
            return None
        return result
    
    