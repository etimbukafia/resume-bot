import pymupdf
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def process_text(doc):
    reader = pymupdf.open(stream=doc, filetype='pdf')
    
    text = ""
            
    # Extract text from all pages
    for page_number in range(reader.page_count):
        page = reader[page_number]
        text += page.get_text() + "\n"

    pdf = [Document(page_content=text)]

    return pdf


def summarizer(pdf, llm):
    template = """
    Summarize the provided resume into three concise sentences:
    {context}
    
    Highlight the applicant's core competencies, relevant experience, and educational background. Include specific details such as job titles, companies, degrees, and certifications. If available, briefly mention desired job roles and contact information. """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": pdf})
    return result







