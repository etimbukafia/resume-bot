import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from bson import ObjectId
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import Configs

configs = Configs()

#model = SentenceTransformer('all-MiniLM-L6-v2')
#llm = fireworks_llm
model = configs.get_embeddings()
llm = configs.get_llm()
collection = configs.get_resume_collection()
vector_index_collection = configs.get_vector_collection()

def preprocess_resume(resume):
    """
    Preprocess the resume PDF by extracting text and splitting it into chunks.
    
    Args:
        resume: The binary resume data.

    Returns:
        pdf (list): A list containing a Document object.
        resume_text_chunks (list): A list of text chunks from the resume.
    """
    # Directly pass the binary data to pymupdf.open()
    reader = pymupdf.open(stream=resume, filetype="pdf")
    text = ""
            
    # Extract text from all pages
    for page_number in range(reader.page_count):
        page = reader[page_number]
        text += page.get_text() + "\n"

    pdf = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    resume_text_chunks = text_splitter.split_text(text)

    return pdf, resume_text_chunks

def embed_function(resume_text_chunks):
    """
    Generate embeddings for the given text chunks.
    
    Args:
        resume_text_chunks (list): A list of text chunks from the resume.

    Returns:
        resume_embeddings (list): A list of embeddings for each text chunk.
    """
     # Generate embeddings for the text chunks
    resume_embeddings = model.encode(resume_text_chunks, batch_size=8, show_progress_bar=True)

    return resume_embeddings

def summarizer(pdf):
    """
    Summarize the provided resume.

    Args:
        pdf (list): A list containing a Document object.

    Returns:
        summarized_resume (str): A summary of the resume.
    """

    template = """
    Summarize the provided resume into three concise sentences:
    {context}
    
    Highlight the applicant's core competencies, relevant experience, and educational background. Include specific details such as job titles, companies, degrees, and certifications. If available, briefly mention desired job roles and contact information. """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = create_stuff_documents_chain(llm, prompt)

    summarized_resume = chain.invoke({"context": pdf})
    return summarized_resume

for record in collection.find():
    try:
        resume = record['resume']
        pdf, resume_text_chunks = preprocess_resume(resume)
        embeddings = embed_function(resume_text_chunks)
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        summary = summarizer(pdf)

        # Prepare documents for the vector search collection
        documents_for_indexing = [
            {
                "_id": ObjectId(),  # Generate a new unique ID for the vector index document
                "embedding": embedding,  
                "chunk_text": chunk,
                "summary": summary,
                "metadata": {
                    "original_id": str(record['_id']),  # Reference to the original document
                    "firstName": record.get('firstName', 'N/A'),
                    "lastName": record.get('lastName', 'N/A'),
                    "created_at": record.get('created_at', None),
                    "chunk_index": idx  # Index of the chunk in the resume
                }
            }
            for idx, (chunk, embedding) in enumerate(zip(resume_text_chunks, embeddings_list))
        ]

        # Insert documents into the vector index collection
        vector_index_collection.insert_many(documents_for_indexing)

        print("vector index populated")
    except Exception as e:
        print(f'An error occurred while processing document ID {record.get("_id")}: {e}')