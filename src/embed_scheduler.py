from pymongo import MongoClient
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import logging
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Configure logging
logging.basicConfig(level=logging.INFO)
scheduler_logger = logging.getLogger('task_scheduler_logger')

model = SentenceTransformer('all-MiniLM-L6-v2')

def emded_doc():
    """This function is a task scheduling function to embed resumes in my mongo db database every 24 hours"""

    client = MongoClient("mongodb+srv://whitebandit02:dontlogmein@resume.ygrttul.mongodb.net/?retryWrites=true&w=majority&appName=resume")
    db = client["resumeDB"]
    collection = db["resumeCollection"]

    # Define the cutoff date
    update_range = datetime.utcnow() - timedelta(hours=24)  # new data over the last 24hrs

    for record in collection.find({'created_at': {'$gt': update_range}, 'resume_embedding': {'$exists': False}}):
        try:
            resume = record['resume']
            reader = pymupdf.open(resume)
            text = ""
            
            # Extract text from all pages
            for page_number in range(reader.page_count):
                page = reader[page_number]
                text += page.get_text()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
            resume_text_chunks = text_splitter.split_text(text)
            resume_embedding = model.encode(resume_text_chunks).tolist()

            collection.update_one(
                {'_id': record['_id']},
                {'$set': {'resume_embedding': resume_embedding}}
            )

            scheduler_logger.info(f"Updated document ID: {record['_id']}")
        except Exception as e:
            scheduler_logger.error(f'An error occurred while processing document ID {record["_id"]}: {e}')
        
if __name__ == "__main__":
    emded_doc()
