from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
import io

model = SentenceTransformer('all-MiniLM-L6-v2')

uri = os.environ.get("MONGO_URI")
client = MongoClient(uri)
db = client["resumeDB"]
collection = db["resumeCollection"]

for record in collection.find({'resume_embedding': {'$exists': False}}):
    try:
        resume = record['resume']
        # Directly pass the binary data to pymupdf.open()
        reader = pymupdf.open(stream=resume, filetype="pdf")
        
        text = ""
            
        # Extract text from all pages
        for page_number in range(reader.page_count):
            page = reader[page_number]
            text += page.get_text() + "\n"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        resume_text_chunks = text_splitter.split_text(text)
        resume_embedding = model.encode(resume_text_chunks).tolist()

        collection.update_one(
            {'_id': record['_id']},
            {'$set': {'resume_embedding': resume_embedding}}
        )

        print(f"Updated document ID: {record['_id']}")
    except Exception as e:
        print(f'An error occurred while processing document ID {record["_id"]}: {e}')
        