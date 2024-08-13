from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()
from resume_summarizer import process_text, summarizer
from utils import fireworks_llm


uri = os.environ.get("MONGO_URI")
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")

client = MongoClient(uri)
db = client["resumeDB"]
collection = db["resumeCollection"]
vector_index_collection = db["resume_vector_index"]

llm = fireworks_llm

def insert_summary():
    for record in vector_index_collection.find():
        try:
            resume = record['resume']
            # collects pdf from database and pass it to the process_text function.
            # returns pdf
            pdf = process_text(resume)

            summary = summarizer(pdf, llm)
            
            # Prepare documents for summary inserion
            summary_for_indexing = {
                '_id': record['_id'],
                'summary': summary
            }

            # Update the document in the vector index collection
            vector_index_collection.update_one(
                {'_id': record['_id']},
                {'$set': {'summary': summary}}
            )
            print(f"inserted summary for: {record['_id']}")

        except Exception as e:
            print(f'An error occurred while processing document ID {record.get("_id")}: {e}')

insert_summary()