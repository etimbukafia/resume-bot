from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import uvicorn
from dotenv import load_dotenv
load_dotenv()

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#password = os.environ.get("PASSWORD")
#username = os.environ.get("USERNAME")


app = FastAPI()

async def connect():
    uri = os.environ.get("MONGO_URI")
    print(uri)
    try:
        client = AsyncIOMotorClient(uri)
        db = client["resumeDB"]
        collection = db["resumeCollection"]
        logger.info("Connected to db")
    except ConnectionError as e:
        logger.error(f"Failed to connect to mongodb: {e}")
        raise
    except Exception as e:
        print(uri)
        logger.error(f"Something went wrong, can't connect to mongo: {e}")
        raise 
    return collection



@app.post('/submitresume')
async def submitResume(
    firstName: str = Form(...),
    lastName: str = Form(...),
    resume: UploadFile = File(...)
):
    try:
        collection = await connect()
        logger.info("connected to collection")

        #Process resume
        resume_content = await resume.read()
        request_data = {
            "firstName": firstName,
            "lastName": lastName,
            "resume": resume_content
        }

        await collection.insert_one(request_data)  #expects a dictionary
        logging.info("Data inserted")
        return {"message": "Data successfully inserted"}
    except Exception as e:
        logger.error(f"Someting went wrong: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, log_level="info")