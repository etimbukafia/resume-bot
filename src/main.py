from db_connect import Database
from chain import retrieval_chain
from config import Configs
import logging
import gradio as gr
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

configs = Configs()

vector_store = None
llm = None

async def startup_event():
    global vector_store, llm

    # Initialize the database connection
    await configs.initialize()

    vector_store = configs.get_vector_store()
    logger.info("Vector store initialized successfully")

    # Load your LLM
    llm = configs.get_llm()
    logger.info("LLM loaded successfully")

async def shutdown_db_client():
    await Database.close()
    logger.info("Database connection closed successfully")

def run_startup_tasks():
    """Run asynchronous startup tasks synchronously"""
    asyncio.run(startup_event())

def run_shutdown_tasks():
    """Run asynchronous shutdown tasks synchronously"""
    asyncio.run(shutdown_db_client())

async def resume_chat(query):
    try:
        # Directly await the retrieval_chain since Gradio supports async functions
        answer = await retrieval_chain(vector_store, llm, query)
        if answer is None:
            raise Exception("Failed to retrieve answer")
        return {"result": answer}
    except Exception as e:
        logger.error(f"Error in resume_chat function: {e}")
        raise Exception("An error occurred while processing your request.")

# Initialize Gradio interface
interface = gr.Interface(fn=resume_chat, inputs=gr.Textbox(lines=2, placeholder="Query here"), outputs="text", allow_flagging="never")

if __name__ == "__main__":
    # Run startup tasks before launching the interface
    run_startup_tasks()
    try:
        interface.launch()
    finally:
        # Ensure shutdown tasks are run after closing the interface
        run_shutdown_tasks()

