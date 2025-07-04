from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

class Settings(BaseSettings):
    WATSONX_API_KEY: str
    PROJECT_ID: str
    MODEL_ID: str
    PINECONE_API_KEY: str

settings = Settings()
