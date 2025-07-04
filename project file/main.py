from fastapi import FastAPI
from pydantic import BaseModel
from app.services.watson import ask_watson

app = FastAPI()

class UserQuery(BaseModel):
    question: str

@app.post("/ask")
def get_response(query: UserQuery):
    answer = ask_watson(query.question)
    return {"response":answer}
