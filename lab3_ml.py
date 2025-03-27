import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

class TextInput(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_text(input_data: TextInput):
    result = classifier(input_data.text)
    return {"label": result[0]['label'], "score": result[0]['score']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
