from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


app = FastAPI()
path = 'emotion_detect/'
path_tokenizer = 'emotion_detect_tokenizer/'
tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(path)


def preprocess(query):
    return tokenizer(query)

def get_output(query):
    inputs = tokenizer(query,return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1)
    return prediction


@app.get("/")
def ping():
    return {"message":"Hello world!"}


@app.get("/sentence/{query}")
def get_emotion(query:str):
    print(query)
    prediction = get_output(query).tolist()
    print(prediction)
    return prediction 