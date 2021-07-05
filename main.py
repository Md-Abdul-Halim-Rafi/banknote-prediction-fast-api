from starlette.routing import Host
import uvicorn
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI

from bank_note import BankNote

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return { "hola": "meefee" }

@app.post("/predict")
def predict_banknote(data: BankNote):

    data = data.dict()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if (prediction[0] > 0.5):
        prediction = "Fake Note"
    else: 
        prediction = "It is a bank note"

    return { "prediction": prediction }

if __name__ == "__main__":
    uvicorn.run(app, Host="127.0.0.1", port=8000)