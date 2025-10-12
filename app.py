from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.post("/predict/")
def predict_emotion(data: TextInput):
    text = data.text
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return {"emotion": prediction}
