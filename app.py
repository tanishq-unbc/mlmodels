import joblib
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("decisiontree.pkl")

@app.get('/')
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(age: int, salary: int):
    prediction = model.predict([[age, salary]])
    
    return {
        "age": age,
        "salary": salary,
        "pred": int(prediction[0])
    }