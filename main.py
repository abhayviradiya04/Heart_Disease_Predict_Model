from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
    
app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load the model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('model_features.pkl')

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("Main.html", {"request": request})


@app.get("/form.html", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    Sex: int = Form(...),
    ChestPainType: int = Form(...),
    RestingBP: int = Form(...),
    Cholesterol: int = Form(...),
    FastingBS: int = Form(...),
    RestingECG: int = Form(...),
    MaxHR: int = Form(...),
    ExerciseAngina: int = Form(...),
    Oldpeak: float = Form(...),
    ST_Slope: int = Form(...)
):
    # Prepare input data as DataFrame
    input_data = pd.DataFrame([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                                 RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]],
                               columns=features)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Convert prediction to readable format
    result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

    return templates.TemplateResponse("result.html", {"request": request, "prediction": result})
