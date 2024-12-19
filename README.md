## Heart Disease Prediction Model
This project is a web application built using FastAPI for predicting the risk of heart disease. It employs a trained machine learning model to process user-provided clinical data and returns predictions with high accuracy.

Features
User-Friendly Interface: Allows users to input clinical data such as age, cholesterol levels, and blood pressure.
Machine Learning Integration: Uses a pre-trained model to predict the likelihood of heart disease.
Data Preprocessing: Includes feature scaling and preparation for accurate model predictions.
Templates: Implements Jinja2 templates for dynamic HTML pages like forms and result displays.
Scalable APIs: API endpoints designed for seamless communication between front-end and back-end.
Technologies Used
FastAPI: Back-end framework for handling requests and responses.
Scikit-learn: For training and using the machine learning model.
Pandas: For data manipulation and input preparation.
Jinja2: For dynamic HTML rendering.
Joblib: For efficient model and scaler serialization.
How to Run
Clone the repository.
Install required dependencies using pip install -r requirements.txt.
Run the FastAPI server with uvicorn main:app --reload.
Access the application at http://127.0.0.1:8000/.
