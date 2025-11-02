from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
import numpy as np # Import numpy for safety, though a list works too

# Create FastAPI app
app = FastAPI(title="Iris Classifier API 🌸")

# Load model
try:
    model = joblib.load("model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classifier API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        # Return a 503 Service Unavailable if model isn't loaded
        return {"error": "Model not loaded or is unavailable"}, 503

    try:
        # 1. Convert input to a 2D list or numpy array
        # The order MUST match the order of the columns in iris.data
        # (sepal_length, sepal_width, petal_length, petal_width)
        input_array = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]
        
        # You could also use numpy explicitly:
        # input_array = np.array([[
        #     data.sepal_length,
        #     data.sepal_width,
        #     data.petal_length,
        #     data.petal_width
        # ]])

        # 2. Make prediction
        # This passes a 2D list, which scikit-learn handles as a numpy array
        prediction = model.predict(input_array)
        
        # 3. Map to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = class_names[int(prediction[0])]
        
        return {
            "predicted_class": predicted_class,
            "predicted_index": int(prediction[0])
        }

    except Exception as e:
        # Catch any errors during prediction and return a proper error message
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}, 500