# =====================================
# Perishable Goods Prediction API
# =====================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# Create FastAPI app
app = FastAPI(title="Perishable Goods Prediction API", version="1.0.0")

# Define what the incoming data should look like
class PredictionRequest(BaseModel):
    Store_ID: int
    Product_ID: int
    Supplier_ID: int
    Temperature: float
    Rainfall: float
    # Add more fields if your model expects them

# Create a prediction endpoint
@app.post("/predict")
def predict_sales(request: PredictionRequest):
    try:
        # Convert incoming data into a DataFrame
        input_df = pd.DataFrame([request.dict()])

        # Load the saved Random Forest model
        with open("rf_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Make prediction
        prediction = model.predict(input_df)[0]

        return {"predicted_units_sold": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

