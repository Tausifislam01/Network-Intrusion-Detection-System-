from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.inference import predict_dataframe


app = FastAPI(
    title="Network Intrusion Detection System API",
    description="FastAPI backend for CICIDS2017 intrusion detection predictions.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "NIDS API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "xgboost_model",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()
        decoded = contents.decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(decoded), low_memory=False)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty.",
            )

        predictions = predict_dataframe(df)

        preview = predictions.head(100).to_dict(orient="records")
        prediction_counts = predictions["prediction"].value_counts().to_dict()

        response = {
            "filename": file.filename,
            "rows_processed": len(predictions),
            "prediction_counts": prediction_counts,
            "preview_limit": 100,
            "preview": preview,
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )