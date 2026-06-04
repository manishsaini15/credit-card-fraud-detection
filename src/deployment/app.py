from fastapi import FastAPI
from src.logger.logger import logger

from src.deployment.schemas import TransactionSchema
from src.deployment.prediction import PredictionPipeline

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0",
    description="Production ML API for fraud detection"
)

pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline

    pipeline = PredictionPipeline()

    logger.info("API initialized successfully")


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running 🚀"
    }


@app.post("/predict")
def predict(data: TransactionSchema):

    try:

        result = pipeline.predict(data)

        return {
            "status": "success",
            "prediction": result["prediction"],
            "fraud_probability": result["fraud_probability"],
            "result": result["result"]
        }

    except Exception as e:

        logger.error(str(e))

        return {
            "status": "error",
            "message": str(e)
        }