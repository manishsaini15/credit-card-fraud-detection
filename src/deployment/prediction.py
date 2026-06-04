import numpy as np
import pandas as pd
from src.logger.logger import logger
from src.deployment.model_loader import ModelLoader


class PredictionPipeline:
    """
    End-to-end prediction pipeline
    """

    def __init__(self):

        self.loader = ModelLoader()

        # Load once (performance optimization)
        self.model, self.preprocessor = self.loader.load_all()

        logger.info("Model and preprocessor loaded at startup")

    def predict(self, input_data):

        try:
            logger.info("Prediction started")

            # Convert input to dict
            data_dict = input_data.model_dump()

            # Convert to DataFrame
            df = pd.DataFrame([data_dict])

            logger.info("Input converted to DataFrame")

            # Preprocessing
            processed_data = self.preprocessor.transform(df)

            # Prediction
            prediction = self.model.predict(processed_data)[0]

            probability = self.model.predict_proba(processed_data)[0][1]

            logger.info(f"Prediction: {prediction}, Probability: {probability}")

            # Result mapping
            result = "Fraud Transaction" if prediction == 1 else "Normal Transaction"

            return {
                "prediction": int(prediction),
                "fraud_probability": float(probability),
                "result": result
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e