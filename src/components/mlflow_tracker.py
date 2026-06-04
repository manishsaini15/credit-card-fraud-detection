import sys
import tempfile

import mlflow
import mlflow.sklearn

import pandas as pd

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)


class MLflowTracker:
    """
    Handles all MLflow operations.

    Responsibilities
    ------------------------
    1. Create experiment
    2. Create run
    3. Log parameters
    4. Log metrics
    5. Log model
    6. Register model
    7. Log artifacts
    """

    def __init__(self, config):

        self.config = config

    def log_model_run(
        self,
        model,
        model_name: str,
        train_f1: float,
        test_f1: float,
        precision: float,
        recall: float,
        roc_auc: float
    ):
        """
        Log model information into MLflow.
        """

        try:

            logger.info(
                "Starting MLflow tracking."
            )

            # ==========================================
            # Configure MLflow
            # ==========================================

            mlflow.set_tracking_uri(
                self.config.tracking_uri
            )

            mlflow.set_experiment(
                self.config.experiment_name
            )

            # ==========================================
            # Start Run
            # ==========================================

            with mlflow.start_run():

                logger.info(
                    "MLflow run started."
                )

                # ==========================================
                # Model Information
                # ==========================================

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                # ==========================================
                # Hyperparameters
                # ==========================================

                if hasattr(model, "get_params"):

                    params = model.get_params()

                    for key, value in params.items():

                        mlflow.log_param(
                            key,
                            str(value)
                        )

                logger.info(
                    "Hyperparameters logged."
                )

                # ==========================================
                # Metrics
                # ==========================================

                mlflow.log_metric(
                    "train_f1_score",
                    train_f1
                )

                mlflow.log_metric(
                    "test_f1_score",
                    test_f1
                )

                mlflow.log_metric(
                    "precision",
                    precision
                )

                mlflow.log_metric(
                    "recall",
                    recall
                )

                mlflow.log_metric(
                    "roc_auc",
                    roc_auc
                )

                logger.info(
                    "Metrics logged."
                )

                # ==========================================
                # Feature Importance
                # ==========================================

                if hasattr(
                    model,
                    "feature_importances_"
                ):

                    feature_importance = pd.DataFrame({

                        "importance":
                        model.feature_importances_
                    })

                    with tempfile.NamedTemporaryFile(
                        suffix=".csv",
                        delete=False
                    ) as temp_file:

                        feature_importance.to_csv(
                            temp_file.name,
                            index=False
                        )

                        mlflow.log_artifact(
                            temp_file.name
                        )

                    logger.info(
                        "Feature importance logged."
                    )

                # ==========================================
                # Log Model
                # ==========================================

                mlflow.sklearn.log_model(

                    sk_model=model,

                    artifact_path="model",

                    registered_model_name=
                    self.config
                    .registered_model_name
                )

                logger.info(
                    "Model registered successfully."
                )

                logger.info(
                    "MLflow tracking completed."
                )

        except Exception as e:

            logger.error(
                f"MLflow tracking failed: {e}"
            )

            raise CustomException(
                e,
                sys
            )