import sys
import numpy as np

# =====================================================
# MACHINE LEARNING MODELS
# =====================================================

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =====================================================
# EVALUATION METRICS
# =====================================================

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# =====================================================
# PROJECT IMPORTS
# =====================================================

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)

from src.entity.config_entity import (
    ModelTrainerConfig
)

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from src.utils.common import (
    save_object
)

from src.config.configuration import (
    ConfigurationManager
)

from src.components.mlflow_tracker import (
    MLflowTracker
)


class ModelTrainer:
    """
    Train multiple models,
    select the best one,
    track it in MLflow,
    and save it.
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact
    ):

        self.config = config

        self.transformation_artifact = (
            transformation_artifact
        )

    def initiate_model_trainer(self):

        try:

            logger.info(
                "Model Training Started"
            )

            # =====================================================
            # LOAD DATA
            # =====================================================

            train_arr = np.load(
                self.transformation_artifact
                .transformed_train_file_path
            )

            test_arr = np.load(
                self.transformation_artifact
                .transformed_test_file_path
            )

            # =====================================================
            # SPLIT FEATURES & TARGET
            # =====================================================

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logger.info(
                "Feature-target separation completed."
            )

            # =====================================================
            # DEFINE MODELS
            # =====================================================

            models = {

                "Logistic Regression":

                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000
                ),

                "Random Forest":

                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "XGBoost":

                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                )
            }

            # =====================================================
            # TRAIN & EVALUATE ALL MODELS
            # =====================================================

            best_model = None
            best_model_name = None
            best_model_score = 0

            for model_name, model in models.items():

                logger.info(
                    f"Training {model_name}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

                score = f1_score(
                    y_test,
                    y_pred
                )

                logger.info(
                    f"{model_name} F1 Score : {score}"
                )

                if score > best_model_score:

                    best_model_score = score

                    best_model_name = model_name

                    best_model = model

            logger.info(
                f"Best Model : {best_model_name}"
            )

            logger.info(
                f"Best F1 Score : {best_model_score}"
            )

            # =====================================================
            # QUALITY CHECK
            # =====================================================

            if best_model_score < (
                self.config.expected_accuracy
            ):

                raise Exception(
                    "No model met the minimum threshold."
                )

            # =====================================================
            # SAVE MODEL
            # =====================================================

            save_object(

                self.config
                .trained_model_file_path,

                best_model
            )

            logger.info(
                "Model saved successfully."
            )

            # =====================================================
            # TRAIN METRICS
            # =====================================================

            train_pred = best_model.predict(
                X_train
            )

            train_f1 = f1_score(
                y_train,
                train_pred
            )

            # =====================================================
            # TEST METRICS
            # =====================================================

            test_pred = best_model.predict(
                X_test
            )

            test_f1 = f1_score(
                y_test,
                test_pred
            )

            precision = precision_score(
                y_test,
                test_pred
            )

            recall = recall_score(
                y_test,
                test_pred
            )

            # =====================================================
            # ROC-AUC
            # =====================================================

            if hasattr(
                best_model,
                "predict_proba"
            ):

                y_prob = best_model.predict_proba(
                    X_test
                )[:, 1]

                roc_auc = roc_auc_score(
                    y_test,
                    y_prob
                )

            else:

                roc_auc = 0.0

            logger.info(
                f"Train F1 : {train_f1}"
            )

            logger.info(
                f"Test F1 : {test_f1}"
            )

            logger.info(
                f"Precision : {precision}"
            )

            logger.info(
                f"Recall : {recall}"
            )

            logger.info(
                f"ROC AUC : {roc_auc}"
            )

            # =====================================================
            # MLFLOW TRACKING
            # =====================================================

            config_manager = (
                ConfigurationManager()
            )

            mlflow_config = (
                config_manager
                .get_mlflow_config()
            )

            tracker = MLflowTracker(
                mlflow_config
            )

            tracker.log_model_run(

                model=best_model,

                model_name=
                best_model_name,

                train_f1=
                train_f1,

                test_f1=
                test_f1,

                precision=
                precision,

                recall=
                recall,

                roc_auc=
                roc_auc
            )

            logger.info(
                "MLflow tracking completed."
            )

            # =====================================================
            # CREATE ARTIFACT
            # =====================================================

            model_trainer_artifact = (

                ModelTrainerArtifact(

                    trained_model_file_path=
                    self.config
                    .trained_model_file_path,

                    model_name=
                    best_model_name,

                    train_f1=
                    train_f1,

                    test_f1=
                    test_f1,

                    precision=
                    precision,

                    recall=
                    recall,

                    roc_auc=
                    roc_auc
                )
            )

            logger.info(
                "Model Training Completed"
            )

            return (
                model_trainer_artifact
            )

        except Exception as e:

            logger.error(
                f"Model training failed : {e}"
            )

            raise CustomException(
                e,
                sys
            )