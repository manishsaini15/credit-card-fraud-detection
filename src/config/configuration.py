import os
import yaml

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    MLflowConfig,
    ModelEvaluationConfig,
    ModelRegistryConfig
)



class ConfigurationManager:
    """
    Creates configuration objects
    for every pipeline stage.
    """

    def __init__(self):

        self.params = self._read_yaml(
            os.path.join(
                "configs",
                "params.yaml"
            )
        )

    # =====================================================
    # READ YAML
    # =====================================================

    @staticmethod
    def _read_yaml(file_path: str):

        with open(
            file_path,
            "r"
        ) as yaml_file:

            return yaml.safe_load(
                yaml_file
            )

    # =====================================================
    # DATA INGESTION CONFIG
    # =====================================================

    def get_data_ingestion_config(self):

        artifact_dir = os.path.join(
            "artifacts",
            "data_ingestion"
        )

        os.makedirs(
            artifact_dir,
            exist_ok=True
        )

        return DataIngestionConfig(

            raw_data_path=os.path.join(
                "data",
                "raw",
                "creditcard.csv"
            ),

            train_data_path=os.path.join(
                artifact_dir,
                "train.csv"
            ),

            test_data_path=os.path.join(
                artifact_dir,
                "test.csv"
            )
        )

    # =====================================================
    # DATA VALIDATION CONFIG
    # =====================================================

    def get_data_validation_config(self):

        artifact_dir = os.path.join(
            "artifacts",
            "data_validation"
        )

        os.makedirs(
            artifact_dir,
            exist_ok=True
        )

        return DataValidationConfig(

            schema_file_path=os.path.join(
                "configs",
                "schema.yaml"
            ),

            validation_report_path=os.path.join(
                artifact_dir,
                "validation_report.txt"
            )
        )

    # =====================================================
    # DATA TRANSFORMATION CONFIG
    # =====================================================

    def get_data_transformation_config(self):

        artifact_dir = os.path.join(
            "artifacts",
            "data_transformation"
        )

        os.makedirs(
            artifact_dir,
            exist_ok=True
        )

        return DataTransformationConfig(

            preprocessor_object_file_path=os.path.join(
                artifact_dir,
                "preprocessor.pkl"
            ),

            transformed_train_file_path=os.path.join(
                artifact_dir,
                "train.npy"
            ),

            transformed_test_file_path=os.path.join(
                artifact_dir,
                "test.npy"
            )
        )

    # =====================================================
    # MODEL TRAINER CONFIG
    # =====================================================

    def get_model_trainer_config(self):

        artifact_dir = os.path.join(
            "artifacts",
            "model_trainer"
        )

        os.makedirs(
            artifact_dir,
            exist_ok=True
        )

        return ModelTrainerConfig(

            trained_model_file_path=os.path.join(
                artifact_dir,
                "model.pkl"
            ),

            expected_accuracy=
            self.params["model_trainer"][
                "expected_accuracy"
            ]
        )

    # =====================================================
    # MLFLOW CONFIG
    # =====================================================

    def get_mlflow_config(self):

        return MLflowConfig(

            experiment_name=
            self.params["mlflow"][
                "experiment_name"
            ],

            tracking_uri=
            self.params["mlflow"][
                "tracking_uri"
            ],

            registered_model_name=
            self.params["mlflow"][
                "registered_model_name"
            ]
        )
    
# =====================================================
# MODEL EVALUATION CONFIG
# =====================================================

    def get_model_evaluation_config(self):

        artifact_dir = os.path.join(
            "artifacts",
            "model_evaluation"
        )

        os.makedirs(
            artifact_dir,
            exist_ok=True
        )

        return ModelEvaluationConfig(

            evaluation_report_file_path=
            os.path.join(
                artifact_dir,
                "evaluation_report.json"
            ),

            improvement_threshold=0.01
        )
# =====================================================
# MODEL REGISTRY CONFIG
# =====================================================

    def get_model_registry_config(self):

        registry_dir = os.path.join(
            "artifacts",
            "model_registry"
        )

        os.makedirs(
            registry_dir,
            exist_ok=True
        )

        return ModelRegistryConfig(

            registry_dir=registry_dir,

            registry_file_path=
            os.path.join(
                registry_dir,
                "registry.json"
            )
        )