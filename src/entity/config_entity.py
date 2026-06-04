from dataclasses import dataclass


# ==================================================
# DATA INGESTION CONFIG
# ==================================================
# Stores paths required for:
# - Raw Dataset
# - Train Dataset
# - Test Dataset
# ==================================================

@dataclass
class DataIngestionConfig:

    raw_data_path: str

    train_data_path: str

    test_data_path: str


# ==================================================
# DATA VALIDATION CONFIG
# ==================================================
# Stores:
# - Schema File Path
# - Validation Report Path
# ==================================================

@dataclass
class DataValidationConfig:

    schema_file_path: str

    validation_report_path: str


# ==================================================
# DATA TRANSFORMATION CONFIG
# ==================================================
# Stores:
# - Preprocessor Object Path
# - Transformed Train Dataset
# - Transformed Test Dataset
# ==================================================

@dataclass
class DataTransformationConfig:

    preprocessor_object_file_path: str

    transformed_train_file_path: str

    transformed_test_file_path: str


# ==================================================
# MODEL TRAINER CONFIG
# ==================================================
# Stores:
# - Best Model Save Path
# - Minimum Acceptable F1 Score
# ==================================================

@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str

    expected_accuracy: float


# ==================================================
# MLFLOW CONFIG
# ==================================================
# Stores:
# - Experiment Name
# - Tracking Server URI
# - Registered Model Name
#
# Local Example:
# tracking_uri = http://127.0.0.1:5000
#
# Azure Example:
# tracking_uri = Azure ML Workspace URI
# ==================================================

@dataclass
class MLflowConfig:

    experiment_name: str

    tracking_uri: str

    registered_model_name: str

# ==================================================
# MODEL EVALUATION CONFIG
# ==================================================

@dataclass
class ModelEvaluationConfig:

    evaluation_report_file_path: str

    improvement_threshold: float


# ==================================================
# MODEL REGISTRY CONFIG
# ==================================================

@dataclass
class ModelRegistryConfig:

    registry_dir: str

    registry_file_path: str