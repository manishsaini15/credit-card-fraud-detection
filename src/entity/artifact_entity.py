from dataclasses import dataclass


# ==================================================
# DATA INGESTION ARTIFACT
# ==================================================
# Output of Data Ingestion Stage
#
# Contains:
# - Train dataset path
# - Test dataset path
#
# Consumed By:
# - Data Validation
# ==================================================

@dataclass
class DataIngestionArtifact:

    train_file_path: str
    test_file_path: str

    def __str__(self):

        return (
            f"\nTrain File Path        : {self.train_file_path}"
            f"\nTest File Path         : {self.test_file_path}"
        )


# ==================================================
# DATA VALIDATION ARTIFACT
# ==================================================
# Output of Data Validation Stage
#
# Contains:
# - Validation Status
# - Validation Report Path
#
# Consumed By:
# - Data Transformation
# ==================================================

@dataclass
class DataValidationArtifact:

    validation_status: bool
    validation_report_path: str

    def __str__(self):

        return (
            f"\nValidation Status      : {self.validation_status}"
            f"\nValidation Report Path : {self.validation_report_path}"
        )


# ==================================================
# DATA TRANSFORMATION ARTIFACT
# ==================================================
# Output of Data Transformation Stage
#
# Contains:
# - train.npy
# - test.npy
# - preprocessor.pkl
#
# Consumed By:
# - Model Trainer
# ==================================================

@dataclass
class DataTransformationArtifact:

    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str

    def __str__(self):

        return (
            f"\nTransformed Train File : "
            f"{self.transformed_train_file_path}"

            f"\nTransformed Test File  : "
            f"{self.transformed_test_file_path}"

            f"\nPreprocessor Path      : "
            f"{self.preprocessor_object_file_path}"
        )


# ==================================================
# MODEL TRAINER ARTIFACT
# ==================================================
# Output of Model Training Stage
#
# Contains:
# - Best Model Name
# - Saved Model Path
# - Train F1 Score
# - Test F1 Score
# - Precision
# - Recall
# - ROC AUC
#
# Consumed By:
# - MLflow Tracking
# - Model Evaluation
# - Model Registry
# - Azure ML Deployment
# ==================================================

@dataclass
class ModelTrainerArtifact:

    trained_model_file_path: str

    model_name: str

    train_f1: float

    test_f1: float

    precision: float

    recall: float

    roc_auc: float

    def __str__(self):

        return (
            f"\nModel Name             : {self.model_name}"

            f"\nTrain F1 Score         : "
            f"{self.train_f1:.4f}"

            f"\nTest F1 Score          : "
            f"{self.test_f1:.4f}"

            f"\nPrecision              : "
            f"{self.precision:.4f}"

            f"\nRecall                 : "
            f"{self.recall:.4f}"

            f"\nROC AUC                : "
            f"{self.roc_auc:.4f}"

            f"\nSaved Model Path       : "
            f"{self.trained_model_file_path}"
        )
# ==================================================
# MODEL EVALUATION ARTIFACT
# ==================================================

@dataclass
class ModelEvaluationArtifact:

    is_model_accepted: bool

    improvement_score: float

    evaluation_report_path: str

    def __str__(self):

        return (

            f"\nModel Accepted       : {self.is_model_accepted}"
            f"\nImprovement Score    : {self.improvement_score:.4f}"
            f"\nEvaluation Report    : {self.evaluation_report_path}"
        )
# ==================================================
# MODEL REGISTRY ARTIFACT
# ==================================================

@dataclass
class ModelRegistryArtifact:

    registered_model_path: str

    model_version: int

    is_production_model: bool

    def __str__(self):

        return (

            f"\nRegistered Model : {self.registered_model_path}"
            f"\nVersion          : {self.model_version}"
            f"\nProduction Model : {self.is_production_model}"
        )