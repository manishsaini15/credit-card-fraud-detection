from src.config.configuration import ConfigurationManager

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_registry import ModelRegistry


if __name__ == "__main__":

    # =====================================================
    # CREDIT CARD FRAUD DETECTION MLOPS PIPELINE
    # =====================================================
    #
    # Pipeline Flow
    #
    # creditcard.csv
    #       │
    #       ▼
    # Data Ingestion
    #       │
    #       ▼
    # Data Validation
    #       │
    #       ▼
    # Data Transformation
    #       │
    #       ▼
    # Model Training
    #       │
    #       ▼
    # MLflow Tracking
    #       │
    #       ▼
    # Model Evaluation
    #       │
    #       ▼
    # Model Registry
    #
    # =====================================================

    # =====================================================
    # STEP 1 : CREATE CONFIGURATION MANAGER
    # =====================================================

    config = ConfigurationManager()

    # =====================================================
    # STEP 2 : DATA INGESTION
    # =====================================================
    #
    # Input:
    #   data/raw/creditcard.csv
    #
    # Output:
    #   artifacts/data_ingestion/train.csv
    #   artifacts/data_ingestion/test.csv
    #
    # =====================================================

    ingestion = DataIngestion(
        config.get_data_ingestion_config()
    )

    ingestion_artifact = (
        ingestion.initiate_data_ingestion()
    )

    print("\n" + "=" * 60)
    print("DATA INGESTION COMPLETED")
    print("=" * 60)
    print(ingestion_artifact)

    # =====================================================
    # STEP 3 : DATA VALIDATION
    # =====================================================
    #
    # Validations:
    #
    # - Schema Validation
    # - Required Columns Check
    # - Missing Values Check
    # - Duplicate Records Check
    #
    # Output:
    #   validation_report.txt
    #
    # =====================================================

    validation = DataValidation(
        config=config.get_data_validation_config(),
        ingestion_artifact=ingestion_artifact
    )

    validation_artifact = (
        validation.validate_dataset_schema()
    )

    print("\n" + "=" * 60)
    print("DATA VALIDATION COMPLETED")
    print("=" * 60)
    print(validation_artifact)

    # =====================================================
    # STEP 4 : DATA TRANSFORMATION
    # =====================================================
    #
    # Performs:
    #
    # - Feature / Target Separation
    # - Scaling
    # - Preprocessing Pipeline Creation
    #
    # Outputs:
    #
    # - train.npy
    # - test.npy
    # - preprocessor.pkl
    #
    # =====================================================

    transformer = DataTransformation(
        config=config.get_data_transformation_config(),
        ingestion_artifact=ingestion_artifact,
        validation_artifact=validation_artifact
    )

    transformation_artifact = (
        transformer.initiate_data_transformation()
    )

    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION COMPLETED")
    print("=" * 60)
    print(transformation_artifact)

    # =====================================================
    # STEP 5 : MODEL TRAINING
    # =====================================================
    #
    # Models:
    #
    # - Logistic Regression
    # - Random Forest
    # - XGBoost
    #
    # Responsibilities:
    #
    # - Train Models
    # - Compare Performance
    # - Select Best Model
    # - Save Best Model
    # - Calculate Metrics
    # - Log Metrics to MLflow
    #
    # Output:
    #
    # - model.pkl
    # - MLflow Run
    #
    # =====================================================

    trainer = ModelTrainer(
        config=config.get_model_trainer_config(),
        transformation_artifact=transformation_artifact
    )

    trainer_artifact = (
        trainer.initiate_model_trainer()
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(trainer_artifact)

    # =====================================================
    # STEP 6 : MODEL EVALUATION
    # =====================================================
    #
    # Responsibilities:
    #
    # - Validate model quality
    # - Generate evaluation report
    # - Decide whether model is acceptable
    #
    # Output:
    #
    # artifacts/model_evaluation/
    #     evaluation_report.json
    #
    # =====================================================

    evaluator = ModelEvaluation(
        config=config.get_model_evaluation_config(),
        trainer_artifact=trainer_artifact
    )

    evaluation_artifact = (
        evaluator.initiate_model_evaluation()
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 60)
    print(evaluation_artifact)

    # =====================================================
    # STEP 7 : MODEL REGISTRY
    # =====================================================
    #
    # Responsibilities:
    #
    # - Version Models
    # - Store Registry Metadata
    # - Track Production Model
    # - Enable Rollback
    #
    # Output:
    #
    # artifacts/model_registry/
    #     model_v1.pkl
    #     registry.json
    #
    # =====================================================

    registry = ModelRegistry(
        config=config.get_model_registry_config(),
        trainer_artifact=trainer_artifact
    )

    registry_artifact = (
        registry.register_model()
    )

    print("\n" + "=" * 60)
    print("MODEL REGISTRY COMPLETED")
    print("=" * 60)
    print(registry_artifact)

    # =====================================================
    # PIPELINE EXECUTION SUCCESSFUL
    # =====================================================

    print("\n" + "=" * 60)
    print("END-TO-END TRAINING PIPELINE COMPLETED")
    print("=" * 60)