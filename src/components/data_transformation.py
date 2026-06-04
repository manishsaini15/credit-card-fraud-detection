import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)

from src.entity.config_entity import (
    DataTransformationConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)

from src.utils.common import (
    save_object
)


class DataTransformation:
    """
    Data Transformation Component

    Responsibilities:
    -----------------
    1. Read train and test datasets
    2. Separate features and target
    3. Apply preprocessing pipeline
    4. Save preprocessor object
    5. Save transformed train/test arrays
    """

    def __init__( self,
        config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact):
        """
        Constructor Parameters
        ----------
        config:
            Data transformation configuration

        ingestion_artifact:
            Output from Data Ingestion stage

        validation_artifact:
            Output from Data Validation stage
        """

        self.config = config
        self.ingestion_artifact = ingestion_artifact
        self.validation_artifact = validation_artifact

    def get_data_transformer_object(self):
        """
        Create preprocessing pipeline.

        Returns
        -------
        ColumnTransformer
        """

        try:

            logger.info("Creating preprocessing pipeline")

            # ----------------------------------------
            # Columns that need scaling
            # ----------------------------------------

            numerical_features = ["Time", "Amount"]

            # ----------------------------------------
            # Numerical Pipeline
            # ----------------------------------------

            num_pipeline = Pipeline(
                steps=[ ( "scaler",StandardScaler())])

            # ----------------------------------------
            # Column Transformer
            # ----------------------------------------

            preprocessor = ColumnTransformer(

                transformers=[("num_pipeline", num_pipeline,numerical_features) ],
                remainder="passthrough")

            logger.info("Preprocessing pipeline created successfully" )

            return preprocessor

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def initiate_data_transformation(self):
        """
        Main Data Transformation Method

        Returns
        -------
        DataTransformationArtifact
        """

        try:

            logger.info("Data Transformation Started" )

            # ----------------------------------------
            # Ensure Validation Passed
            # ----------------------------------------

            if not self.validation_artifact.validation_status:

                raise Exception(
                    "Data Validation Failed. "
                    "Transformation cannot proceed."
                )

            # ----------------------------------------
            # Load Train Dataset
            # ----------------------------------------

            train_df = pd.read_csv(
                self.ingestion_artifact.train_file_path
            )

            logger.info(
                f"Train dataset loaded with shape "
                f"{train_df.shape}"
            )

            # ----------------------------------------
            # Load Test Dataset
            # ----------------------------------------

            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            logger.info(
                f"Test dataset loaded with shape "
                f"{test_df.shape}")

            # ----------------------------------------
            # Target Column
            # ----------------------------------------

            target_column = "Class"

            # ----------------------------------------
            # Separate Train Features and Target
            # ----------------------------------------

            X_train = train_df.drop(columns=[target_column],axis=1)

            y_train = train_df[target_column]

            # ----------------------------------------
            # Separate Test Features and Target
            # ----------------------------------------

            X_test = test_df.drop(columns=[target_column],axis=1)

            y_test = test_df[target_column]

            logger.info("Feature-target separation completed")

            # ----------------------------------------
            # Create Preprocessor
            # ----------------------------------------

            preprocessor = (self.get_data_transformer_object())

            # ----------------------------------------
            # Fit and Transform Training Data
            # ----------------------------------------

            X_train_transformed = (preprocessor.fit_transform( X_train) )

            logger.info("Training data transformed successfully")

            # ----------------------------------------
            # Transform Test Data
            # ----------------------------------------

            X_test_transformed = (preprocessor.transform(X_test))

            logger.info("Testing data transformed successfully")

            # ----------------------------------------
            # Combine Features and Target
            # ----------------------------------------

            train_arr = np.c_[X_train_transformed,np.array(y_train)]

            test_arr = np.c_[X_test_transformed,np.array(y_test)]

            logger.info("Train and Test arrays created")

            # ----------------------------------------
            # Save Preprocessor Object
            # ----------------------------------------

            save_object(file_path=self.config.preprocessor_object_file_path,
                        obj=preprocessor)

            logger.info( "Preprocessor object saved")

            # ----------------------------------------
            # Save Train Array
            # ----------------------------------------

            np.save(self.config.transformed_train_file_path,train_arr)

            # ----------------------------------------
            # Save Test Array
            # ----------------------------------------

            np.save(self.config.transformed_test_file_path,test_arr)

            logger.info("Transformed arrays saved successfully")

            # ----------------------------------------
            # Return Artifact
            # ----------------------------------------

            transformation_artifact = (
                DataTransformationArtifact(

                    transformed_train_file_path=self.config.transformed_train_file_path,

                    transformed_test_file_path= self.config.transformed_test_file_path,

                    preprocessor_object_file_path= self.config.preprocessor_object_file_path
                )
            )

            logger.info("Data Transformation Completed")

            return transformation_artifact

        except Exception as e:

            logger.error(
                "Error occurred during data transformation"
            )

            raise CustomException(
                e,
                sys
            )