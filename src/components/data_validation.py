import os
import sys
import yaml
import pandas as pd

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)

from src.entity.config_entity import (
    DataValidationConfig
)

from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact
)


class DataValidation:
    """
    DataValidation Component

    Responsibilities:
    1. Load schema from schema.yaml
    2. Validate required columns
    3. Validate missing values
    4. Validate duplicate records
    5. Generate validation report
    """

    def __init__( self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact
    ):
        """
        Constructor

        Parameters
        ----------
        config:
            Contains validation configuration
            (schema path, report path)

        ingestion_artifact:
            Contains output from Data Ingestion stage
            (train file path, test file path)
        """

        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def load_schema(self):
        """
        Load schema.yaml file

        Returns
        -------
        dict
            Schema dictionary
        """

        try:

            logger.info(
                "Loading schema file"
            )

            with open(
                self.config.schema_file_path,
                "r"
            ) as file:

                schema = yaml.safe_load(file)

            logger.info(
                "Schema loaded successfully"
            )

            return schema

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def validate_dataset_schema(self):
        """
        Main validation method

        Performs:
        1. Column validation
        2. Missing value validation
        3. Duplicate validation

        Returns
        -------
        DataValidationArtifact
        """

        try:

            logger.info(
                "Data Validation Started"
            )

            # -----------------------------------
            # Load Training Dataset
            # -----------------------------------

            train_df = pd.read_csv(
                self.ingestion_artifact.train_file_path
            )

            logger.info(
                f"Training dataset loaded "
                f"with shape {train_df.shape}"
            )

            # -----------------------------------
            # Load Schema
            # -----------------------------------

            schema = self.load_schema()

            # -----------------------------------
            # Expected Columns
            # -----------------------------------

            expected_columns = set(
                schema["columns"].keys()
            )

            # -----------------------------------
            # Actual Columns
            # -----------------------------------

            actual_columns = set(
                train_df.columns
            )

            # -----------------------------------
            # Column Validation
            # -----------------------------------

            validation_status = (
                expected_columns ==
                actual_columns
            )

            logger.info(
                f"Column Validation Status: "
                f"{validation_status}"
            )

            # -----------------------------------
            # Missing Values Check
            # -----------------------------------

            missing_values = (
                train_df
                .isnull()
                .sum()
                .sum()
            )

            logger.info(
                f"Missing Values Found: "
                f"{missing_values}"
            )

            # -----------------------------------
            # Duplicate Records Check
            # -----------------------------------

            duplicate_rows = (
                train_df
                .duplicated()
                .sum()
            )

            logger.info(
                f"Duplicate Rows Found: "
                f"{duplicate_rows}"
            )

            # -----------------------------------
            # Create Report Directory
            # -----------------------------------

            report_dir = os.path.dirname(
                self.config.validation_report_path
            )

            os.makedirs(
                report_dir,
                exist_ok=True
            )

            # -----------------------------------
            # Generate Validation Report
            # -----------------------------------

            report = f"""
=================================================
DATA VALIDATION REPORT
=================================================

Column Validation Status:
{validation_status}

Expected Columns:
{len(expected_columns)}

Actual Columns:
{len(actual_columns)}

Missing Values:
{missing_values}

Duplicate Rows:
{duplicate_rows}

=================================================
"""

            # -----------------------------------
            # Save Validation Report
            # -----------------------------------

            with open(
                self.config.validation_report_path,
                "w"
            ) as file:

                file.write(report)

            logger.info(
                "Validation report generated"
            )

            # -----------------------------------
            # Return Artifact
            # -----------------------------------

            return DataValidationArtifact(

                validation_status=
                validation_status,

                validation_report_path=
                self.config.validation_report_path
            )

        except Exception as e:

            logger.error(
                "Error occurred during validation"
            )

            raise CustomException(
                e,
                sys
            )