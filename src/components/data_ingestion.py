import os
import sys

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)

from src.entity.artifact_entity import (
    DataIngestionArtifact
)

from src.entity.config_entity import (
    DataIngestionConfig
)


class DataIngestion:

    """
    Responsible for:

    1. Reading raw data
    2. Creating train/test split
    3. Saving datasets
    """

    def __init__(self, config: DataIngestionConfig ):

        self.config = config

    def initiate_data_ingestion(self):

        try:

            logger.info("Data ingestion started")

            # --------------------------------
            # Read Dataset
            # --------------------------------

            df = pd.read_csv(self.config.raw_data_path)

            logger.info(
                f"Dataset loaded. "
                f"Shape: {df.shape}"
            )

            # --------------------------------
            # Train Test Split
            # --------------------------------

            train_set, test_set = (
                train_test_split(
                    df,
                    test_size=0.2,
                    random_state=42,
                    stratify=df["Class"]
                )
            )

            logger.info(
                "Train Test Split completed"
            )

            # --------------------------------
            # Save Train Dataset
            # --------------------------------

            train_set.to_csv( self.config.train_data_path, index=False)

            # --------------------------------
            # Save Test Dataset
            # --------------------------------

            test_set.to_csv(self.config.test_data_path,index=False)

            logger.info( "Datasets saved successfully")

            return (
                DataIngestionArtifact(
                    train_file_path=
                    self.config.train_data_path,

                    test_file_path=
                    self.config.test_data_path
                )
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )