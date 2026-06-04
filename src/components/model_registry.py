import os
import sys
import json
import shutil

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)

from src.entity.config_entity import (
    ModelRegistryConfig
)

from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelRegistryArtifact
)


class ModelRegistry:

    def __init__(
        self,
        config: ModelRegistryConfig,
        trainer_artifact: ModelTrainerArtifact
    ):

        self.config = config

        self.trainer_artifact = trainer_artifact

    def register_model(self):

        try:

            # =====================================
            # FIRST MODEL
            # =====================================

            if not os.path.exists(self.config.registry_file_path):
                version = 1
                production = True

            else:

                with open(self.config.registry_file_path,"r") as file:
                    registry = json.load(file)

                version = (registry["latest_version"] + 1)
                production = False

            model_name = ( f"model_v{version}.pkl")

            destination_path = os.path.join( self.config.registry_dir,
                model_name)

            shutil.copy(

                self.trainer_artifact
                .trained_model_file_path,

                destination_path
            )

            registry_data = {

                "latest_version":
                version,

                "production_model":
                model_name,

                "production_metrics": {

                    "accuracy": 
                    self.trainer_artifact.train_f1,

                    "precision":
                    self.trainer_artifact.precision,

                    "recall":
                    self.trainer_artifact.recall,

                    "f1":
                    self.trainer_artifact.test_f1,

                    "roc_auc":
                    self.trainer_artifact.roc_auc
                        }

            }

            with open( self.config.registry_file_path, "w") as file:

                json.dump(registry_data,file,indent=4)

            logger.info(f"Model v{version} registered")

            return ModelRegistryArtifact(

                registered_model_path=
                destination_path,

                model_version=
                version,

                is_production_model=
                production
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )