import os
import sys
import json

from src.logger.logger import logger

from src.exception.exception import (
    CustomException
)

from src.entity.config_entity import (
    ModelEvaluationConfig
)

from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)


class ModelEvaluation:
    """
    Model Evaluation Component

    Responsibilities
    ------------------------
    1. Read model performance
    2. Compare against threshold
    3. Generate evaluation report
    4. Decide model acceptance
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact
    ):

        self.config = config

        self.trainer_artifact = trainer_artifact

    def initiate_model_evaluation(self):

        try:

            logger.info(
                "Model evaluation started"
            )

            # =====================================
            # Acceptance Logic
            # =====================================

            is_model_accepted = (

                self.trainer_artifact.test_f1
                >=
                self.config.improvement_threshold
            )

            report = {

                "model_name":
                self.trainer_artifact.model_name,

                "train_f1":
                self.trainer_artifact.train_f1,

                "test_f1":
                self.trainer_artifact.test_f1,

                "precision":
                self.trainer_artifact.precision,

                "recall":
                self.trainer_artifact.recall,

                "roc_auc":
                self.trainer_artifact.roc_auc,

                "model_accepted":
                is_model_accepted
            }

            with open( self.config.evaluation_report_file_path, "w") as file:
                json.dump(report,file,indent=4)

            logger.info(
                "Evaluation report generated"
            )

            return ModelEvaluationArtifact(

                is_model_accepted=
                is_model_accepted,

                improvement_score=
                self.trainer_artifact.test_f1,

                evaluation_report_path=
                self.config
                .evaluation_report_file_path
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )