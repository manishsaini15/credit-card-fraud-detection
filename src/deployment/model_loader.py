import os
import json
from src.utils.common import load_object
from src.logger.logger import logger


class ModelLoader:
    """
    Loads production model from registry
    and loads preprocessing pipeline.
    """

    def __init__(self):

        self.registry_file_path = os.path.join(
            "artifacts",
            "model_registry",
            "registry.json"
        )

        self.registry_dir = os.path.join(
            "artifacts",
            "model_registry"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "data_transformation",
            "preprocessor.pkl"
        )

    def load_production_model(self):

        try:
            if not os.path.exists(self.registry_file_path):
                raise Exception("Registry file not found")

            with open(self.registry_file_path, "r") as file:
                registry = json.load(file)

            model_file = registry["production_model"]

            model_path = os.path.join(
                self.registry_dir,
                model_file
            )

            logger.info(f"Loading model: {model_file}")

            model = load_object(model_path)

            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def load_preprocessor(self):

        try:
            logger.info("Loading preprocessor")

            preprocessor = load_object(self.preprocessor_path)

            return preprocessor

        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise e

    def load_all(self):

        model = self.load_production_model()
        preprocessor = self.load_preprocessor()

        return model, preprocessor