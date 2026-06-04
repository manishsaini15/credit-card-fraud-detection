def test_fastapi_import():
    from src.deployment.app import app
    assert app is not None


def test_prediction_pipeline_import():
    from src.deployment.prediction import PredictionPipeline
    assert PredictionPipeline is not None