"""Minimal smoke tests. Expand with real unit tests per component."""
import importlib


def test_imports():
    modules = [
        "src.logger",
        "src.exception",
        "src.utils",
        "src.constants",
        "src.components.data_ingestion",
        "src.components.data_validation",
        "src.components.data_transformation",
        "src.components.model_trainer",
        "src.components.model_evaluation",
        "src.pipeline.training_pipeline",
        "src.pipeline.prediction_pipeline",
    ]
    for m in modules:
        importlib.import_module(m)
