import os, yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_dir: str
    ingested_dir: str
    train_file: str
    test_file: str

@dataclass
class DataTransformationConfig:
    transformed_dir: str
    scaler_path: str
    window_size: int

@dataclass
class ModelTrainingConfig:
    model_dir: str
    model_name: str
    batch_size: int
    epochs: int
    learning_rate: float

@dataclass
class EvaluationConfig:
    metrics_dir: str
    report_file: str
    
@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str

class Configuration:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        params_path: str = "params.yaml"
    ):
        self.config = self._read_yaml(config_path)
        self.params = self._read_yaml(params_path)

    def _read_yaml(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        return DataIngestionConfig(**cfg)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config["data_transformation"]
        return DataTransformationConfig(**cfg)

    def get_model_training_config(self) -> ModelTrainingConfig:
        cfg = self.config["model_training"]
        return ModelTrainingConfig(**cfg)

    def get_evaluation_config(self) -> EvaluationConfig:
        cfg = self.config["evaluation"]
        return EvaluationConfig(**cfg)
    
    def get_mlflow_config(self) -> MLflowConfig:
        cfg = self.config["mlflow"]
        return MLflowConfig(**cfg)