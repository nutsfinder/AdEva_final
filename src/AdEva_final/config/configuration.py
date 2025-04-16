import os, yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_dir: str
    ingested_dir: str
    train_file: str
    test_file: str

# similarly for DataTransformationConfig, ModelTrainingConfig, EvaluationConfig...

class Configuration:
    def __init__(self, config_path: str = "config/config.yaml", params_path: str = "params.yaml"):
        self.config = self._read_yaml(config_path)
        self.params = self._read_yaml(params_path)

    def _read_yaml(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        return DataIngestionConfig(**cfg)

    # add getters for other configs...
