import math
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from AdEva_final.config.configuration import Configuration

# ──────────────────────────────────────────────────────────────────────
# Positional encoding & transformer model
# ──────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position encodings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Lightweight Transformer encoder for n-step forecasting."""

    def __init__(
        self,
        feature_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        forecast_horizon: int = 7,
    ):
        super().__init__()
        self.input_linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.head = nn.Linear(d_model, forecast_horizon)

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # (batch, seq_len, feature_size)
        x = self.input_linear(src)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)   # (seq_len, batch, d_model)
        x = self.encoder(x)
        return self.head(x[-1])  # use last time step


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

class ModelTrainer:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.params = cfg.params
        self.mt_cfg = cfg.get_model_training_config()
        self.ml_cfg = cfg.get_mlflow_config()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # MLflow setup
        mlflow.set_tracking_uri(self.ml_cfg.tracking_uri)
        mlflow.set_experiment(self.ml_cfg.experiment_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TimeSeriesTransformer(
            feature_size=X.shape[2],
            d_model=self.params["d_model"],
            nhead=self.params["nhead"],
            num_encoder_layers=self.params["num_encoder_layers"],
            dim_feedforward=self.params["dim_feedforward"],
            dropout=self.params["dropout"],
            forecast_horizon=self.params["forecast_horizon"],
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.mt_cfg.learning_rate)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=self.mt_cfg.batch_size, shuffle=True)

        # output path
        Path(self.mt_cfg.model_dir).mkdir(parents=True, exist_ok=True)
        pt_path = Path(self.mt_cfg.model_dir) / f"{self.mt_cfg.model_name}.pt"

        with mlflow.start_run():
            # log hyper-parameters
            mlflow.log_params({
                "window_size":        self.cfg.get_data_transformation_config().window_size,
                "forecast_horizon":   self.params["forecast_horizon"],
                "batch_size":         self.mt_cfg.batch_size,
                "epochs":             self.mt_cfg.epochs,
                "learning_rate":      self.mt_cfg.learning_rate,
                "d_model":            self.params["d_model"],
                "nhead":              self.params["nhead"],
                "num_encoder_layers": self.params["num_encoder_layers"],
                "dim_feedforward":    self.params["dim_feedforward"],
                "dropout":            self.params["dropout"],
            })

            # training loop
            for epoch in range(self.mt_cfg.epochs):
                model.train()
                running_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(loader)
                print(f"Epoch {epoch+1}/{self.mt_cfg.epochs} – loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # save & log
            torch.save(model.state_dict(), pt_path)
            mlflow.pytorch.log_model(model, artifact_path="pytorch-model")
            mlflow.log_artifact(str(pt_path), artifact_path="state-dict")
            print(f"Model saved → {pt_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Configuration()
    dt_cfg = cfg.get_data_transformation_config()

    X_train = np.load(Path(dt_cfg.transformed_dir) / "X_train.npy")
    y_train = np.load(Path(dt_cfg.transformed_dir) / "y_train.npy")

    ModelTrainer(cfg).train(X_train, y_train)