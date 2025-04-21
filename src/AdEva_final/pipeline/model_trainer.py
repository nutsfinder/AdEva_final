import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader
from AdEva_final.config.configuration import Configuration

# ——— Transformer architecture inlined ———

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]

class TimeSeriesTransformer(nn.Module):
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
        self.pos_encoder  = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.head    = nn.Linear(d_model, forecast_horizon)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch, seq_len, feature_size)
        x = self.input_linear(src)         # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)            # + positional
        x = x.permute(1, 0, 2)             # → (seq_len, batch, d_model)
        x = self.encoder(x)                # same shape
        last = x[-1, :, :]                 # (batch, d_model)
        return self.head(last)             # (batch, forecast_horizon)

# ——— Training pipeline ———

class ModelTrainer:
    def __init__(self, cfg: Configuration):
        self.cfg    = cfg
        self.params = cfg.params
        self.mt_cfg = cfg.get_model_training_config()
        self.ml_cfg = cfg.get_mlflow_config()

    def train(self, X: np.ndarray, y: np.ndarray):
        # MLflow setup
        mlflow.set_tracking_uri(self.ml_cfg.tracking_uri)
        mlflow.set_experiment(self.ml_cfg.experiment_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimeSeriesTransformer(
            feature_size      = X.shape[2],
            d_model            = self.params["d_model"],
            nhead              = self.params["nhead"],
            num_encoder_layers = self.params["num_encoder_layers"],
            dim_feedforward    = self.params["dim_feedforward"],
            dropout            = self.params["dropout"],
            forecast_horizon   = self.params["forecast_horizon"],
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.mt_cfg.learning_rate)

        ds     = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=self.mt_cfg.batch_size, shuffle=True)

        with mlflow.start_run():
            # Log hyperparameters
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

            for epoch in range(self.mt_cfg.epochs):
                model.train()
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1}/{self.mt_cfg.epochs} – Loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Save model
            os.makedirs(self.mt_cfg.model_dir, exist_ok=True)
            path = os.path.join(self.mt_cfg.model_dir, self.mt_cfg.model_name)
            torch.save(model.state_dict(), path)
            mlflow.pytorch.log_model(model, artifact_path="model")
            print(f"Model saved → {path}")

if __name__ == "__main__":
    cfg    = Configuration()
    dt_cfg = cfg.get_data_transformation_config()
    X_train = np.load(os.path.join(dt_cfg.transformed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dt_cfg.transformed_dir, "y_train.npy"))

    trainer = ModelTrainer(cfg)
    trainer.train(X_train, y_train)
