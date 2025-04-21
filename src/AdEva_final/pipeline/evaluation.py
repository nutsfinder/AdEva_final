import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from AdEva_final.config.configuration import Configuration

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# ——— Inlined Transformer architecture ———

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.head    = nn.Linear(d_model, forecast_horizon)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch, seq_len, feature_size)
        x = self.input_linear(src)         # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)            # add positional encoding
        x = x.permute(1, 0, 2)             # → (seq_len, batch, d_model)
        x = self.encoder(x)                # transformer encoder
        last = x[-1, :, :]                 # (batch, d_model)
        return self.head(last)             # (batch, forecast_horizon)

# ——— Evaluation pipeline ———

class Evaluation:
    def __init__(self, cfg: Configuration):
        self.cfg    = cfg
        self.ev_cfg = cfg.get_evaluation_config()
        self.ml_cfg = cfg.get_mlflow_config()
        self.mt_cfg = cfg.get_model_training_config()

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        mlflow.set_tracking_uri(self.ml_cfg.tracking_uri)
        mlflow.set_experiment(self.ml_cfg.experiment_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimeSeriesTransformer(
            feature_size      = X.shape[2],
            d_model            = self.cfg.params["d_model"],
            nhead              = self.cfg.params["nhead"],
            num_encoder_layers = self.cfg.params["num_encoder_layers"],
            dim_feedforward    = self.cfg.params["dim_feedforward"],
            dropout            = self.cfg.params["dropout"],
            forecast_horizon   = self.cfg.params["forecast_horizon"],
        ).to(device)

        # load the trained weights
        ckpt = os.path.join(self.mt_cfg.model_dir, self.mt_cfg.model_name)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        X_tensor = torch.from_numpy(X).to(device)
        with mlflow.start_run(nested=True):
            with torch.no_grad():
                preds = model(X_tensor).cpu().numpy()

            mae = mean_absolute_error(y, preds)
            mse = mean_squared_error(y, preds)
            rmse = math.sqrt(mse)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            logging.info(f"MAE={mae:.4f}, RMSE={rmse:.4f}")

        os.makedirs(self.ev_cfg.metrics_dir, exist_ok=True)
        report_path = os.path.join(self.ev_cfg.metrics_dir, self.ev_cfg.report_file)
        with open(report_path, "w") as f:
            json.dump({"MAE": mae, "RMSE": rmse}, f, indent=4)
        logging.info(f"Report saved → {report_path}")

if __name__ == "__main__":
    cfg    = Configuration()
    dt_cfg = cfg.get_data_transformation_config()

    X_test = np.load(os.path.join(dt_cfg.transformed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dt_cfg.transformed_dir, "y_test.npy"))

    Evaluation(cfg).evaluate(X_test, y_test)
