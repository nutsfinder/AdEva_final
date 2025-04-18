import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader

from AdEva_final.config.configuration import Configuration

class SimpleForecastModel(nn.Module):
    def __init__(self, window_size: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class ModelTrainer:
    def __init__(self, cfg: Configuration, params: dict):
        self.cfg = cfg
        self.params = params

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # fetch configs
        mt_cfg = self.cfg.get_model_training_config()
        ml_cfg = self.cfg.get_mlflow_config()

        # MLflow setup
        mlflow.set_tracking_uri(ml_cfg.tracking_uri)
        mlflow.set_experiment(ml_cfg.experiment_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleForecastModel(window_size=X_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.params["learning_rate"])

        dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        loader  = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            print(f"📋 Active MLflow run_id: {run_id}")
            # log hyperparameters
            mlflow.log_params({
                "window_size": X_train.shape[1],
                "batch_size":    self.params["batch_size"],
                "epochs":        self.params["epochs"],
                "learning_rate": self.params["learning_rate"]
            })

            for epoch in range(self.params["epochs"]):
                model.train()
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss  = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1}/{self.params['epochs']} – Loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # save & log model
            os.makedirs(mt_cfg.model_dir, exist_ok=True)
            model_path = os.path.join(mt_cfg.model_dir, mt_cfg.model_name)
            torch.save(model.state_dict(), model_path)
            mlflow.pytorch.log_model(model, artifact_path="model")
            print(f"Model checkpoint saved to {model_path}")

def main():
    cfg   = Configuration()
    dt_cfg = cfg.get_data_transformation_config()
    params = cfg.params

    X_train = np.load(os.path.join(dt_cfg.transformed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dt_cfg.transformed_dir, "y_train.npy"))

    trainer = ModelTrainer(cfg, params)
    trainer.train(X_train, y_train)

if __name__ == "__main__":
    main()
