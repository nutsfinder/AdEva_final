import os
import json
import numpy as np
import torch
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

from AdEva_final.config.configuration import Configuration
from AdEva_final.pipeline.model_trainer import SimpleForecastModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

class Evaluation:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        ev_cfg = self.cfg.get_evaluation_config()
        ml_cfg = self.cfg.get_mlflow_config()
        mt_cfg = self.cfg.get_model_training_config()

        mlflow.set_tracking_uri(ml_cfg.tracking_uri)
        mlflow.set_experiment(ml_cfg.experiment_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleForecastModel(window_size=self.cfg.params["window_size"]).to(device)

        # load checkpoint
        ckpt = os.path.join(mt_cfg.model_dir, mt_cfg.model_name)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        X_tensor = torch.Tensor(X_test).to(device)
        with mlflow.start_run(nested=True):
            with torch.no_grad():
                preds = model(X_tensor).cpu().numpy().flatten()

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            logging.info(f"Logged metrics to MLflow • MAE={mae:.4f}, RMSE={rmse:.4f}")

        # also write JSON
        os.makedirs(ev_cfg.metrics_dir, exist_ok=True)
        report_path = os.path.join(ev_cfg.metrics_dir, ev_cfg.report_file)
        with open(report_path, "w") as f:
            json.dump({"MAE": mae, "RMSE": rmse}, f, indent=4)
        logging.info(f"Saved metrics JSON to {report_path}")

def main():
    cfg   = Configuration()
    dt_cfg = cfg.get_data_transformation_config()
    ev    = Evaluation(cfg)

    X_test = np.load(os.path.join(dt_cfg.transformed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dt_cfg.transformed_dir, "y_test.npy"))

    ev.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
