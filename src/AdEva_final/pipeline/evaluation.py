import os, json, torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from AdEva_final.config.configuration import Configuration

class Evaluation:
    def __init__(self, config):
        self.config = config

    def evaluate(self, model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            preds = model(torch.Tensor(X_test)).cpu().numpy().flatten()

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        os.makedirs(self.config.metrics_dir, exist_ok=True)
        metrics = {"MAE": mae, "RMSE": rmse}
        with open(os.path.join(self.config.metrics_dir, self.config.report_file), "w") as f:
            json.dump(metrics, f, indent=4)

        print("Evaluation:", metrics)
        return metrics
