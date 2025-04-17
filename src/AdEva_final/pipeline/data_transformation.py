import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

from AdEva_final.config.configuration import Configuration

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

class DataTransformation:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def create_windows(self, series: np.ndarray, window_size: int):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i : i + window_size])
            y.append(series[i + window_size])
        return np.array(X), np.array(y)

    def transform(self, train_path, test_path):
        # 1) Load ingested CSVs
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test_df  = pd.read_csv(test_path,  index_col=0, parse_dates=True)

        # 2) Pick first numeric column
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in train data!")
        if len(num_cols) > 1:
            logging.warning(f"Multiple numeric columns {list(num_cols)}, selecting '{num_cols[0]}'")
        col = num_cols[0]

        # 3) Coerce to numeric & drop NaNs
        train_series = pd.to_numeric(train_df[col], errors="coerce")
        test_series  = pd.to_numeric(test_df[col],  errors="coerce")

        n_train_bad = train_series.isna().sum()
        n_test_bad  = test_series.isna().sum()
        if n_train_bad or n_test_bad:
            logging.warning(
                f"Dropping {n_train_bad} bad rows in train and {n_test_bad} in test for column '{col}'"
            )

        train_series = train_series.dropna().values.reshape(-1, 1)
        test_series  = test_series.dropna().values.reshape(-1, 1)

        # 4) Scale
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_series).flatten()
        test_scaled  = scaler.transform(test_series).flatten()

        # 5) Window
        w = self.config.window_size
        X_train, y_train = self.create_windows(train_scaled, w)
        X_test,  y_test  = self.create_windows(test_scaled,  w)

        # 6) Save outputs
        os.makedirs(self.config.transformed_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.scaler_path), exist_ok=True)

        with open(self.config.scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        np.save(os.path.join(self.config.transformed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.config.transformed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.config.transformed_dir, "X_test.npy"),  X_test)
        np.save(os.path.join(self.config.transformed_dir, "y_test.npy"),  y_test)

        logging.info(f"Saved X/Y train & test in {self.config.transformed_dir}")
        logging.info(f"Saved scaler at {self.config.scaler_path}")

def main():
    cfg = Configuration()
    di_cfg = cfg.get_data_ingestion_config()
    dt_cfg = cfg.get_data_transformation_config()
    params = cfg.params

    train_path = os.path.join(di_cfg.ingested_dir, di_cfg.train_file)
    test_path  = os.path.join(di_cfg.ingested_dir, di_cfg.test_file)

    transformer = DataTransformation(dt_cfg, params)
    transformer.transform(train_path, test_path)

if __name__ == "__main__":
    main()