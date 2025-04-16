import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from AdEva_final.config.configuration import Configuration

class DataTransformation:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def create_windows(self, series: np.ndarray, window_size: int):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
        return np.array(X), np.array(y)

    def transform(self, train_path, test_path):
        train = pd.read_csv(train_path, index_col=0)
        test  = pd.read_csv(test_path,  index_col=0)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train.values.reshape(-1,1)).flatten()
        test_scaled  = scaler.transform(test.values.reshape(-1,1)).flatten()

        w = self.params["window_size"]
        X_train, y_train = self.create_windows(train_scaled, w)
        X_test, y_test   = self.create_windows(test_scaled,  w)

        os.makedirs(self.config.transformed_dir, exist_ok=True)
        with open(self.config.scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        np.save(os.path.join(self.config.transformed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.config.transformed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.config.transformed_dir, "X_test.npy"),  X_test)
        np.save(os.path.join(self.config.transformed_dir, "y_test.npy"),  y_test)

        return X_train, y_train, X_test, y_test
