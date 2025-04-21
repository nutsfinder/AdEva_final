import os
import pickle
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from AdEva_final.config.configuration import Configuration

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

class DataTransformation:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.params = cfg.params

    def create_sequences(self, data: np.ndarray, window_size: int, horizon: int):
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i: i + window_size, :])
            y.append(data[i + window_size: i + window_size + horizon, 0])
        return np.array(X), np.array(y)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Melt time slots into records and filter category 'GC'
        records = []
        for _, row in df.iterrows():
            if row.get('Consumption Category') != 'GC':
                continue
            date_val = row['date']
            cust = int(row['Customer'])
            code = row['Postcode']
            for col, val in row.items():
                if col in ['Customer','Postcode','Consumption Category','date','Generator Capacity','Row Quality']:
                    continue
                try:
                    cons = float(val)
                except Exception:
                    continue
                records.append({
                    'date': date_val,
                    'Customer': cust,
                    'Postcode': code,
                    'consumption': cons
                })
        out = pd.DataFrame(records)

        # Encode Customer
        if not out.empty:
            out['Customer'] = OrdinalEncoder().fit_transform(out[['Customer']]).astype(int)

        # Aggregate daily consumption
        grp = out.groupby(['date','Customer','Postcode'], as_index=False)['consumption'].mean()

        # Flag weekends
        grp['is_holiday_or_weekend'] = grp['date'].dt.day_name().isin(
            ['Saturday','Sunday']
        ).astype(int)

        # Season: 1=winter,2=spring,3=summer,4=autumn
        grp['saison'] = grp['date'].dt.month % 12 // 3 + 1

        return grp

    def transform(self, train_path: str, test_path: str):
        dt_cfg = self.cfg.get_data_transformation_config()
        w = dt_cfg.window_size
        h = self.params.get('forecast_horizon', 7)

        # Load ingested CSVs
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Ensure 'date' column exists and is datetime
        if 'date' in train_df.columns:
            train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')
        else:
            raise KeyError("'date' column not found in train data")
        if 'date' in test_df.columns:
            test_df['date'] = pd.to_datetime(test_df['date'], errors='coerce')
        else:
            raise KeyError("'date' column not found in test data")

        # Preprocess and aggregate
        train_grp = self._preprocess(train_df)
        test_grp = self._preprocess(test_df)

        # Normalize based on training data
        scaler = MinMaxScaler()
        if train_grp.empty:
            raise ValueError("No data after preprocessing; cannot proceed.")
        train_grp['consumption_daily_normalized'] = scaler.fit_transform(
            train_grp[['consumption']]
        )
        test_grp['consumption_daily_normalized'] = scaler.transform(
            test_grp[['consumption']]
        )

        # Build feature arrays
        features = ['consumption_daily_normalized', 'is_holiday_or_weekend', 'saison']
        X_train_vals = train_grp[features].to_numpy(dtype=np.float32)
        X_test_vals = test_grp[features].to_numpy(dtype=np.float32)

        # Create sequences
        X_train, y_train = self.create_sequences(X_train_vals, w, h)
        X_test, y_test = self.create_sequences(X_test_vals, w, h)

        # Save sequences and scaler
        os.makedirs(dt_cfg.transformed_dir, exist_ok=True)
        with open(dt_cfg.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        np.save(os.path.join(dt_cfg.transformed_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(dt_cfg.transformed_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(dt_cfg.transformed_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(dt_cfg.transformed_dir, 'y_test.npy'), y_test)

        logging.info(f"Saved transformed data to {dt_cfg.transformed_dir}")

if __name__ == '__main__':
    cfg = Configuration()
    di_cfg = cfg.get_data_ingestion_config()
    dt = DataTransformation(cfg)
    dt.transform(
        os.path.join(di_cfg.ingested_dir, di_cfg.train_file),
        os.path.join(di_cfg.ingested_dir, di_cfg.test_file)
    )