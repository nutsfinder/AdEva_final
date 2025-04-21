import os
import glob
import logging
import pandas as pd
from AdEva_final.config.configuration import Configuration

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

class DataIngestion:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def ingest_data(self):
        di = self.cfg.get_data_ingestion_config()
        raw_dir      = di.raw_data_dir
        ingested_dir = di.ingested_dir
        train_file   = di.train_file
        test_file    = di.test_file

        os.makedirs(ingested_dir, exist_ok=True)

        # Load all CSVs from raw_dir, using second row as header
        pattern = os.path.join(raw_dir, "*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")

        df_list = []
        for fp in files:
            # Read with header=1 to skip metadata row and use true column names
            df = pd.read_csv(fp, header=1)
            df_list.append(df)
        df_all = pd.concat(df_list, ignore_index=True)

        # Ensure 'date' column is treated as string
        if 'date' not in df_all.columns and df_all.columns[0].lower() in ['date', 'Date']:
            df_all.rename(columns={df_all.columns[0]: 'date'}, inplace=True)

        # Split 80/20 chronologically if date exists
        if 'date' in df_all.columns:
            df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce')
            df_all.sort_values('date', inplace=True)
            split_idx = int(len(df_all) * 0.8)
            train_df = df_all.iloc[:split_idx]
            test_df  = df_all.iloc[split_idx:]
        else:
            # fallback random split
            split_idx = int(len(df_all) * 0.8)
            train_df = df_all.iloc[:split_idx]
            test_df  = df_all.iloc[split_idx:]

        # Write out ingested CSVs (including 'date' column)
        train_path = os.path.join(ingested_dir, train_file)
        test_path  = os.path.join(ingested_dir, test_file)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        logging.info(f"Ingested data → {train_path}, {test_path}")

if __name__ == "__main__":
    cfg = Configuration()
    DataIngestion(cfg).ingest_data()