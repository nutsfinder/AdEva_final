import os, pandas as pd
from sklearn.model_selection import train_test_split
from AdEva_final.config.configuration import Configuration

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def ingest(self):
        raw_dir = self.config.raw_data_dir
        all_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
        df_list = [pd.read_csv(f, parse_dates=True, index_col=0) for f in all_files]
        data = pd.concat(df_list).sort_index()
        
        train, test = train_test_split(data, shuffle=False, test_size=0.2)
        os.makedirs(self.config.ingested_dir, exist_ok=True)
        train.to_csv(os.path.join(self.config.ingested_dir, self.config.train_file))
        test.to_csv(os.path.join(self.config.ingested_dir, self.config.test_file))
        return os.path.join(self.config.ingested_dir, self.config.train_file), \
               os.path.join(self.config.ingested_dir, self.config.test_file)
