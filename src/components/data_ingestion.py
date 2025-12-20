import os
import pandas as pd


class DataIngestion:
    def __init__(self, raw_data_path, processed_dir):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir

    def load_data(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")

        df = pd.read_csv(self.raw_data_path)
        return df

    def basic_validation(self, df):
        if df.empty:
            raise ValueError("Dataset is empty")

        if "Churn" not in df.columns:
            raise ValueError("Target column 'Churn' not found")

        # Fix TotalCharges datatype (critical for this dataset)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Drop rows with missing critical values
        df = df.dropna().reset_index(drop=True)

        return df

    def split_reference_production(self, df, reference_ratio=0.7, random_state=42):
        reference_df = df.sample(frac=reference_ratio, random_state=random_state)
        production_df = df.drop(reference_df.index)

        return reference_df, production_df

    def save(self, reference_df, production_df):
        os.makedirs(self.processed_dir, exist_ok=True)

        reference_path = os.path.join(self.processed_dir, "reference.csv")
        production_path = os.path.join(self.processed_dir, "production_batch_01.csv")

        reference_df.to_csv(reference_path, index=False)
        production_df.to_csv(production_path, index=False)

        return reference_path, production_path

    def run(self):
        df = self.load_data()
        df = self.basic_validation(df)
        reference_df, production_df = self.split_reference_production(df)
        return self.save(reference_df, production_df)
