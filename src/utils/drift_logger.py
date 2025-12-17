import os
import pandas as pd


def log_drift(history_path, drift_result):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    df = pd.DataFrame([drift_result])

    if os.path.exists(history_path):
        df_existing = pd.read_csv(history_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(history_path, index=False)
