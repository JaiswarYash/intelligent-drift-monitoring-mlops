# let's go
import os
import pandas as pd
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetSummaryMetric, DatasetMissingValuesMetric
from evidently.pipeline.column_mapping import ColumnMapping

class monitor_drift:
    def __init__(self, target_col, numerical_cols, categorical_cols):
        self.column_mapping = ColumnMapping(
            target=target_col,
            numerical_features=numerical_cols,
            categorical_features=categorical_cols
        )
    
    def run(self, reference_path, current_path, output_dir):
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)

        reference_df["Churn"] = reference_df["Churn"].map({"Yes": 1, "No": 0})
        current_df["Churn"] = current_df["Churn"].map({"Yes": 1, "No": 0})

        os.makedirs(output_dir, exist_ok=True)

        # Drift Report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=self.column_mapping
        )

        drift_path = os.path.join(output_dir, "drift_report.html")
        drift_report.save_html(drift_path)

        # Extract drift results
        drift_result = drift_report.as_dict()

        metric_result = drift_result["metrics"][0]["result"]

        dataset_drift = metric_result.get("dataset_drift", False)
        drifted_features = []

        columns_info = metric_result.get("columns", {})
        for col, stats in columns_info.items():
            if stats.get("drift_detected"):
                drifted_features.append(col)
                
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_drift": dataset_drift,
            "num_drifted_features": len(drifted_features),
            "drifted_features": drifted_features
            }