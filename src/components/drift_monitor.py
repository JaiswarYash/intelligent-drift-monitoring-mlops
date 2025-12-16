# let's go
import os
import pandas as pd
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

        # Encode target column for consistency
        reference_df["Churn"] = reference_df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
        current_df["Churn"] = current_df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

        os.makedirs(output_dir, exist_ok=True)

        quality_report = Report(metrics=[
            DatasetSummaryMetric(),
            DatasetMissingValuesMetric()
        ])

        quality_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=self.column_mapping
        )

        quality_report.save_html(os.path.join(output_dir, "data_quality_report.html"))

        # Drift Report
        drift_report = Report(metrics=[DataDriftPreset()])

        drift_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=self.column_mapping
        )

        drift_report.save_html(
            os.path.join(output_dir, "drift_report.html")
        )