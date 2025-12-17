# starting with monitoring pipeline code

from src.components.drift_monitor import monitor_drift
from src.utils.drift_logger import log_drift

if __name__ == "__main__":

    TARGET_COL = "Churn"

    NUMERICAL_COLS = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    CATEGORICAL_COLS = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "customerID"
    ]

    monitor = monitor_drift(
        target_col=TARGET_COL,
        numerical_cols=NUMERICAL_COLS,
        categorical_cols=CATEGORICAL_COLS
    )

    result = monitor.run(
        reference_path="data/processed/reference.csv",
        current_path="data/processed/production_batch_01.csv",
        output_dir="artifacts/reports/latest"
    )

    print("DRIFT RESULT:", result)


    log_drift(
        history_path="artifacts/reports/drift_history.csv",
        drift_result=result
    )

    print("Drift monitoring + logging completed.")