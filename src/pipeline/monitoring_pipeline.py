import mlflow
from src.components.drift_monitor import monitor_drift
from src.utils.drift_logger import log_drift
from src.utils.alerting import should_alert


def get_latest_model_run_id(experiment_name: str) -> str:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return "unknown"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        return "unknown"

    return runs[0].info.run_id


if __name__ == "__main__":

    TARGET_COL = "Churn"

    NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    CATEGORICAL_COLS = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
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

    # attach model metadata
    result["model_run_id"] = get_latest_model_run_id(
        "CatBoost_Churn_Classification"
    )

    log_drift(
        history_path="artifacts/reports/drift_history.csv",
        drift_result=result
    )

    alert, message = should_alert(result)
    if alert:
        print("ALERT TRIGGERED:", message)
    else:
        print("No alert triggered.")

    print("Monitoring run completed.")
