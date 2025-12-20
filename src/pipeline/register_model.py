import mlflow
import os
from src.utils.catboost_pyfunc import CatBoostPyFuncModel


def main():
    mlflow.set_experiment("CatBoost_Churn_Classification")

    # Get latest run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("CatBoost_Churn_Classification")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    run = runs[0]
    run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/catboost_model/catboost_model.cbm"

    mlflow.pyfunc.log_model(
        artifact_path="registered_catboost_model",
        python_model=CatBoostPyFuncModel(),
        artifacts={"model_path": model_uri},
        registered_model_name="ChurnCatBoostModel"
    )

    print("Model registered successfully as ChurnCatBoostModel")


if __name__ == "__main__":
    main()
