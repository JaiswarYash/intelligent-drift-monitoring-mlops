import mlflow
from src.components.model_trainer import CatBoostTrainer


def main():
    DATA_PATH = "data/processed/reference.csv"

    TARGET_COL = "Churn"

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
        "PaymentMethod"
    ]

    mlflow.set_experiment("CatBoost_Churn_Classification_v2")


    trainer = CatBoostTrainer(
        target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        random_state=42,
    )

    acc, f1 = trainer.train(DATA_PATH)

    print(f"Training completed | Accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
