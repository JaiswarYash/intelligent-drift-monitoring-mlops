import os
import tempfile
import pandas as pd
import mlflow
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class CatBoostTrainer:
    def __init__(self, target_col, categorical_cols, random_state=42):
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.random_state = random_state

    def train(self, data_path):
        df = pd.read_csv(data_path)

        df[self.target_col] = df[self.target_col].map({"Yes": 1, "No": 0})

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X = df.drop(columns=[self.target_col, "customerID"])


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cat_features = [X.columns.get_loc(col) for col in self.categorical_cols]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric='F1',
            loss_function='Logloss',
            random_seed=self.random_state,
            early_stopping_rounds=50,
            verbose=100
        )

        mlflow.set_experiment("churn_catboost_model")

        with mlflow.start_run():
            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                eval_set=(X_test, y_test)
            )

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_param("model_type", "CatBoost")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            tmp_dir = tempfile.mkdtemp()
            model_path = os.path.join(tmp_dir, "catboost_model.cbm")

            model.save_model(model_path)

            mlflow.log_artifact(model_path, artifact_path="catboost_model")

        return acc, f1
