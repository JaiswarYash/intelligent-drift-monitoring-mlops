# Register the model using MLflow pyfunc wrapper

import mlflow.pyfunc
import pandas as pd
from catboost import CatBoostClassifier

class CatBoostPyFuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = CatBoostClassifier()
        self.model.load_model(context.artifacts["model_path"])
    
    def predict(self, context, model_input, params = None):
        if isinstance(model_input, pd.DataFrame):
            return self.model.predict(model_input)
        else:
            raise ValueError("Input must be a pandas DataFrame")
        