import mlflow
import pandas as pd

def load_model(code_dir):
    print(code_dir)
    model = mlflow.pyfunc.load_model(code_dir)
    return model


def score(data, model, **kwargs):
    predictions = model.predict(data)
    return predictions
