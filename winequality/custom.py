import mlflow


def load_model(code_dir):
    mlflow.pyfunc.load_model(code_dir)


def score(data, model):
    return model.predict(data)
