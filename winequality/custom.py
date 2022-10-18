import mlflow


def load_model(code_dir):
    model = mlflow.pyfunc.load_model(code_dir + 'MLmodel')
    return model

def score(data, model):
    predictions = model.predict(data)
    return predictions
