import mlflow
import pandas as pd

def load_model(code_dir):
    model = mlflow.pyfunc.load_model(code_dir)
    return model

# for binary classification use case:
def score(data, model, **kwargs):
    predictions = model.predict(data)
    df_predictions = pd.DataFrame(data=predictions, columns=['True'])
    df_predictions['False'] = 1-df_predictions['True']
    return df_predictions


# # for regression use case:
# def score(data, model, **kwargs):
#     predictions = model.predict(data)
#     df_predictions = pd.DataFrame(data=predictions, columns=['Predictions'])
#     return df_predictions
