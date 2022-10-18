import mlflow
import pandas as pd
import numpy as np

def load_model(code_dir):
    model = mlflow.pyfunc.load_model(code_dir)
    return model

def read_input_data(input_binary_data):
    import io
    df_scoring = pd.read_csv(io.BytesIO(input_binary_data))
    df_scoring['is_red'] = df_scoring['is_red'].astype(np.int64)
    return df_scoring

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
