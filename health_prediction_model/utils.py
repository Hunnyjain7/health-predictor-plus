import json

import joblib
import pandas as pd


def load_data(data):
    return pd.DataFrame(data)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_training_data(training_json_path="health_prediction_model/training_data.json"):
    # Save DataFrame to CSV
    csv_filename = 'health_prediction_model/health_data.csv'

    # Load CSV back into JSON
    df_loaded = pd.read_csv(csv_filename)
    json_data_loaded = df_loaded.to_dict(orient='list')
    # return read_json(training_json_path)
    return json_data_loaded


def read_config(config_path="health_prediction_model/config.json"):
    return read_json(config_path)


def load_model():
    config = read_config()
    return joblib.load((config["health_model_path"]))
