import json

import joblib
import pandas as pd


def load_data(data):
    return pd.DataFrame(data)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_training_data(training_json_path="health_prediction_model/training_data.json"):
    return read_json(training_json_path)


def read_config(config_path="health_prediction_model/config.json"):
    return read_json(config_path)


def load_model():
    config = read_config()
    return joblib.load((config["health_model_path"]))
