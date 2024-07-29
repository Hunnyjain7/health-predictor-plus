from health_prediction.data_preprocessing import preprocess_data
from health_prediction.utils import load_data, load_model


def make_prediction(data):
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = ", ".join(value)

    input_df = load_data([data])
    model = load_model()
    X, _, _ = preprocess_data(input_df, is_training=False)
    X_preprocessed = model.named_steps['preprocessor'].transform(X)  # Use preprocessor from the trained model pipeline
    prediction = model.named_steps['classifier'].predict(X_preprocessed)
    return prediction.flat[0]
