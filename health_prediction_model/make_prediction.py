from health_prediction_model.data_preprocessing import preprocess_data
from health_prediction_model.utils import load_data, load_model


def make_prediction(data):
    numeric_features = [
        'age', 'height_cm', 'weight_kg', 'average_sleep_hours', 'daily_water_intake_litres', 'steps_count_per_day',
        'daily_exercise_minutes', 'work_hours', 'systolic_pressure', 'diastolic_pressure', 'heart_rate_bpm',
        'blood_sugar_levels_mg_dl'
    ]
    categorical_features = [
        'gender', 'medical_history', 'heredity_diseases', 'smoking_status', 'alcohol_consumption',
        'physical_activity_level', 'diet_type', 'stress_level', 'current_medications', 'frequency_of_checkups',
        'type_of_physical_activities'
    ]

    new_data = {}
    for key, value in data.items():
        if key in numeric_features + categorical_features:
            if isinstance(value, list):
                value = ", ".join(value)
            new_data[key] = value

    input_df = load_data([new_data])
    model = load_model()
    X, _, _ = preprocess_data(input_df, is_training=False)
    X_preprocessed = model.named_steps['preprocessor'].transform(X)  # Use preprocessor from the trained model pipeline
    prediction = model.named_steps['classifier'].predict(X_preprocessed)
    return prediction.flat[0]
