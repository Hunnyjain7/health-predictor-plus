import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(df, is_training=True):
    if is_training:
        X = df.drop('health_outcome', axis=1)
        y = df['health_outcome']
    else:
        X = df.copy()
        y = None

    numeric_features = ['age', 'height_cm', 'weight_kg', 'average_sleep_hours', 'daily_water_intake_litres',
                        'steps_count_per_day', 'daily_exercise_minutes', 'work_hours', 'systolic_pressure',
                        'diastolic_pressure', 'heart_rate_bpm', 'blood_sugar_levels_mg_dl']
    categorical_features = ['gender', 'medical_history', 'heredity_diseases', 'smoking_status', 'alcohol_consumption',
                            'physical_activity_level', 'diet_type', 'stress_level', 'current_medications',
                            'frequency_of_checkups', 'type_of_physical_activities']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return X, y, preprocessor
