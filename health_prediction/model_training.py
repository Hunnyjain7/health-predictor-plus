from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

from data_preprocessing import preprocess_data
from utils import load_data, get_training_data


def train_model():
    training_data = get_training_data()
    df = load_data(training_data)
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit the preprocessor on your training data
    preprocessor.fit(X_train)
    

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, 'health_prediction/models/health_model.pkl')
    print("Model training completed and saved as 'health_model.pkl'.")


if __name__ == "__main__":
    train_model()
