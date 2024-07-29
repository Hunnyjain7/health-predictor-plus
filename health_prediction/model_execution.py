import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data

from utils import load_data, get_training_data, load_model


def evaluate_model(data):
    df = load_data(data)
    X, y, preprocessor = preprocess_data(df)

    model = load_model()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    training_data = get_training_data()
    evaluate_model(training_data)
