# Health Prediction Model

This project uses machine learning to predict health outcomes based on various health metrics and lifestyle factors.

## Overview of ML Implementation

- **Objective and Dataset**:
  Aim to predict health outcomes using a dataset comprising demographic, lifestyle, and medical history features.
  Data preprocessing involves handling missing values, scaling numerical features, and encoding categorical features.

- **Preprocessing with ColumnTransformer**:
  Utilizes StandardScaler for numeric features to standardize data.
  Applies OneHotEncoder to convert categorical variables into a binary format.

- **Model Selection and Pipeline**:
  Algorithm: Random Forest Classifier, chosen for its robustness and ability to handle mixed data types.
  A pipeline integrates the preprocessing and model training steps, ensuring consistency across datasets.

- **Model Evaluation**:
  The model achieved an accuracy of 75%, with strong performance in predicting "average" , “good” & “poor ” outcomes.

- **Model Scores**:

  **Accuracy: 75%** \
  **Precision: 56%** \
  **Recall: 75%** \
  **F1 Score: 64%**

- **Results and Future Directions**:
  Achieved balanced performance across evaluation metrics, with insights into important predictive features.
  Future work includes exploring other algorithms, tuning model hyperparameters, and expanding the feature set.

## Directory Structure

- `data_preprocessing.py`: Contains functions for loading and preprocessing the data.
- `model_training.py`: Script to train the model and save it as `health_model.pkl`.
- `model_execution.py`: Script to load the saved model and evaluate its performance.

## How to Use

1. **Train the Model**:
   Run the `model_training.py` script to train the model and save it.
   ```sh
   python model_training.py
   ```
