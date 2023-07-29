import pandas as pd
import pickle
import traceback

def load_feature_names():
    # Load the feature names from the pickle file
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)

    return feature_names

def preprocess_input_data_prediction(data, train_features):
    # Preprocess the input data for prediction
    # ... your preprocessing logic here ...
    data_encoded = data  # Replace this line with your actual preprocessing code

    return data_encoded

try:
    # Load the trained feature names
    train_features = load_feature_names()

    # Generate a new prediction dataset
    data = pd.DataFrame({
        'gender': ['Male'],
        'married': ['Yes'],
        'dependents': [1],
        'education': ['Graduate'],
        'self_employed': ['No'],
        'applicant_income': [8000],
        'coapplicant_income': [2000],
        'loan_amount': [130],
        'loan_amount_term': [360],
        'credit_history': [1],
        'property_area': ['Urban']
    })

    # Preprocess the input data for prediction
    data_encoded = preprocess_input_data_prediction(data, train_features)
    predict_features = data_encoded.columns.tolist()

    # Compare the feature names
    missing_features = set(train_features) - set(predict_features)
    extra_features = set(predict_features) - set(train_features)

    with open("missing_features.txt", "w") as file:
        for feature in missing_features:
            file.write(f"{feature}\n")

    with open("extra_features.txt", "w") as file:
        for feature in extra_features:
            file.write(f"{feature}\n")

except Exception as e:
    with open("error_log.txt", "w") as file:
        file.write(f"Error: {str(e)}\n")
        file.write(traceback.format_exc())
