import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle
import traceback

imputer = SimpleImputer(strategy='mean')  # Initialize the imputer

def train_model():
    # Load the dataset
    data = pd.read_csv('loan_approval_data.csv')

    # Drop the loan_id column
    data = data.drop('loan_id', axis=1)

    # Preprocess the dependents column
    data['dependents'] = data['dependents'].replace('3+', 3)

    # Convert categorical columns to numeric values
    label_encoder = LabelEncoder()
    categorical_columns = ['gender', 'married', 'education', 'self_employed', 'property_area']
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Separate the target and independent features
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    # Encode categorical variables using one-hot encoding
    categorical_features = ['education', 'self_employed', 'property_area']
    X_encoded = pd.get_dummies(X, columns=categorical_features)

    # Save the feature names for later comparison
    with open('feature_names.pkl', 'wb') as file:
        pickle.dump(X_encoded.columns, file)

    # Handle missing values
    numeric_columns = X_encoded.select_dtypes(include='number').columns
    X_encoded[numeric_columns] = imputer.fit_transform(X_encoded[numeric_columns])

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def load_model():
    # Load the model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    return model

def load_feature_names():
    # Load the feature names from the pickle file
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)

    return feature_names


def preprocess_input_data(data):
    # Convert categorical columns to numeric values
    label_encoder = LabelEncoder()
    categorical_columns = ['gender', 'married']
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Renaming columns to match trained model's features
    data = data.rename(columns={'applicant_income': 'applicantincome',
                                'coapplicant_income': 'coapplicantincome',
                                'loan_amount': 'loanamount'})

    # One-hot encoding for columns 'education', 'self_employed' and 'property_area'
    data = pd.get_dummies(data, columns=['education', 'self_employed', 'property_area'])

    # Loading trained features
    trained_features = load_feature_names()

    # Adding missing columns with default value of zero
    for feature in trained_features:
        if feature not in data.columns:
            data[feature] = 0

    # Ordering columns to match the order of the trained model's features
    data = data[trained_features]

    return data


def compare_features():
    try:
        # Load the feature names from the trained model
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

        # Preprocess the input data
        data_encoded = preprocess_input_data(data)
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
if __name__ == '__main__':
    # Train the model
    train_model()
    compare_features()
