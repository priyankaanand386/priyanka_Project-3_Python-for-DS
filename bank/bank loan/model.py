import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pickle
import traceback

def preprocess_input_data(data):
    # Separate the target and independent features
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    return X, y

def train_model():
    # Load the dataset
    data = pd.read_csv('loan_approval_data_numeric.csv')

    # Preprocess the input data
    X, y = preprocess_input_data(data)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Save the preprocessed data as a new CSV file
    preprocessed_data = pd.DataFrame(X, columns=data.drop('loan_status', axis=1).columns)
    preprocessed_data['loan_status'] = y
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    try:
        # Train the model
        train_model()
    except Exception as e:
        with open("error_log.txt", "w") as file:
            file.write(f"Error: {str(e)}\n")
            file.write(traceback.format_exc())

