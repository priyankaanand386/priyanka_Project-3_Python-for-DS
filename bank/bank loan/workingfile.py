import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle

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

def preprocess_input_data(data):
    # Convert gender, married, and self_employed to numeric values
    data['gender'] = data['gender'].replace({'Male': 1, 'Female': 0})
    data['married'] = data['married'].replace({'Yes': 1, 'No': 0})
    data['self_employed'] = data['self_employed'].replace({'Yes': 1, 'No': 0})

    # Convert education and property_area to one-hot encoded features
    data_encoded = pd.get_dummies(data, columns=['education', 'property_area'])

    # Ensure that the column names match the feature names used during training
    data_encoded = data_encoded.rename(columns={
        'applicant_income': 'applicantincome',
        'coapplicant_income': 'coapplicantincome',
        'loan_amount': 'loanamount',
        'education_Graduate': 'education_0',
        'education_Not Graduate': 'education_1',
        'property_area_Urban': 'property_area_0',
        'property_area_Rural': 'property_area_1',
        'property_area_Semiurban': 'property_area_2'
    })

    # Select only the columns used during training
    selected_columns = ['applicantincome', 'coapplicantincome', 'loanamount',
                        'education_0', 'education_1', 'property_area_0',
                        'property_area_1', 'property_area_2']
    data_encoded = data_encoded[selected_columns]

    return data_encoded

def predict_loan_eligibility(gender, married, dependents, education, self_employed,
                             applicant_income, coapplicant_income, loan_amount,
                             loan_amount_term, credit_history, property_area):
    # Load the trained model
    model = load_model()

    # Prepare the input data
    data = pd.DataFrame({
        'gender': [gender],
        'married': [married],
        'dependents': [dependents],
        'education': [education],
        'self_employed': [self_employed],
        'applicant_income': [applicant_income],
        'coapplicant_income': [coapplicant_income],
        'loan_amount': [loan_amount],
        'loan_amount_term': [loan_amount_term],
        'credit_history': [credit_history],
        'property_area': [property_area]
    })

    # Preprocess the input data
    data_encoded = preprocess_input_data(data)

    # Make the prediction using the trained model
    prediction = model.predict(data_encoded)

    return prediction[0]  # Return the first prediction


if __name__ == '__main__':
    # Train the model
    train_model()

    # Provide example input values for prediction
    gender = 'Male'
    married = 'Yes'
    dependents = 1
    education = 'Graduate'
    self_employed = 'No'
    applicant_income = 5000
    coapplicant_income = 2000
    loan_amount = 100000
    loan_amount_term = 360
    credit_history = 1
    property_area = 'Urban'

    # Make the prediction
    prediction = predict_loan_eligibility(gender, married, dependents, education, self_employed,
                                          applicant_income, coapplicant_income, loan_amount,
                                          loan_amount_term, credit_history, property_area)

    # Display the result
    if prediction == 1:
        print("Congrats!! You are eligible for the loan.")
    else:
        print("Sorry, you are not eligible for the loan.")
