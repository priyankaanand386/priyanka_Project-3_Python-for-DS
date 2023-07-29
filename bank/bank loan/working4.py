import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('loan_approval_data.csv')

# Drop the loan_id column
data = data.drop('loan_id', axis=1)

# Copy the dataset to avoid modifying the original dataframe
data_numeric = data.copy()

# Convert categorical columns to numeric values
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
for column in categorical_columns:
    data_numeric[column] = label_encoder.fit_transform(data[column])

# Convert loan_status column to numeric values
data_numeric['loan_status'] = label_encoder.fit_transform(data['loan_status'])

# Save the modified dataset as a new CSV file without changing the column headers
data_numeric.to_csv('loan_approval_data_numeric.csv', index=False)
