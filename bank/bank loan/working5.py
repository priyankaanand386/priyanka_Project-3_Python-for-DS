import pandas as pd

# Load the original and numeric datasets
original_data = pd.read_csv('loan_approval_data.csv')
numeric_data = pd.read_csv('loan_approval_data_numeric.csv')
# For each categorical variable...
for variable in ['gender', 'married', 'education', 'self_employed', 'credit_history', 'property_area']:
    # Get unique values in the original and numeric datasets
    original_values = original_data[variable].unique()
    numeric_values = numeric_data[variable].unique()

    # Print these unique values
    print(f"For '{variable}', unique original values are: {original_values}")
    print(f"For '{variable}', unique numeric values are: {numeric_values}")
