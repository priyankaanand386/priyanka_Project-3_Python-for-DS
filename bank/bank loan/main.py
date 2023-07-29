import mysql.connector
from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import bcrypt



app = Flask(__name__)
app.secret_key = '1234'  # Replace 'your_secret_key' with your own secret key

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Varu@2022",
    database="bankdbms"
)

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input_data(data):
    # Convert 'education' column to numeric values
    education_mapping = {'graduate': 1, 'not graduate': 0}
    data['education'] = data['education'].map(education_mapping)

    return data

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

    # Explicit mapping of categorical values to numeric values
    mapping_dict = {
        'gender': {'male': 1, 'female': 0, None: 2},  # 'None' is used to represent 'nan'
        'married': {'no': 0, 'yes': 1, None: 2},
        'education': {'graduate': 0, 'not graduate': 1},
        'self_employed': {'no': 0, 'yes': 1, None: 2},
        'credit_history': {0.0: 0, 1.0: 1, None: None},  # Assuming 'nan' in credit history corresponds to 'nan' in numeric data
        'property_area': {'urban': 2, 'rural': 0, 'semiurban': 1}
    }

    # Map the categorical values to numeric values using the mappings
    for col, mapping in mapping_dict.items():
        data[col] = data[col].map(mapping)

    # Make the prediction using the trained model
    prediction = model.predict(data)

    # If the prediction is a probability, use a threshold to make a binary decision
    return 1 if prediction[0] >= 1 else 0  # Assuming that your model outputs probabilities





# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Perform necessary operations to store the user data in the database
        cursor = db.cursor()
        cursor.execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, hashed_password))
        db.commit()
        cursor.close()

        return redirect(url_for('login'))
    return render_template('register.html')



import bcrypt

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Perform necessary operations to verify the user credentials
        cursor = db.cursor()
        cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if user is not None:
            stored_password = user[2]  # Assuming the hashed password is stored in the third column of the user table

            # Compare the hashed password with the provided password
            if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                # Password matches, user authenticated
                session['loggedin'] = True
                session['id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('predict'))  # Redirect to the predict page

    return render_template('login.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        married = request.form['married']
        dependents = int(request.form['dependents'])
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = request.form['credit_history']
        property_area = request.form['property_area']

        # Perform the prediction using your model
        prediction = predict_loan_eligibility(gender, married, dependents, education, self_employed,
                                              applicant_income, coapplicant_income, loan_amount,
                                              loan_amount_term, credit_history, property_area)

        # Process the prediction result
        if prediction == 1:
            result = "Congrats!! You are eligible for the loan."
        else:
            result = "Sorry, you are not eligible for the loan."

        # Render the template with the prediction result and form data
        return render_template('predictresult.html', result=result, gender=gender, married=married,
                               dependents=dependents, education=education, self_employed=self_employed,
                               applicant_income=applicant_income, coapplicant_income=coapplicant_income,
                               loan_amount=loan_amount, loan_amount_term=loan_amount_term,
                               credit_history=credit_history, property_area=property_area)

    return render_template('predict.html')


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)