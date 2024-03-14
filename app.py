from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

app = Flask(__name__)

import pickle
# Load the model
gb_model=pickle.load(open('pickle.pkl','rb')) 

import joblib
feature_transform = joblib.load('feature_transform.pkl')
label_encoders = joblib.load('label_encoders.pkl')



# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    Age = float(request.form['Age'])
    Gender = request.form['Gender']
    Country = request.form['Country']
    state = request.form['state']
    self_employed = request.form['self_employed']
    family_history = request.form['family_history']
    work_interfere = request.form['work_interfere']
    no_employees = request.form['no_employees']
    remote_work = request.form['remote_work']
    tech_company = request.form['tech_company']
    benefits = request.form['benefits']
    care_options = request.form['care_options']
    wellness_program = request.form['wellness_program']
    seek_help = request.form['seek_help']
    anonymity = request.form['anonymity']
    leave = request.form['leave']
    mental_health_consequence = request.form['mental_health_consequence']
    phys_health_consequence = request.form['phys_health_consequence']
    coworkers = request.form['coworkers']
    supervisor = request.form['supervisor']
    mental_health_interview = request.form['mental_health_interview']
    phys_health_interview = request.form['phys_health_interview']
    mental_vs_physical = request.form['mental_vs_physical']
    obs_consequence = request.form['obs_consequence']
        
    # Handle encoding for 'self_employed' column
    self_employed_column = 'self_employed'
    self_employed_encoded_value = None

    if self_employed_column in label_encoders:
        self_employed_encoded_value = label_encoders[self_employed_column]

    self_employed_encoded = 1 if self_employed == 'Yes' else 0 if self_employed == 'No' else None

    # Handle encoding for 'family_history' column
    family_history_column = 'family_history'
    family_history_encoded_value = None    

    if family_history_column in label_encoders:
        family_history_encoded_value = label_encoders[family_history_column]

        # Assuming family_history is a binary column
    family_history_encoded = 1 if family_history == 'Yes' else 0 if family_history == 'No' else None
   
    # Handle encoding for 'work_interfere' column
    work_interfere_column = 'work_interfere'
    work_interfere_encoded_value = None

    if work_interfere_column in label_encoders:
        work_interfere_encoded_value = label_encoders[work_interfere_column]

        # Assuming 'work_interfere' is an ordinal variable
    work_interfere_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    work_interfere_encoded = work_interfere_mapping.get(work_interfere, None)

    # Handle encoding for 'no_employees' column
    no_employees_column = 'no_employees'
    no_employees_encoded_value = label_encoders.get(no_employees_column, None)

    if no_employees_encoded_value is not None:
        try:
            no_employees_encoded = no_employees_encoded_value.transform([no_employees])[0]
        except ValueError:
        # Handle the case where an unseen label is encountered
            print(f"Unseen label '{no_employees}' in column '{no_employees_column}'. Using a default value.")
            no_employees_encoded = None
            
    else:
        print(f"Label encoder not found for column '{no_employees_column}'.")
        no_employees_encoded = None


    # Handle encoding for 'remote_work' column
    remote_work_column = 'remote_work'
    remote_work_encoded_value = None

    if remote_work_column in label_encoders:
        remote_work_encoded_value = label_encoders[remote_work_column]

        # Assuming remote_work is a binary column
    remote_work_encoded = 1 if remote_work == 'Yes' else 0 if remote_work == 'No' else None

    # Handle encoding for 'tech_company' column
    tech_company_column = 'tech_company'
    tech_company_encoded_value = None

    if tech_company_column in label_encoders:
        tech_company_encoded_value = label_encoders[tech_company_column]

    # Assuming tech_company is a binary column
    tech_company_encoded = 1 if tech_company == 'Yes' else 0 if tech_company == 'No' else None
       
    
    categorical_columns = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
                        'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor',
                        'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']


        
    encoded_values = {}

    for column in categorical_columns:
        column_encoded_value = label_encoders.get(column, None)

        if column_encoded_value is not None:
        # Assuming binary encoding for these columns
            encoded_values[column] = 1 if request.form[column] == 'Yes' else 0 if request.form[column] == 'No' else None
        else:
            print(f"Label encoder not found for column '{column}'.")

        
        # Create a DataFrame with the entered values
    employee_data =pd.DataFrame({
        'Age':Age,
        'Gender': label_encoders['Gender'].transform([Gender])[0] if 'Gender' in label_encoders else None,
        'Country': label_encoders['Country'].transform([Country])[0] if 'Country' in label_encoders else None,
        'state': state,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': encoded_values['benefits'],
        'care_options': encoded_values['care_options'],
        'wellness_program': encoded_values['wellness_program'],
        'seek_help': encoded_values['seek_help'],
        'anonymity': encoded_values['anonymity'],
        'leave': encoded_values['leave'],
        'mental_health_consequence': encoded_values['mental_health_consequence'],
        'phys_health_consequence': encoded_values['phys_health_consequence'],
        'coworkers': encoded_values['coworkers'],
        'supervisor': encoded_values['supervisor'],
        'mental_health_interview': encoded_values['mental_health_interview'],
        'phys_health_interview': encoded_values['phys_health_interview'],
        'mental_vs_physical': encoded_values['mental_vs_physical'],
        'obs_consequence': encoded_values['obs_consequence']
    }, index=[0]) 
       
        
    categorical_columns= ['Age', 'Gender', 'Country','state', 'self_employed', 'family_history', 'work_interfere',
                            'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
                            'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                            'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                            'phys_health_interview', 'mental_vs_physical', 'obs_consequence']   
       
    
        # Convert categorical variables using label encoders
    for column in categorical_columns:
        column_encoded_value = label_encoders.get(column, None)
        if column_encoded_value is not None and column in employee_data.columns:
            try:
                not_null_indices = employee_data[column].notnull()
                employee_data.loc[not_null_indices, column] = column_encoded_value.transform(
                employee_data.loc[not_null_indices, column]
                )
            except ValueError as e:
                print(f"Error transforming column '{column}': {e}")
            # Handle the error (e.g., set to None or a default value)
                employee_data[column] = None
        else:
        # Handle the case where the column is not present in the input data or encoder is None
           employee_data[column] = None

    import numpy as np
    from sklearn.preprocessing import StandardScaler  # Import the scaler    

    data = pd.read_csv("data/data/survey.csv")
    # Assuming 'treatment' is the target variable
    X_train = data.drop(['treatment', 'comments','Timestamp'], axis=1)
    y_train = data['treatment']
    
    data = data.drop(columns=['Timestamp', 'comments'])
    categorical_columns = ['Gender', 'Country', 'state', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
                        'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
                        'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                        'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

    X_train = pd.get_dummies(X_train, columns=categorical_columns)
    
    # Split the dataset into training and testing sets
    categorical_columns_prediction = ['Gender', 'Country', 'state', 'self_employed', 'family_history', 'work_interfere',
                                    'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
                                    'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                                    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                                    'phys_health_interview', 'mental_vs_physical', 'obs_consequence']




    new_employee_data = pd.DataFrame({'Age':Age,
                                        'Gender': label_encoders['Gender'].transform([Gender])[0] if 'Gender' in label_encoders else None,
                                        'Country': label_encoders['Country'].transform([Country])[0] if 'Country' in label_encoders else None,
                                        'state': state,
                                        'self_employed': self_employed,
                                        'family_history': family_history,
                                        'work_interfere': work_interfere,
                                        'no_employees': no_employees,
                                        'remote_work': remote_work,
                                        'tech_company': tech_company,
                                        'benefits': encoded_values['benefits'],
                                        'care_options': encoded_values['care_options'],
                                        'wellness_program': encoded_values['wellness_program'],
                                        'seek_help': encoded_values['seek_help'],
                                        'anonymity': encoded_values['anonymity'],
                                        'leave': encoded_values['leave'],
                                        'mental_health_consequence': encoded_values['mental_health_consequence'],
                                        'phys_health_consequence': encoded_values['phys_health_consequence'],
                                        'coworkers': encoded_values['coworkers'],
                                        'supervisor': encoded_values['supervisor'],
                                        'mental_health_interview': encoded_values['mental_health_interview'],
                                        'phys_health_interview': encoded_values['phys_health_interview'],
                                        'mental_vs_physical': encoded_values['mental_vs_physical'],
                                        'obs_consequence': encoded_values['obs_consequence']}, index=[0]) 
    
    employee_data_encoded = pd.get_dummies(employee_data, columns=categorical_columns_prediction)

    missing_columns = set(X_train.columns) - set(employee_data_encoded.columns)   
  
    for col in missing_columns:
        employee_data_encoded[col] = 0

    employee_data_encoded = employee_data_encoded[X_train.columns]


    employee_data_encoded = employee_data_encoded.fillna(0)


    prediction = gb_model.predict(employee_data_encoded)

    
       
    


        # Map numerical prediction back to "yes" or "no"
    prediction_label = "Take care of your Health" if prediction[0] == 1 else "You need to consult your Doctor"

    return render_template('result.html',prediction_label=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
