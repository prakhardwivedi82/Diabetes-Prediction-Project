import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
with open('diabetes_prediction_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the scaler (we need to recreate it since it wasn't saved)
# We'll recreate the scaler with the same parameters as in the notebook
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the original dataset to fit the scaler
diabetes_dataset = pd.read_csv('diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler = StandardScaler()
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])
        
        # Create input array
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        
        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        
        # Make prediction
        prediction = classifier.predict(std_data)
        
        # Prepare result
        if prediction[0] == 0:
            result = "The person is NOT diabetic"
            result_class = "not-diabetic"
        else:
            result = "The person IS diabetic"
            result_class = "diabetic"
        
        return render_template('index.html', prediction=result, result_class=result_class)
        
    except Exception as e:
        error_message = f"Error processing prediction: {str(e)}"
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
