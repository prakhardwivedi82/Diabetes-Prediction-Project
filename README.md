# Diabetes Prediction Flask Web Application

A web application that predicts whether a person has diabetes based on health parameters using a trained machine learning model.

## Features

- Clean, modern web interface
- Real-time diabetes prediction
- Input validation and error handling
- Responsive design

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - Fill in the health parameters
   - Click "Predict Diabetes Risk"

## Required Files

- `app.py` - Main Flask application
- `diabetes_prediction_model.pkl` - Trained machine learning model
- `diabetes.csv` - Original dataset (for scaler fitting)
- `templates/index.html` - Web interface template
- `requirements.txt` - Python dependencies

## Input Parameters

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (kg/m²)
- **Diabetes Pedigree Function**: Diabetes pedigree function
- **Age**: Age in years

## Model Information

- **Algorithm**: Support Vector Machine (SVM) with linear kernel
- **Accuracy**: ~77% on test data
- **Preprocessing**: StandardScaler normalization

## Notes

- This is a prediction tool and should not replace professional medical advice
- The model is trained on the PIMA Diabetes Dataset
- Results should be interpreted with caution and verified by healthcare professionals
