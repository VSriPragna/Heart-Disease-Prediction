import pickle
import numpy as np

# Load the saved model and scaler using pickle
with open('heart_disease_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to get user input for prediction
def get_user_input():
    age = float(input("Enter your age: "))
    sex = int(input("Enter your sex (1 = male, 0 = female): "))
    chest_pain = int(input("Enter chest pain type (0-3): "))
    resting_blood_pressure = float(input("Enter resting blood pressure (in mm Hg): "))
    cholesterol = float(input("Enter serum cholesterol (mg/dl): "))
    fasting_blood_sugar = int(input("Enter fasting blood sugar (1 if > 120 mg/dl, else 0): "))
    rest_ecg = int(input("Enter resting electrocardiographic results (0-2): "))
    max_heart_rate = float(input("Enter maximum heart rate achieved: "))
    exercise_induced_angina = int(input("Enter exercise induced angina (1 if yes, 0 if no): "))
    oldpeak = float(input("Enter depression induced by exercise relative to rest: "))
    slope = int(input("Enter slope of peak exercise ST segment (0-2): "))
    ca = int(input("Enter number of major vessels colored by fluoroscopy (0-3): "))
    thalassemia = int(input("Enter thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect): "))

    return np.array([[age, sex, chest_pain, resting_blood_pressure, cholesterol, fasting_blood_sugar,
                      rest_ecg, max_heart_rate, exercise_induced_angina, oldpeak, slope, ca, thalassemia]])

# Get user input
user_data = get_user_input()

# Scale the user input data using the saved scaler
user_data_scaled = scaler.transform(user_data)

# Predict the heart disease outcome (0 = no disease, 1 = disease)
prediction = model.predict(user_data_scaled)

# Output prediction result
if prediction == 1:
    print("The model predicts: Heart Disease Detected.")
else:
    print("The model predicts: No Heart Disease.")
