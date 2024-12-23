import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset (replace with your dataset)
path='Heart_Disease_Prediction.csv'
df = pd.read_csv(path)

# Check the first few rows of the dataset
print(df.head())
X = df.drop('Heart Disease', axis=1)  # Features (all columns except 'HeartDisease')
y = df['Heart Disease'] 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for some models, e.g., Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.to_numpy())  # Convert to NumPy array
X_test_scaled = scaler.transform(X_test.to_numpy()) 

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler to .pkl file using pickle
with open('heart_disease_model.sav', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and Scaler saved successfully.")

