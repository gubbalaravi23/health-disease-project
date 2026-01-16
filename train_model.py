import pandas as pd
import numpy as np
import pickle # Used to save the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
try:
    df = pd.read_csv('heart.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: heart.csv not found. Run generate_data.py first!")
    exit()

# 2. Data Preprocessing
X = df.drop(columns=['target'])
y = df['target']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for Medical Data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete!")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Save Model & Scaler
# We save the scaler too, because user input needs to be scaled exactly like training data
pickle.dump(model, open('heart_disease_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("💾 Model and Scaler saved successfully as .pkl files!")