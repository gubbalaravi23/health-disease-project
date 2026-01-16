import pandas as pd
import numpy as np

# Create a synthetic dataset similar to the UCI Heart Disease dataset
np.random.seed(42)
num_samples = 300

data = {
    'age': np.random.randint(29, 78, num_samples),
    'sex': np.random.randint(0, 2, num_samples), # 1: Male, 0: Female
    'cp': np.random.randint(0, 4, num_samples), # Chest Pain Type
    'trestbps': np.random.randint(94, 200, num_samples), # Resting BP
    'chol': np.random.randint(126, 564, num_samples), # Cholesterol
    'fbs': np.random.randint(0, 2, num_samples), # Fasting Blood Sugar
    'restecg': np.random.randint(0, 3, num_samples), # Resting ECG
    'thalach': np.random.randint(71, 202, num_samples), # Max Heart Rate
    'exang': np.random.randint(0, 2, num_samples), # Exercise Angina
    'oldpeak': np.round(np.random.uniform(0, 6.2, num_samples), 1), # ST Depression
    'slope': np.random.randint(0, 3, num_samples), 
    'ca': np.random.randint(0, 5, num_samples), # Major vessels
    'thal': np.random.randint(0, 4, num_samples), 
    'target': np.random.randint(0, 2, num_samples) # 0: No Disease, 1: Disease
}

df = pd.DataFrame(data)
df.to_csv('heart.csv', index=False)
print("✅ 'heart.csv' generated successfully!")