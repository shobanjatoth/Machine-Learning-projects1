import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

# Load training data
train_data = pd.read_csv("artifacts/train.csv")  # Update with actual path

# Select numerical columns for scaling (modify based on your dataset)
num_features = ["feature1", "feature2", "feature3"]  # Replace with actual feature names
X_train = train_data[num_features]

# Fit the preprocessor
preprocessor = StandardScaler()
preprocessor.fit(X_train)  

# Save the trained preprocessor
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
save_object(preprocessor_path, preprocessor)

print(f" Preprocessor trained and saved at: {preprocessor_path}")

