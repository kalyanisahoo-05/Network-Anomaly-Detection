# train_model.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load Dataset
data = pd.read_csv(r"C:\Users\Kalyani\OneDrive\Desktop\Network Anomaly Detection\dataset.csv.csv")

print("Dataset Loaded Successfully")
print(data.head())


# 2. Define Features and Target
X = data.drop("Label", axis=1)
y = data["Label"]


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

print("Model Training Completed")


# 6. Model Prediction
y_pred = model.predict(X_test)


# 7. Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Save Model
pickle.dump(model, open("anomaly_model.pkl", "wb"))

print("Model saved as anomaly_model.pkl")