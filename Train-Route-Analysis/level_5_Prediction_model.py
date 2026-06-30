# ==============================================================
# Project : Train Route Analysis and Journey Time Prediction
# Intern Name : Arya Makwana
# Internship : Data Science Internship
# Organization : Sysslan IT Solutions
# Level : 5 - Prediction Model Devlopment
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("=" * 55)
print("LEVEL 5 - PREDICTION MODEL DEVELOPMENT")
print("=" * 55)

# Load Dataset
df = pd.read_csv("C:/data_sci/Dataset1.csv")

# ==========================================
# Task 5.1 Prepare Modeling Dataset
# ==========================================

print("\nTASK 5.1 - PREPARE MODEL DATA\n")

# Create Number of Stops Feature
stops = df.groupby("Train_No").size().reset_index(name="Number_of_Stops")

# Maximum distance for each train
distance = df.groupby("Train_No")["Distance"].max().reset_index()

# Merge data
model_df = pd.merge(distance, stops, on="Train_No")

# Create target variable
# Approximate Journey Duration (Hours)
model_df["Journey_Duration"] = model_df["Distance"] / 50

print(model_df.head())

# Features
X = model_df[["Distance", "Number_of_Stops"]]

# Target
y = model_df["Journey_Duration"]

# ==========================================
# Task 5.2 Train-Test Split
# ==========================================

print("\nTASK 5.2 - TRAIN TEST SPLIT\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# Task 5.3 Build Linear Regression Model
# ==========================================

print("\nTASK 5.3 - BUILD MODEL\n")

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Trained Successfully")

# ==========================================
# Task 5.4 Evaluate Model
# ==========================================

print("\nTASK 5.4 - MODEL EVALUATION\n")

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

print("\nSample Predictions\n")

results = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": predictions[:10]
})

print(results)

print("\nLevel 5 Completed Successfully!\n")