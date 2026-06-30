# ==============================================================
# Project : Train Route Analysis and Journey Time Prediction
# Intern Name : Arya Makwana
# Internship : Data Science Internship
# Organization : Sysslan IT Solutions
# Level : 6 - Final Data Science Project
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("=" * 60)
print("---------FINAL DATA SCIENCE PROJECT---------")
print("TRAIN ROUTE ANALYSIS AND JOURNEY TIME PREDICTION")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("C:/data_sci/Dataset1.csv")

print("\nDataset Loaded Successfully\n")
print("Total Records:", len(df))

# ==========================================
# Feature Engineering
# ==========================================

# Number of stops per train
stops = df.groupby("Train_No").size().reset_index(name="Number_of_Stops")

# Maximum distance covered by train
distance = df.groupby("Train_No")["Distance"].max().reset_index()

# Merge datasets
model_df = pd.merge(distance, stops, on="Train_No")

# Estimated Journey Duration
model_df["Journey_Duration"] = model_df["Distance"] / 50

print("\nModel Dataset Created Successfully\n")
print(model_df.head())

# ==========================================
# Prepare Features and Target
# ==========================================

X = model_df[["Distance", "Number_of_Stops"]]
y = model_df["Journey_Duration"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Trained Successfully\n")

# ==========================================
# Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nMODEL PERFORMANCE\n")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))

# ==========================================
# Actual vs Predicted Graph
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Journey Duration")
plt.ylabel("Predicted Journey Duration")
plt.title("Actual vs Predicted Journey Duration")

plt.tight_layout()
plt.show()

# ==========================================
# Sample Results
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": predictions[:10]
})

print("\nSample Predictions\n")
print(results)

print("\nProject Completed Successfully!\n")