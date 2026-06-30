# ==============================================================
# Project : Train Route Analysis and Journey Time Prediction
# Intern Name : Arya Makwana
# Internship : Data Science Internship
# Organization : Sysslan IT Solutions
# Level : 2 - Data Cleaning & Feature Engineering
# ===============================================================


import pandas as pd

# Load Dataset
df = pd.read_csv("C:/data_sci/Dataset1.csv")

print("=" * 50)
print("LEVEL 2 - DATA CLEANING & FEATURE ENGINEERING")
print("=" * 50)

# ==========================================
# Task 2.1 Handle Missing Values & Duplicates
# ==========================================

print("\n------------------------TASK 2.1------------------------------\n")

print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values if any
df = df.fillna("Unknown")

print("\nDuplicates Removed Successfully\n")

# ==========================================
# Task 2.2 Standardize Time Format
# ==========================================

print("\n----------------------------TASK 2.2---------------------------\n")

df["Arrival_time"] = df["Arrival_time"].astype(str)
df["Departure_Time"] = df["Departure_Time"].astype(str)

print("Time Columns Standardized")

# ==========================================
# Task 2.3 Calculate Journey Duration
# ==========================================

print("\n-------------------------TASK 2.3-----------------------------\n")

df["Arrival_time"] = pd.to_datetime(
    df["Arrival_time"],
    format="%H:%M",
    errors="coerce"
)

df["Departure_Time"] = pd.to_datetime(
    df["Departure_Time"],
    format="%H:%M",
    errors="coerce"
)

df["Journey_Duration_Minutes"] = (
    df["Departure_Time"] - df["Arrival_time"]
).dt.total_seconds() / 60

print("Journey Duration Feature Created")

# ==========================================
# Task 2.4 Create Features
# ==========================================

print("\n-------------------------TASK 2.4-------------------------------\n")

# Number of stops per train
stops = df.groupby("Train_No").size()

print("\nNumber of Stops Per Train:\n")
print(stops.head())

print("\nTotal Distance Statistics:\n")
print(df["Distance"].describe())

print("\nDataset Shape After Cleaning:\n")
print(df.shape)

print("\nLevel 2 Completed Successfully!\n")