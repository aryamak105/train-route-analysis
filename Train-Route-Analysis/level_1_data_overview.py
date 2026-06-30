# =============================================================
# Project : Train Route Analysis and Journey Time Prediction
# Intern Name : Arya Makwana
# Internship : Data Science Internship
# Organization : Sysslan IT Solutions
# Level : 1 - Data Overview
# ==============================================================

import pandas as pd

# Load Dataset
df = pd.read_csv("C:/data_sci/Dataset1.csv")

print("=" * 50)
print("LEVEL 1 - DATA OVERVIEW")
print("=" * 50)

# Task 1.1 Dataset Summary
print("\nTASK 1.1 - DATASET SUMMARY\n")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumn Names:\n")
print(df.columns.tolist())

# Task 1.2 Train-wise Start and End Stations
print("\nTASK 1.2 - TRAIN ROUTES\n")

route_table = df.groupby("Train_No")["Station_Name"].agg(
    Start_Station="first",
    End_Station="last"
)

print(route_table.head(10))

# Task 1.3 Basic Statistics
print("\nTASK 1.3 - DISTANCE STATISTICS\n")

print(df["Distance"].describe())

# Number of stops per train
stops = df.groupby("Train_No").size()

print("\nStops Statistics")
print(stops.describe())

# Task 1.4 Missing Values
print("\nTASK 1.4 - MISSING VALUES\n")

print(df.isnull().sum())

print("\nDuplicate Records:\n", df.duplicated().sum())

print("\nLevel 1 Completed Successfully!\n")