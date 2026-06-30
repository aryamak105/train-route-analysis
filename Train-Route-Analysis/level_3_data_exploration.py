# ============================================================
# Project : Train Route Analysis and Journey Time Prediction
# Intern Name : Arya Makwana
# Internship : Data Science Internship
# Organization : Sysslan IT Solutions
# Level : 3 - Data Exploration
# ============================================================


import pandas as pd

# Load Dataset
df = pd.read_csv("C:/data_sci/Dataset1.csv")

print("=" * 55)
print("LEVEL 3 - DATA EXPLORATION")
print("=" * 55)

# ==========================================
# Task 3.1 Compare Routes
# ==========================================

print("\nTASK 3.1 - ROUTE ANALYSIS\n")

route_summary = df.groupby("Route_Number").agg({
    "Distance": ["count", "min", "max", "mean"]
})

print(route_summary.head(10))

# ==========================================
# Task 3.2 High and Low Traffic Stations
# ==========================================

print("\nTASK 3.2 - STATION TRAFFIC ANALYSIS\n")

traffic = df["Station_Name"].value_counts()

print("\nTop 10 High Traffic Stations\n")
print(traffic.head(10))

print("\nTop 10 Low Traffic Stations\n")
print(traffic.tail(10))

# ==========================================
# Task 3.3 Distance Analysis
# ==========================================

print("\nTASK 3.3 - DISTANCE ANALYSIS\n")

print(df["Distance"].describe())

# ==========================================
# Task 3.4 Key Insights
# ==========================================

print("\nTASK 3.4 - KEY INSIGHTS\n")

print("""
1. Different routes cover different distances.

2. Some stations appear much more frequently,
   indicating higher train traffic.

3. Distance varies significantly across routes.

4. Major railway stations handle more trains.

5. Railway network contains both short and
   long-distance routes.
""")

print("\nLevel 3 Completed Successfully!\n")