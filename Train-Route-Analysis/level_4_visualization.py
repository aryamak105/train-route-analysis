# ============================================================
# Project : Train Route Analysis and Journey Time Prediction
# Intern Name : Arya Makwana
# Internship : Data Science Internship
# Organization : Sysslan IT Solutions
# Level : 4 - Visualization & Pattern Analysis
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("C:/data_sci/Dataset1.csv")

print("=" * 55)
print("LEVEL 4 - VISUALIZATION & PATTERN ANALYSIS")
print("=" * 55)

# ==========================================
# Task 4.1 Route-wise Distance Visualization
# ==========================================

route_distance = df.groupby("Route_Number")["Distance"].mean().head(10)

plt.figure(figsize=(10, 5))
route_distance.plot(kind="bar")
plt.title("Average Distance by Route")
plt.xlabel("Route Number")
plt.ylabel("Average Distance")
plt.tight_layout()
plt.show()

print("-------Task 4.1 Completed-------")

# ==========================================
# Task 4.2 Station Traffic Visualization
# ==========================================

top_stations = df["Station_Name"].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=top_stations.values,
    y=top_stations.index
)
plt.title("Top 10 High Traffic Stations")
plt.xlabel("Train Count")
plt.ylabel("Station Name")
plt.tight_layout()
plt.show()

print("-------Task 4.2 Completed--------")

# ==========================================
# Task 4.3 Distance Distribution
# ==========================================

plt.figure(figsize=(10, 5))
plt.hist(df["Distance"], bins=30)
plt.title("Distance Distribution")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("-------Task 4.3 Completed---------")

# ==========================================
# Task 4.4 Pattern Analysis
# ==========================================

print("\nTASK 4.4 - PATTERN ANALYSIS\n")

print("""
1. Some routes cover much larger distances.

2. A few stations handle most train traffic.

3. Most records belong to shorter distances.

4. Distance distribution is not uniform.

5. Railway traffic is concentrated around
   major stations and routes.
""")

print("\nLevel 4 Completed Successfully!\n")