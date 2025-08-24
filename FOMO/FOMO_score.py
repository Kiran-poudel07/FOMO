import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace filename with your CSV file path)
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")
# df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# -------------------------
# 1. Gender-wise FOMO
# -------------------------
gender_fomo = df.groupby(["Gender", "FOMO_Level"]).size().reset_index(name="count")
gender_fomo_pivot = gender_fomo.pivot(index="FOMO_Level", columns="Gender", values="count").fillna(0)

print("Gender-wise FOMO counts:\n", gender_fomo_pivot)

# Bar plot
plt.figure(figsize=(8,6))
sns.countplot(data=df, x="FOMO_Level", hue="Gender")
plt.title("FOMO Level by Gender")
plt.show()

# Pie chart
plt.figure(figsize=(8,8))
df["FOMO_Level"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
plt.title("Overall FOMO Distribution")
plt.ylabel("")
plt.show()

# -------------------------
# 2. Year-wise FOMO
# -------------------------
year_fomo = df.groupby(["Year", "FOMO_Level"]).size().reset_index(name="count")
year_fomo_pivot = year_fomo.pivot(index="FOMO_Level", columns="Year", values="count").fillna(0)

print("Year-wise FOMO counts:\n", year_fomo_pivot)

# Bar plot
plt.figure(figsize=(8,6))
sns.countplot(data=df, x="FOMO_Level", hue="Year")
plt.title("FOMO Level by Academic Year")
plt.show()

# -------------------------
# 3. Department-wise FOMO
# -------------------------
dept_fomo = df.groupby(["Department", "FOMO_Level"]).size().reset_index(name="count")
dept_fomo_pivot = dept_fomo.pivot(index="FOMO_Level", columns="Department", values="count").fillna(0)

print("Department-wise FOMO counts:\n", dept_fomo_pivot)

# Bar plot
plt.figure(figsize=(8,6))
sns.countplot(data=df, x="FOMO_Level", hue="Department")
plt.title("FOMO Level by Department")
plt.show()

# -------------------------
# 4. Percentages
# -------------------------
def percentage_table(group_col):
    return (df.groupby([group_col, "FOMO_Level"]).size() /
            df.groupby(group_col).size() * 100).reset_index(name="percentage")

print("\nGender-wise FOMO percentages:\n", percentage_table("Gender"))
print("\nYear-wise FOMO percentages:\n", percentage_table("Year"))
print("\nDepartment-wise FOMO percentages:\n", percentage_table("Department"))
