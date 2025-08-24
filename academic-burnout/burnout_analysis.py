import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("academic_burnout_encoded.csv")

# Classification function for academic burnout
def classify_burnout(value):
    if value >= 3.5:
        return "Exhausted"
    elif value >= 2.5:
        return "At Risk"
    else:
        return "Coping Well"

# Apply classification
df['academic_burnout_level'] = df['academic_burnout'].apply(classify_burnout)

# Count and percentage for burnout levels

burnout_counts = df['academic_burnout_level'].value_counts(normalize=True) * 100
burnout_counts = burnout_counts.reindex(['Exhausted', 'At Risk', 'Coping Well']).fillna(0)

# Count of Academic Burnout Levels by Gender
burnout_gender_counts = df.groupby(['Gender', 'academic_burnout_level']).size().unstack().reindex(columns=['Exhausted', 'At Risk', 'Coping Well']).fillna(0)
print("Academic Burnout Level Counts by Gender:")
print(burnout_gender_counts)

# Count of Academic Burnout Levels by Department
burnout_dept_counts = df.groupby(['Department', 'academic_burnout_level']).size().unstack().reindex(columns=['Exhausted', 'At Risk', 'Coping Well']).fillna(0)
print("\nAcademic Burnout Level Counts by Department:")
print(burnout_dept_counts)

# Count of Academic Burnout Levels by Year
burnout_year_counts = df.groupby(['Year', 'academic_burnout_level']).size().unstack().reindex(columns=['Exhausted', 'At Risk', 'Coping Well']).fillna(0)
print("\nAcademic Burnout Level Counts by Year:")
print(burnout_year_counts)
# Bar Chart
plt.figure(figsize=(6,4))
burnout_counts.plot(kind='bar', color=['red', 'orange', 'skyblue'])
plt.title('Academic Burnout Levels (Bar Chart)')
plt.ylabel('Percentage (%)')
plt.xlabel('Burnout Level')
plt.xticks(rotation=0)
for i, v in enumerate(burnout_counts):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
colors = ['red', 'orange', 'skyblue']
plt.pie(burnout_counts, labels=burnout_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Academic Burnout Levels (Pie Chart)')
plt.tight_layout()
plt.show()

# Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='academic_burnout_level', order=['Exhausted', 'At Risk', 'Coping Well'],
              palette=['red', 'orange', 'skyblue'])
plt.title('Academic Burnout Level Counts')
plt.ylabel('Number of Students')
plt.xlabel('Burnout Level')
plt.tight_layout()
plt.show()

# # Distribution histogram + KDE plot
# plt.figure(figsize=(6,4))
# sns.histplot(df['academic_burnout'], bins=10, kde=True, color='purple')
# plt.title('Distribution of Academic Burnout Scores')
# plt.xlabel('Burnout Score (1 = Low, 5 = High)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# # Box Plot by Gender
# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='Gender', y='academic_burnout', palette='Set2')
# plt.title('Academic Burnout Scores by Gender')
# plt.ylabel('Burnout Score')
# plt.tight_layout()
# plt.show()

# # Violin Plot by Gender
# plt.figure(figsize=(6,4))
# sns.violinplot(data=df, x='Gender', y='academic_burnout', palette='Set2')
# plt.title('Academic Burnout Distribution by Gender')
# plt.ylabel('Burnout Score')
# plt.tight_layout()
# plt.show()

# # Box Plot by Department
# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='Department', y='academic_burnout', palette='coolwarm')
# plt.title('Academic Burnout Scores by Department')
# plt.ylabel('Burnout Score')
# plt.tight_layout()
# plt.show()

# # Violin Plot by Department
# plt.figure(figsize=(6,4))
# sns.violinplot(data=df, x='Department', y='academic_burnout', palette='coolwarm')
# plt.title('Academic Burnout Distribution by Department')
# plt.ylabel('Burnout Score')
# plt.tight_layout()
# plt.show()

# # Box Plot by Year
# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='Year', y='academic_burnout', palette='Spectral')
# plt.title('Academic Burnout Scores by Academic Year')
# plt.ylabel('Burnout Score')
# plt.tight_layout()
# plt.show()

# # Violin Plot by Year
# plt.figure(figsize=(6,4))
# sns.violinplot(data=df, x='Year', y='academic_burnout', palette='Spectral')
# plt.title('Academic Burnout Distribution by Academic Year')
# plt.ylabel('Burnout Score')
# plt.tight_layout()
# plt.show()

# # Countplot: Burnout Level by Gender
# plt.figure(figsize=(6, 4))
# sns.countplot(data=df, x='academic_burnout_level', hue='Gender',
#               order=['Exhausted', 'At Risk', 'Coping Well'], palette='Set2')
# plt.title('Academic Burnout Level by Gender')
# plt.xlabel('Burnout Level')
# plt.ylabel('Number of Students')
# plt.tight_layout()
# plt.show()

# # Pie Chart by Gender
# genders = df['Gender'].unique()
# for gender in genders:
#     subset = df[df['Gender'] == gender]
#     dist = subset['academic_burnout_level'].value_counts(normalize=True) * 100
#     dist = dist.reindex(['Exhausted', 'At Risk', 'Coping Well']).fillna(0)

#     plt.figure(figsize=(5,5))
#     plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
#             colors=['red', 'orange', 'skyblue'], startangle=140)
#     plt.title(f'Academic Burnout Distribution – {gender}')
#     plt.tight_layout()
#     plt.show()



# Pie Chart by Department
departments = df['Department'].unique()
for dept in departments:
    subset = df[df['Department'] == dept]
    dist = subset['academic_burnout_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['Exhausted', 'At Risk', 'Coping Well']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Academic Burnout Distribution – {dept} Department')
    plt.tight_layout()
    plt.show()

# # Bar Chart by Department
# plt.figure(figsize=(7, 4))
# sns.countplot(data=df, x='academic_burnout_level', hue='Department',
#               order=['Exhausted', 'At Risk', 'Coping Well'], palette='coolwarm')
# plt.title('Academic Burnout Level by Department')
# plt.xlabel('Burnout Level')
# plt.ylabel('Number of Students')
# plt.tight_layout()
# plt.show()

# # Bar Chart by Year
# plt.figure(figsize=(7, 4))
# sns.countplot(data=df, x='academic_burnout_level', hue='Year',
#               order=['Exhausted', 'At Risk', 'Coping Well'], palette='Spectral')
# plt.title('Academic Burnout Level by Academic Year')
# plt.xlabel('Burnout Level')
# plt.ylabel('Number of Students')
# plt.tight_layout()
# plt.show()

# # Pie Chart by Year
# years = df['Year'].unique()
# for year in years:
#     subset = df[df['Year'] == year]
#     dist = subset['academic_burnout_level'].value_counts(normalize=True) * 100
#     dist = dist.reindex(['Exhausted', 'At Risk', 'Coping Well']).fillna(0)

#     plt.figure(figsize=(5,5))
#     plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
#             colors=['red', 'orange', 'skyblue'], startangle=140)
#     plt.title(f'Academic Burnout Distribution – {year} Year')
#     plt.tight_layout()
#     plt.show()

# # Normalize label to lowercase + underscore
# df['academic_burnout_level'] = df['academic_burnout_level'].str.lower().str.replace(" ", "_")

# # One-hot encode the burnout levels
# one_hot = pd.get_dummies(df['academic_burnout_level'], prefix='burnout')
# one_hot = one_hot.astype(int)

# # Rename columns
# df = pd.concat([df, one_hot], axis=1)
# df = df.rename(columns={
#     'burnout_exhausted': 'exhausted',
#     'burnout_at_risk': 'at_risk',
#     'burnout_coping_well': 'coping_well'
# })

# # Export to CSV
# df.to_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\academic_burnout_encoded.csv", index=False)
