import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your life satisfaction data (adjust path)
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\life_satisfaction_encoded.csv")

# Classification function for life satisfaction (1=Strongly Agree, 5=Strongly Disagree)
def classify_ls(value):
    if value <= 2.0:
        return "Satisfied"
    elif value <= 3.0:
        return "Neutral"
    else:
        return "Dissatisfied"

# Apply classification
df['life_satisfaction_level'] = df['life_satisfaction'].apply(classify_ls)

# Count and percentage for life satisfaction levels
ls_counts = df['life_satisfaction_level'].value_counts(normalize=True) * 100
ls_counts = ls_counts.reindex(['Satisfied', 'Neutral', 'Dissatisfied']).fillna(0)

# Bar Chart
plt.figure(figsize=(6,4))
ls_counts.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Life Satisfaction Levels (Bar Chart)')
plt.ylabel('Percentage (%)')
plt.xlabel('Life Satisfaction Level')
plt.xticks(rotation=0)
for i, v in enumerate(ls_counts):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
colors = ['green', 'orange', 'red']
plt.pie(ls_counts, labels=ls_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Life Satisfaction Levels (Pie Chart)')
plt.tight_layout()
plt.show()

# Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='life_satisfaction_level', order=['Satisfied', 'Neutral', 'Dissatisfied'],
              palette=['green', 'orange', 'red'])
plt.title('Life Satisfaction Level Counts')
plt.ylabel('Number of Students')
plt.xlabel('Life Satisfaction Level')
plt.tight_layout()
plt.show()

# Histogram + KDE of raw life satisfaction scores
plt.figure(figsize=(6,4))
sns.histplot(df['life_satisfaction'], bins=10, kde=True, color='purple')
plt.title('Distribution of Life Satisfaction Scores')
plt.xlabel('Life Satisfaction Score (1 = High Satisfaction, 5 = Low)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Box Plot by Gender
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Gender', y='life_satisfaction', palette='Set2')
plt.title('Life Satisfaction Scores by Gender')
plt.ylabel('Life Satisfaction Score')
plt.tight_layout()
plt.show()

# Violin Plot by Gender
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Gender', y='life_satisfaction', palette='Set2')
plt.title('Life Satisfaction Distribution by Gender')
plt.ylabel('Life Satisfaction Score')
plt.tight_layout()
plt.show()

# Box Plot by Department
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Department', y='life_satisfaction', palette='coolwarm')
plt.title('Life Satisfaction Scores by Department')
plt.ylabel('Life Satisfaction Score')
plt.tight_layout()
plt.show()

# Violin Plot by Department
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Department', y='life_satisfaction', palette='coolwarm')
plt.title('Life Satisfaction Distribution by Department')
plt.ylabel('Life Satisfaction Score')
plt.tight_layout()
plt.show()

# Box Plot by Year
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Year', y='life_satisfaction', palette='Spectral')
plt.title('Life Satisfaction Scores by Academic Year')
plt.ylabel('Life Satisfaction Score')
plt.tight_layout()
plt.show()

# Violin Plot by Year
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Year', y='life_satisfaction', palette='Spectral')
plt.title('Life Satisfaction Distribution by Academic Year')
plt.ylabel('Life Satisfaction Score')
plt.tight_layout()
plt.show()

# Countplot: Life Satisfaction Level by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='life_satisfaction_level', hue='Gender',
              order=['Satisfied', 'Neutral', 'Dissatisfied'], palette='Set2')
plt.title('Life Satisfaction Level by Gender')
plt.xlabel('Life Satisfaction Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Pie Chart by Gender
genders = df['Gender'].unique()
for gender in genders:
    subset = df[df['Gender'] == gender]
    dist = subset['life_satisfaction_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['Satisfied', 'Neutral', 'Dissatisfied']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['green', 'orange', 'red'], startangle=140)
    plt.title(f'Life Satisfaction Distribution – {gender}')
    plt.tight_layout()
    plt.show()

# Bar Chart by Department
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='life_satisfaction_level', hue='Department',
              order=['Satisfied', 'Neutral', 'Dissatisfied'], palette='coolwarm')
plt.title('Life Satisfaction Level by Department')
plt.xlabel('Life Satisfaction Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Bar Chart by Year
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='life_satisfaction_level', hue='Year',
              order=['Satisfied', 'Neutral', 'Dissatisfied'], palette='Spectral')
plt.title('Life Satisfaction Level by Academic Year')
plt.xlabel('Life Satisfaction Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Pie Chart by Year
years = df['Year'].unique()
for year in years:
    subset = df[df['Year'] == year]
    dist = subset['life_satisfaction_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['Satisfied', 'Neutral', 'Dissatisfied']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['green', 'orange', 'red'], startangle=140)
    plt.title(f'Life Satisfaction Distribution – {year} Year')
    plt.tight_layout()
    plt.show()

# # Normalize label to lowercase + replace spaces with underscores for column naming
# df['life_satisfaction_level'] = df['life_satisfaction_level'].str.lower().str.replace(" ", "_")

# # One-hot encode the life satisfaction levels
# one_hot = pd.get_dummies(df['life_satisfaction_level'], prefix='life_satisfaction')
# one_hot = one_hot.astype(int)

# # Rename columns
# one_hot = one_hot.rename(columns={
#     'life_satisfaction_satisfied': 'satisfied_life_satisfaction',
#     'life_satisfaction_neutral': 'neutral_life_satisfaction',
#     'life_satisfaction_dissatisfied': 'dissatisfied_life_satisfaction'
# })

# # Merge one-hot columns
# df = pd.concat([df, one_hot], axis=1)

# # Export to CSV
# df.to_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\life_satisfaction_encoded.csv", index=False)
