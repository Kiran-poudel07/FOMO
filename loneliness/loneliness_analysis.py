import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("loneliness_encoded.csv")

# Classification function for loneliness based on the loneliness score
def classify_loneliness(value):
    if value <= 2.5:           # Strongly agree or Agree
        return "High Loneliness"
    elif value <= 3.5:         # Neutral
        return "Moderate Loneliness"
    else:                    # Disagree or Strongly Disagree
        return "Low Loneliness"

# Apply classification
df['loneliness_level'] = df['loneliness'].apply(classify_loneliness)

# Count and percentage for loneliness levels
loneliness_counts = df['loneliness_level'].value_counts(normalize=True) * 100
loneliness_counts = loneliness_counts.reindex(['High Loneliness', 'Moderate Loneliness', 'Low Loneliness']).fillna(0)

# Bar Chart
plt.figure(figsize=(6,4))
loneliness_counts.plot(kind='bar', color=['red', 'orange', 'skyblue'])
plt.title('Loneliness Levels (Bar Chart)')
plt.ylabel('Percentage (%)')
plt.xlabel('Loneliness Level')
plt.xticks(rotation=0)
for i, v in enumerate(loneliness_counts):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
colors = ['red', 'orange', 'skyblue']
plt.pie(loneliness_counts, labels=loneliness_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Loneliness Levels (Pie Chart)')
plt.tight_layout()
plt.show()

# Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='loneliness_level', order=['High Loneliness', 'Moderate Loneliness', 'Low Loneliness'],
              palette=['red', 'orange', 'skyblue'])
plt.title('Loneliness Level Counts')
plt.ylabel('Number of Students')
plt.xlabel('Loneliness Level')
plt.tight_layout()
plt.show()

# Distribution histogram + KDE plot of loneliness scores
plt.figure(figsize=(6,4))
sns.histplot(df['loneliness'], bins=10, kde=True, color='purple')
plt.title('Distribution of Loneliness Scores')
plt.xlabel('Loneliness Score (1 = Low, 5 = High)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Box Plot by Gender
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Gender', y='loneliness', palette='Set2')
plt.title('Loneliness Scores by Gender')
plt.ylabel('Loneliness Score')
plt.tight_layout()
plt.show()

# Violin Plot by Gender
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Gender', y='loneliness', palette='Set2')
plt.title('Loneliness Distribution by Gender')
plt.ylabel('Loneliness Score')
plt.tight_layout()
plt.show()

# Box Plot by Department
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Department', y='loneliness', palette='coolwarm')
plt.title('Loneliness Scores by Department')
plt.ylabel('Loneliness Score')
plt.tight_layout()
plt.show()

# Violin Plot by Department
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Department', y='loneliness', palette='coolwarm')
plt.title('Loneliness Distribution by Department')
plt.ylabel('Loneliness Score')
plt.tight_layout()
plt.show()

# Box Plot by Year
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Year', y='loneliness', palette='Spectral')
plt.title('Loneliness Scores by Academic Year')
plt.ylabel('Loneliness Score')
plt.tight_layout()
plt.show()

# Violin Plot by Year
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Year', y='loneliness', palette='Spectral')
plt.title('Loneliness Distribution by Academic Year')
plt.ylabel('Loneliness Score')
plt.tight_layout()
plt.show()

# Countplot: Loneliness Level by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='loneliness_level', hue='Gender',
              order=['High Loneliness', 'Moderate Loneliness', 'Low Loneliness'], palette='Set2')
plt.title('Loneliness Level by Gender')
plt.xlabel('Loneliness Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Pie Chart by Gender
genders = df['Gender'].unique()
for gender in genders:
    subset = df[df['Gender'] == gender]
    dist = subset['loneliness_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['High Loneliness', 'Moderate Loneliness', 'Low Loneliness']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Loneliness Distribution – {gender}')
    plt.tight_layout()
    plt.show()

# Bar Chart by Department
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='loneliness_level', hue='Department',
              order=['High Loneliness', 'Moderate Loneliness', 'Low Loneliness'], palette='coolwarm')
plt.title('Loneliness Level by Department')
plt.xlabel('Loneliness Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Bar Chart by Year
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='loneliness_level', hue='Year',
              order=['High Loneliness', 'Moderate Loneliness', 'Low Loneliness'], palette='Spectral')
plt.title('Loneliness Level by Academic Year')
plt.xlabel('Loneliness Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Pie Chart by Year
years = df['Year'].unique()
for year in years:
    subset = df[df['Year'] == year]
    dist = subset['loneliness_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['High Loneliness', 'Moderate Loneliness', 'Low Loneliness']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Loneliness Distribution – {year} Year')
    plt.tight_layout()
    plt.show()

# # Normalize label to lowercase + replace spaces with underscores for column naming
# df['loneliness_level'] = df['loneliness_level'].str.lower().str.replace(" ", "_")

# # One-hot encode the loneliness levels
# one_hot = pd.get_dummies(df['loneliness_level'], prefix='loneliness')
# one_hot = one_hot.astype(int)

# # Rename columns
# one_hot = one_hot.rename(columns={
#     'loneliness_high_loneliness': 'high_loneliness',
#     'loneliness_moderate_loneliness': 'moderate_loneliness',
#     'loneliness_low_loneliness': 'low_loneliness'
# })

# # Merge one-hot columns
# df = pd.concat([df, one_hot], axis=1)

# # Export to CSV
# df.to_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\loneliness_encoded.csv", index=False)
