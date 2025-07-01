import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv("social_comparison_encoded.csv")
 # Adjust path if needed

# Classify social comparison levels based on value
def classify_sc(value):
    if value <= 2.5:
        return "High"
    elif value <= 3.5:
        return "Medium"
    else:
        return "Low"

df['social_comparison_level'] = df['social_comparison'].apply(classify_sc)

# Count and percentage
level_counts = df['social_comparison_level'].value_counts(normalize=True) * 100
level_counts = level_counts.sort_index()  # Order: High, Medium, Low

# # Bar Chart
# plt.figure(figsize=(6,4))
# level_counts.plot(kind='bar', color=['red', 'orange', 'skyblue'])
# plt.title('Social Comparison Levels (Bar Chart)')
# plt.ylabel('Percentage (%)')
# plt.xlabel('Social Comparison Level')
# plt.xticks(rotation=0)
# for i, v in enumerate(level_counts):
#     plt.text(i, v + 1, f"{v:.1f}%", ha='center')
# plt.tight_layout()
# plt.show()

# #  Pie Chart
# plt.figure(figsize=(6,6))
# colors = ['red', 'orange', 'skyblue']
# plt.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
# plt.title('Social Comparison Levels (Pie Chart)')
# plt.tight_layout()
# plt.show()


# # Count Plot 
# plt.figure(figsize=(6, 4))
# sns.countplot(data=df, x='social_comparison_level', order=['High', 'Medium', 'Low'],
#               palette=['red', 'orange', 'skyblue'])
# plt.title('Social Comparison Level Counts')
# plt.ylabel('Number of Students')
# plt.xlabel('Social Comparison Level')
# plt.tight_layout()
# plt.show()

# # histogram
# plt.figure(figsize=(6,4))
# sns.histplot(df['social_comparison'], bins=10, kde=True, color='purple')
# plt.title('Distribution of Social Comparison Scores')
# plt.xlabel('Social Comparison Score (1 = High SC, 5 = Low SC)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# # box plot
# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='Gender', y='social_comparison', palette='Set2')
# plt.title('Social Comparison Scores by Gender')
# plt.ylabel('Social Comparison Score')
# plt.tight_layout()
# plt.show()

# # Violin Plot
# plt.figure(figsize=(6,4))
# sns.violinplot(data=df, x='Gender', y='social_comparison', palette='Set2')
# plt.title('Social Comparison Distribution by Gender')
# plt.ylabel('Social Comparison Score')
# plt.tight_layout()
# plt.show()
# #  Box Plot by Department
# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='Department', y='social_comparison', palette='coolwarm')
# plt.title('Social Comparison Scores by Department')
# plt.ylabel('Social Comparison Score')
# plt.tight_layout()
# plt.show()

# # Violin Plot by Department
# plt.figure(figsize=(6,4))
# sns.violinplot(data=df, x='Department', y='social_comparison', palette='coolwarm')
# plt.title('Social Comparison Distribution by Department')
# plt.ylabel('Social Comparison Score')
# plt.tight_layout()
# plt.show()

# # Box Plot by Year
# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='Year', y='social_comparison', palette='Spectral')
# plt.title('Social Comparison Scores by Academic Year')
# plt.ylabel('Social Comparison Score')
# plt.tight_layout()
# plt.show()

# #  Violin Plot by Year
# plt.figure(figsize=(6,4))
# sns.violinplot(data=df, x='Year', y='social_comparison', palette='Spectral')
# plt.title('Social Comparison Distribution by Academic Year')
# plt.ylabel('Social Comparison Score')
# plt.tight_layout()
# plt.show()

# Countplot: Social Comparison Level by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='social_comparison_level', hue='Gender', order=['High', 'Medium', 'Low'],
              palette='Set2')
plt.title('Social Comparison Level by Gender')
plt.xlabel('Social Comparison Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Social Comparison by Gender – Pie Chart 
genders = df['Gender'].unique()
for gender in genders:
    subset = df[df['Gender'] == gender]
    dist = subset['social_comparison_level'].value_counts(normalize=True) * 100

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Social Comparison Distribution – {gender}')
    plt.tight_layout()
    plt.show()
# Social Comparison by Department – Bar Chart
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='social_comparison_level', hue='Department',
              order=['High', 'Medium', 'Low'], palette='coolwarm')
plt.title('Social Comparison Level by Department')
plt.xlabel('Social Comparison Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Social Comparison by Year – Bar Chart
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='social_comparison_level', hue='Year',
              order=['High', 'Medium', 'Low'], palette='Spectral')
plt.title('Social Comparison Level by Academic Year')
plt.xlabel('Social Comparison Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Pie Chart by Year or Department?
years = df['Year'].unique()
for year in years:
    subset = df[df['Year'] == year]
    dist = subset['social_comparison_level'].value_counts(normalize=True) * 100

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Social Comparison Distribution – {year} Year')
    plt.tight_layout()
    plt.show()

# # One-hot encode social_comparison_level
# one_hot = pd.get_dummies(df['social_comparison_level'], prefix='social_comparison')

# # Convert True/False to 1/0 explicitly (in case it's boolean)
# one_hot = one_hot.astype(int)

# # Merge into original DataFrame
# df = pd.concat([df, one_hot], axis=1)

# # Rename columns exactly as you want
# df = df.rename(columns={
#     'social_comparison_High': 'high_social_comparison',
#     'social_comparison_Medium': 'medium_social_comparision',
#     'social_comparison_Low': 'low_social_comparision'
# })

# # Save as CSV
# df.to_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\social_comparison_encoded.csv", index=False)
