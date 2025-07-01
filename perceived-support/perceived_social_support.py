import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("perceived_support_encoded.csv")

# Classification function for perceived social support
def classify_support(value):
    if value <= 2.5:
        return "Strong Support"
    elif value <= 3.5:
        return "Moderate Support"
    else:
        return "Low Support"

# Apply classification
df['perceived_support_level'] = df['perceived_support'].apply(classify_support)

# Count and percentage
support_counts = df['perceived_support_level'].value_counts(normalize=True) * 100
support_counts = support_counts.reindex(['Strong Support', 'Moderate Support', 'Low Support']).fillna(0)

# # Bar Chart
# plt.figure(figsize=(6,4))
# support_counts.plot(kind='bar', color=['skyblue', 'orange', 'red'])
# plt.title('Perceived Social Support Levels (Bar Chart)')
# plt.ylabel('Percentage (%)')
# plt.xlabel('Support Level')
# plt.xticks(rotation=0)
# for i, v in enumerate(support_counts):
#     plt.text(i, v + 1, f"{v:.1f}%", ha='center')
# plt.tight_layout()
# plt.show()

# # Pie Chart
# plt.figure(figsize=(6,6))
# colors = ['skyblue', 'orange', 'red']
# plt.pie(support_counts, labels=support_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
# plt.title('Perceived Social Support Levels (Pie Chart)')
# plt.tight_layout()
# plt.show()

# # Count Plot
# plt.figure(figsize=(6, 4))
# sns.countplot(data=df, x='perceived_support_level', 
#               order=['Strong Support', 'Moderate Support', 'Low Support'], 
#               palette=['skyblue', 'orange', 'red'])
# plt.title('Support Level Counts')
# plt.ylabel('Number of Students')
# plt.xlabel('Support Level')
# plt.tight_layout()
# plt.show()

# # Distribution histogram + KDE plot
# plt.figure(figsize=(6,4))
# sns.histplot(df['perceived_support'], bins=10, kde=True, color='purple')
# plt.title('Distribution of Perceived Social Support Scores')
# plt.xlabel('Support Score (1 = Low, 5 = High)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# # Box and Violin Plots by Gender, Department, Year
# for cat in ['Gender', 'Department', 'Year']:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(data=df, x=cat, y='perceived_support', palette='Set2')
#     plt.title(f'Perceived Support by {cat}')
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(6,4))
#     sns.violinplot(data=df, x=cat, y='perceived_support', palette='Set2')
#     plt.title(f'Perceived Support Distribution by {cat}')
#     plt.tight_layout()
#     plt.show()

# # Countplot: Support Level by Gender
# plt.figure(figsize=(6, 4))
# sns.countplot(data=df, x='perceived_support_level', hue='Gender', 
#               order=['Strong Support', 'Moderate Support', 'Low Support'], palette='Set2')
# plt.title('Support Level by Gender')
# plt.xlabel('Support Level')
# plt.ylabel('Number of Students')
# plt.tight_layout()
# plt.show()

# # Pie Chart by Gender
# genders = df['Gender'].unique()
# for gender in genders:
#     subset = df[df['Gender'] == gender]
#     dist = subset['perceived_support_level'].value_counts(normalize=True) * 100
#     dist = dist.reindex(['Strong Support', 'Moderate Support', 'Low Support']).fillna(0)

#     plt.figure(figsize=(5,5))
#     plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
#             colors=['skyblue', 'orange', 'red'], startangle=140)
#     plt.title(f'Support Distribution – {gender}')
#     plt.tight_layout()
#     plt.show()

# # Bar Chart by Department
# plt.figure(figsize=(7, 4))
# sns.countplot(data=df, x='perceived_support_level', hue='Department',
#               order=['Strong Support', 'Moderate Support', 'Low Support'], palette='coolwarm')
# plt.title('Support Level by Department')
# plt.xlabel('Support Level')
# plt.ylabel('Number of Students')
# plt.tight_layout()
# plt.show()

# # Bar Chart by Year
# plt.figure(figsize=(7, 4))
# sns.countplot(data=df, x='perceived_support_level', hue='Year',
#               order=['Strong Support', 'Moderate Support', 'Low Support'], palette='Spectral')
# plt.title('Support Level by Academic Year')
# plt.xlabel('Support Level')
# plt.ylabel('Number of Students')
# plt.tight_layout()
# plt.show()

# # Pie Chart by Year
# years = df['Year'].unique()
# for year in years:
#     subset = df[df['Year'] == year]
#     dist = subset['perceived_support_level'].value_counts(normalize=True) * 100
#     dist = dist.reindex(['Strong Support', 'Moderate Support', 'Low Support']).fillna(0)

#     plt.figure(figsize=(5,5))
#     plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
#             colors=['skyblue', 'orange', 'red'], startangle=140)
#     plt.title(f'Support Distribution – {year} Year')
#     plt.tight_layout()
#     plt.show()

# # Normalize label to lowercase + replace spaces with underscores for column naming
# df['perceived_support_level'] = df['perceived_support_level'].str.lower().str.replace(" ", "_")

# # One-hot encode the support levels
# one_hot = pd.get_dummies(df['perceived_support_level'], prefix='support')
# one_hot = one_hot.astype(int)  #  ensures values are 1 and 0, not True/False

# # Rename columns to final format
# one_hot = one_hot.rename(columns={
#     'support_high_support': 'high_support',
#     'support_moderate_support': 'moderate_support',
#     'support_low_support': 'low_support'
# })

# # Merge into original dataframe
# df = pd.concat([df, one_hot], axis=1)

# # Export to CSV
# df.to_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\perceived_support_encoded.csv", index=False)
