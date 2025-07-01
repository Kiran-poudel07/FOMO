import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (adjust path)
df = pd.read_csv("peer_pressure_encoded.csv")

# Define classification function for peer pressure levels
def classify_pp(value):
    if value <= 2.0:
        return "High Pressure"
    elif value <= 3.5:
        return "Medium Pressure"
    else:
        return "Low Pressure"


df['peer_pressure_level'] = df['peer_pressure'].apply(classify_pp)

# Count and percentage for peer pressure levels
level_counts = df['peer_pressure_level'].value_counts(normalize=True) * 100
level_counts = level_counts.reindex(['High Pressure', 'Medium Pressure', 'Low Pressure']).fillna(0)

# Bar Chart of peer pressure levels (%)
plt.figure(figsize=(6,4))
level_counts.plot(kind='bar', color=['red', 'orange', 'skyblue'])
plt.title('Peer Pressure Levels (Bar Chart)')
plt.ylabel('Percentage (%)')
plt.xlabel('Peer Pressure Level')
plt.xticks(rotation=0)
for i, v in enumerate(level_counts):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()

# Pie Chart of peer pressure levels
plt.figure(figsize=(6,6))
colors = ['red', 'orange', 'skyblue']
plt.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Peer Pressure Levels (Pie Chart)')
plt.tight_layout()
plt.show()

# Count Plot of peer pressure levels
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='peer_pressure_level', order=['High Pressure', 'Medium Pressure', 'Low Pressure'],
              palette=['red', 'orange', 'skyblue'])
plt.title('Peer Pressure Level Counts')
plt.ylabel('Number of Students')
plt.xlabel('Peer Pressure Level')
plt.tight_layout()
plt.show()

# Distribution histogram + KDE plot
plt.figure(figsize=(6,4))
sns.histplot(df['peer_pressure'], bins=10, kde=True, color='purple')
plt.title('Distribution of Peer Pressure Scores')
plt.xlabel('Peer Pressure Score (1 = Low, 5 = High)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Box Plot by Gender
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Gender', y='peer_pressure', palette='Set2')
plt.title('Peer Pressure Scores by Gender')
plt.ylabel('Peer Pressure Score')
plt.tight_layout()
plt.show()

# Violin Plot by Gender
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Gender', y='peer_pressure', palette='Set2')
plt.title('Peer Pressure Distribution by Gender')
plt.ylabel('Peer Pressure Score')
plt.tight_layout()
plt.show()

# Box Plot by Department
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Department', y='peer_pressure', palette='coolwarm')
plt.title('Peer Pressure Scores by Department')
plt.ylabel('Peer Pressure Score')
plt.tight_layout()
plt.show()

# Violin Plot by Department
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Department', y='peer_pressure', palette='coolwarm')
plt.title('Peer Pressure Distribution by Department')
plt.ylabel('Peer Pressure Score')
plt.tight_layout()
plt.show()

# Box Plot by Year
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Year', y='peer_pressure', palette='Spectral')
plt.title('Peer Pressure Scores by Academic Year')
plt.ylabel('Peer Pressure Score')
plt.tight_layout()
plt.show()

# Violin Plot by Year
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='Year', y='peer_pressure', palette='Spectral')
plt.title('Peer Pressure Distribution by Academic Year')
plt.ylabel('Peer Pressure Score')
plt.tight_layout()
plt.show()

# Countplot: Peer Pressure Level by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='peer_pressure_level', hue='Gender', order=['High Pressure', 'Medium Pressure', 'Low Pressure'],
              palette='Set2')
plt.title('Peer Pressure Level by Gender')
plt.xlabel('Peer Pressure Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Peer Pressure by Gender – Pie Chart
genders = df['Gender'].unique()
for gender in genders:
    subset = df[df['Gender'] == gender]
    dist = subset['peer_pressure_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['High Pressure', 'Medium Pressure', 'Low Pressure']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Peer Pressure Distribution – {gender}')
    plt.tight_layout()
    plt.show()

# Peer Pressure by Department – Bar Chart
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='peer_pressure_level', hue='Department',
              order=['High Pressure', 'Medium Pressure', 'Low Pressure'], palette='coolwarm')
plt.title('Peer Pressure Level by Department')
plt.xlabel('Peer Pressure Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Peer Pressure by Year – Bar Chart
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='peer_pressure_level', hue='Year',
              order=['High Pressure', 'Medium Pressure', 'Low Pressure'], palette='Spectral')
plt.title('Peer Pressure Level by Academic Year')
plt.xlabel('Peer Pressure Level')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Pie Chart by Year
years = df['Year'].unique()
for year in years:
    subset = df[df['Year'] == year]
    dist = subset['peer_pressure_level'].value_counts(normalize=True) * 100
    dist = dist.reindex(['High Pressure', 'Medium Pressure', 'Low Pressure']).fillna(0)

    plt.figure(figsize=(5,5))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%',
            colors=['red', 'orange', 'skyblue'], startangle=140)
    plt.title(f'Peer Pressure Distribution – {year} Year')
    plt.tight_layout()
    plt.show()

#     # Normalize label to lowercase
# df['peer_pressure_level'] = df['peer_pressure_level'].str.lower().str.replace(" pressure", "")

# # One-hot encode the peer_pressure_level column
# one_hot = pd.get_dummies(df['peer_pressure_level'], prefix='peer_pressure')

# # Convert boolean to integer (1/0)
# one_hot = one_hot.astype(int)

# # Rename columns as required
# df = pd.concat([df, one_hot], axis=1)
# df = df.rename(columns={
#     'peer_pressure_low': 'low_peer_pressure',
#     'peer_pressure_medium': 'medium_peer_pressure',
#     'peer_pressure_high': 'high_peer_pressure'
# })

# # Save to CSV
# df.to_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\peer_pressure_encoded.csv", index=False)

