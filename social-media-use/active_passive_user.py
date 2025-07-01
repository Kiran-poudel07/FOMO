import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("social_media_users_classified.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Classifier for Active vs Passive
def classify_social_media_user(row):
    active_criteria = 0
    
    # Time spent
    if row['Q3 = On average, how much time do you spend on social media daily?'] in ['3-6 hours', 'More than 7 hours']:
        active_criteria += 1
    
    # Reason for use
    if row['Q4 = What is the main reason for using social media?'] in ['To connect with other(social interaction)', 'To escape myself creatively(self-expression)']:
        active_criteria += 1

    # Post count
    post_val = row['Q6 = On average , how many photos ,videos,posts or reels do you share on social media montly?']
    if isinstance(post_val, str) and "More than 10" in post_val:
        active_criteria += 1
    elif isinstance(post_val, (int, float)) and post_val >= 5:
        active_criteria += 1
    
    return "Active" if active_criteria >= 2 else "Passive"

# Apply classification
df['User_Type'] = df.apply(classify_social_media_user, axis=1)
df['active_user'] = df['User_Type'].map({'Active': 1, 'Passive': 0})  # Binary version

# -----------------------------------
# Gender-wise breakdown
gender_counts = df.groupby(['Gender', 'User_Type']).size().unstack().fillna(0)
gender_percent = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
print("\n Gender-wise Breakdown (%):\n", gender_percent.round(2))

# -----------------------------------
# Department-wise breakdown
dept_counts = df.groupby(['Department', 'User_Type']).size().unstack().fillna(0)
dept_percent = dept_counts.div(dept_counts.sum(axis=1), axis=0) * 100
print("\n Department-wise Breakdown (%):\n", dept_percent.round(2))

# ----------------------------------
# Year-wise breakdown
year_counts = df.groupby(['Year', 'User_Type']).size().unstack().fillna(0)
year_percent = year_counts.div(year_counts.sum(axis=1), axis=0) * 100
print("\n Year-wise Breakdown (%):\n", year_percent.round(2))

# -----------------------------------
# Plotting

# Gender
gender_percent.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.title("Gender-wise: Active vs Passive Social Media Users")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="User Type")
plt.tight_layout()
plt.show()

# Department
dept_percent.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.title("Department-wise: Active vs Passive Social Media Users")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.legend(title="User Type")
plt.tight_layout()
plt.show()

# Year
year_percent.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.title("Year-wise: Active vs Passive Social Media Users")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="User Type")
plt.tight_layout()
plt.show()

# -----------------------------------
# # Optional: Export the modified dataset
# df.to_csv("social_media_users_classified.csv", index=False)
