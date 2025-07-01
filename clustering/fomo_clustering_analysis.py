# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from scipy.spatial.distance import squareform
# import gower

# # === Step 0: Load Original Dataset ===
# df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# # === Step 1: Subset Q7–Q36 for clustering and reverse code selected items ===

# # Extract Likert-scale columns Q7 to Q36 based on full verbose column names
# likert_questions = [col for col in df.columns if col.strip().startswith("Q") and col.strip().split(" ")[0][1:].isdigit() and 7 <= int(col.strip().split(" ")[0][1:]) <= 36]

# # Define reverse-coded questions by their question numbers
# reverse_q_nums = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,27,28,29,30,31]
# reverse_questions = [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in reverse_q_nums]

# # Copy and convert to float to avoid Gower casting errors
# cleaned_df = df[likert_questions].copy().astype(float)

# # Reverse code specified items: 1 = Strongly Agree to 5 = Strongly Disagree
# for q in reverse_questions:
#     cleaned_df[q] = cleaned_df[q].apply(lambda x: 6 - x if pd.notna(x) else x)

# # === Step 2: Compute Gower Distance and Perform Clustering ===
# gower_distance = gower.gower_matrix(cleaned_df)
# gower_dist = squareform(gower_distance, checks=False)

# linked = linkage(gower_dist, method='ward')

# # Decide number of clusters
# n_clusters = 3
# cluster_labels = fcluster(linked, t=n_clusters, criterion='maxclust')

# # Add cluster labels
# df['Cluster'] = cluster_labels
# cleaned_df['Cluster'] = cluster_labels

# # === Step 3: Profile clusters by construct (mean score) ===

# # Construct definitions based on question numbers
# constructs = {
#     'Social_Comparison': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(7, 12)],
#     'Peer_Pressure': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(12, 17)],
#     'Academic_Burnout': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(17, 22)],
#     'Life_Satisfaction': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(22, 27)],
#     'Loneliness': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(27, 32)],
#     'Perceived_Support': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(32, 37)],
# }

# construct_means = {}
# for name, questions in constructs.items():
#     construct_means[name] = cleaned_df[questions].groupby(cleaned_df['Cluster']).mean().mean(axis=1)

# cluster_profile = pd.DataFrame(construct_means)
# print("\n=== Construct Means Per Cluster ===")
# print(cluster_profile)

# # === Step 4: Demographic Profiling ===
# demo_counts = df.groupby(['Cluster', 'Gender', 'Department', 'User_Type']).size().unstack(fill_value=0)
# print("\n=== Demographics Per Cluster ===")
# print(demo_counts)

# # === Step 5: Visualizations ===

# # Cluster Sizes
# plt.figure(figsize=(6, 4))
# sns.countplot(data=df, x='Cluster', palette='Set2')
# plt.title("Cluster Sizes")
# plt.xlabel("Cluster")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()

# # Heatmap of average construct scores
# plt.figure(figsize=(8, 5))
# sns.heatmap(cluster_profile, annot=True, cmap='coolwarm')
# plt.title("Average Construct Scores per Cluster")
# plt.ylabel("Cluster")
# plt.xlabel("Construct")
# plt.tight_layout()
# plt.show()

# # Pie charts for User_Type per cluster
# for i in range(1, n_clusters + 1):
#     cluster_data = df[df['Cluster'] == i]
#     plt.figure()
#     cluster_data['User_Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
#     plt.title(f'User Type Distribution - Cluster {i}')
#     plt.ylabel("")
#     plt.tight_layout()
#     plt.show()

# # === Step 6: Save output ===
# df.to_csv("fomo_clustered_output.csv", index=False)
# print("Clustered data saved to fomo_clustered_output.csv")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import gower

# === Step 0: Load Original Dataset ===
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# === Step 1: Subset Q7–Q36 for clustering and reverse code selected items ===

# Extract Likert-scale columns Q7 to Q36 based on full verbose column names
likert_questions = [
    col for col in df.columns
    if col.strip().startswith("Q") and col.strip().split(" ")[0][1:].isdigit()
    and 7 <= int(col.strip().split(" ")[0][1:]) <= 36
]

# Define reverse-coded questions by their question numbers
reverse_q_nums = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,27,28,29,30,31]
reverse_questions = [
    col for col in likert_questions
    if int(col.strip().split(" ")[0][1:]) in reverse_q_nums
]

# Copy and convert to float to avoid Gower casting errors
cleaned_df = df[likert_questions].copy().astype(float)

# Reverse code specified items: 1 = Strongly Agree to 5 = Strongly Disagree
for q in reverse_questions:
    cleaned_df[q] = cleaned_df[q].apply(lambda x: 6 - x if pd.notna(x) else x)

# === Step 2: Compute Gower Distance and Perform Clustering ===
gower_distance = gower.gower_matrix(cleaned_df)
gower_dist = squareform(gower_distance, checks=False)

linked = linkage(gower_dist, method='ward')

# Decide number of clusters
n_clusters = 3
cluster_labels = fcluster(linked, t=n_clusters, criterion='maxclust')

# Add cluster labels
df['Cluster'] = cluster_labels
cleaned_df['Cluster'] = cluster_labels

# === Step 3: Assign descriptive labels to each cluster ===
cluster_name_map = {
    1: 'Peer-Driven Burnouts',
    2: 'Comparison FOMO',
    3: 'Balanced but Exhausted'
}
df['FOMO_Profile_Label'] = df['Cluster'].map(cluster_name_map)

# === Step 4: Profile clusters by construct (mean score) ===
constructs = {
    'Social_Comparison': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(7, 12)],
    'Peer_Pressure': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(12, 17)],
    'Academic_Burnout': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(17, 22)],
    'Life_Satisfaction': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(22, 27)],
    'Loneliness': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(27, 32)],
    'Perceived_Support': [col for col in likert_questions if int(col.strip().split(" ")[0][1:]) in range(32, 37)],
}

construct_means = {}
for name, questions in constructs.items():
    construct_means[name] = cleaned_df[questions].groupby(cleaned_df['Cluster']).mean().mean(axis=1)

cluster_profile = pd.DataFrame(construct_means)

# Map cluster index to profile names for heatmap
cluster_profile.index = cluster_profile.index.map(cluster_name_map)

print("\n=== Construct Means Per Cluster ===")
print(cluster_profile)

# === Step 5: Demographic Profiling ===
demo_counts = df.groupby(['Cluster', 'Gender', 'Department', 'User_Type']).size().unstack(fill_value=0)
print("\n=== Demographics Per Cluster ===")
print(demo_counts)

# === Step 6: Visualizations ===

# 6.1 Cluster Sizes Bar Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='FOMO_Profile_Label', palette='Set2')
plt.title("Cluster Sizes by FOMO Profile")
plt.xlabel("FOMO Profile")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 6.2 Heatmap of Average Construct Scores
plt.figure(figsize=(8, 5))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm')
plt.title("Average Construct Scores per FOMO Profile")
plt.ylabel("FOMO Profile")
plt.xlabel("Construct")
plt.tight_layout()
plt.show()

# 6.3 Pie charts for User_Type per cluster
for i in range(1, n_clusters + 1):
    cluster_data = df[df['Cluster'] == i]
    plt.figure()
    cluster_data['User_Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'User Type Distribution - {cluster_name_map[i]}')
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# === Step 7: Save Output ===
# df.to_csv("fomo_clustered_output.csv", index=False)
# print(" Clustered data saved to fomo_clustered_output.csv")
