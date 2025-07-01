# import pandas as pd
# import gower
# import numpy as np

# # Load the processed Likert data
# df = pd.read_csv("processed_fomo_likert.csv")

# # Drop ID column for distance calculation
# likert_only = df.drop(columns=["P_ID"])

# # 🔧 Fix: Convert to float to avoid dtype casting error
# likert_only = likert_only.astype(float)

# # Compute Gower distance matrix
# gower_dist = gower.gower_matrix(likert_only)

# # Optional: Save matrix if needed
# np.savetxt("gower_distance_matrix.csv", gower_dist, delimiter=",")



# print("Gower distance matrix computed and saved.")

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import gower
import numpy as np
# Step 1: Load cleaned dataset (Q7–Q36 only)
df = pd.read_csv("processed_fomo_likert.csv")


# Convert all int columns to float to avoid dtype issues with gower
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.integer):
        df[col] = df[col].astype(float)


# Calculate Gower distance matrix (n x n)
gower_dist_square = gower.gower_matrix(df)

# Convert the squareform matrix to condensed (1D format)
gower_dist_condensed = squareform(gower_dist_square, checks=False)

# Step 3: Perform hierarchical clustering (Ward’s method)
linked = linkage(gower_dist_condensed, method='ward')

# Step 4: Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(
    linked,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,                   # number of leaf nodes to display
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True
)
plt.title("Hierarchical Clustering Dendrogram (Gower Distance)")
plt.xlabel("Sample Index or (Cluster Size)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
