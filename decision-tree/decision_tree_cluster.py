import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ========== Step 0: Load Dataset ==========
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# ========== Step 1: Feature & Target ==========
features = [
    'social_comparison', 'peer_pressure', 'academic_burnout',
    'life_satisfaction', 'loneliness', 'perceived_support',
    'Gender', 'Department', 'Year', 'User_Type', 'heavy_user', 'active_user'
]
target = 'Cluster'

# Drop missing values in target
df = df.dropna(subset=[target])

# ========== Step 2: Encode Categorical Variables ==========
categorical_cols = ['Gender', 'Department', 'Year', 'User_Type']
df_encoded = df.copy()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode Cluster column if not numeric
if df_encoded[target].dtype == 'object':
    le_target = LabelEncoder()
    df_encoded[target] = le_target.fit_transform(df_encoded[target])
    cluster_labels = le_target.classes_
else:
    le_target = None
    cluster_labels = df_encoded[target].astype(str).unique()

X = df_encoded[features]
y = df_encoded[target]

# ========== Step 3: Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== Step 4: Decision Tree with GridSearch ==========
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    clf, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

print(f"\n Best Parameters: {grid_search.best_params_}")
print(f" Best cross-val macro F1: {grid_search.best_score_:.4f}")

# ========== Step 5: Evaluation ==========
report = classification_report(y_test, y_pred)
print("\n Classification Report:\n", report)

os.makedirs("outputs", exist_ok=True)
with open("outputs/cluster_decision_tree_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(cluster_labels),
            yticklabels=sorted(cluster_labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Decision Tree (Cluster Prediction)')
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_tree.png", dpi=300)
plt.show()

# ========== Step 6: Feature Importance ==========
importances = best_clf.feature_importances_
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.round(importances, 4)
}).sort_values(by='Importance', ascending=False)

print("\nTop Feature Importances:\n", fi_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='mako')
plt.title('Feature Importance - Decision Tree (Cluster Prediction)')
plt.tight_layout()
plt.savefig("outputs/feature_importance_tree.png", dpi=300)
plt.show()

# ========== Step 7: Tree Visualization ==========
plt.figure(figsize=(24, 16))
plot_tree(
    best_clf,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(np.unique(y))],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Cluster Prediction")
plt.tight_layout()
plt.savefig("outputs/cluster_decision_tree.png", dpi=300)
plt.show()

# ========== Step 8: Inferred FOMO Labels ==========
cluster_to_fomo_label = {
    1: 'Medium FOMO',     # Peer-Driven Burnouts
    2: 'High FOMO',       # Comparison FOMO
    3: 'Low FOMO'         # Balanced but Exhausted
}
df['FOMO_Level_Inferred'] = df['Cluster'].map(cluster_to_fomo_label)

fomo_counts = df['FOMO_Level_Inferred'].value_counts()
print("\n FOMO Level Distribution:")
print(fomo_counts)

# ========== Step 9: Binary FOMO Classification ==========
def binary_fomo(label):
    return 'FOMO' if label in ['High FOMO', 'Medium FOMO'] else 'No FOMO'

df['FOMO_Binary'] = df['FOMO_Level_Inferred'].apply(binary_fomo)
binary_counts = df['FOMO_Binary'].value_counts()
print("\n FOMO vs No FOMO Distribution:")
print(binary_counts)

# Optional plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='FOMO_Binary', palette='Set2')
plt.title('FOMO vs No FOMO Students')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig("outputs/fomo_vs_no_fomo_countplot.png", dpi=300)
plt.show()
