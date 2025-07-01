import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

# ========== Step 4: Random Forest with GridSearch ==========
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10, None],
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

print(f"\n Best Parameters (Random Forest): {grid_search.best_params_}")
print(f" Best cross-val macro F1 (Random Forest): {grid_search.best_score_:.4f}")

# ========== Step 5: Evaluation ==========
report = classification_report(y_test, y_pred)
print("\n Classification Report (Random Forest):\n", report)

os.makedirs("outputs", exist_ok=True)
with open("outputs/cluster_random_forest_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=sorted(cluster_labels),
            yticklabels=sorted(cluster_labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest (Cluster Prediction)')
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_rf.png", dpi=300)
plt.show()

# ========== Step 6: Feature Importance ==========
importances = best_clf.feature_importances_
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.round(importances, 4)
}).sort_values(by='Importance', ascending=False)

print("\n Top Feature Importances (Random Forest):\n", fi_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Random Forest (Cluster Prediction)')
plt.tight_layout()
plt.savefig("outputs/feature_importance_rf.png", dpi=300)
plt.show()

# ========== Step 7: FOMO Level Inference ==========
cluster_to_fomo_label = {
    1: 'Medium FOMO',     # Peer-Driven Burnouts
    2: 'High FOMO',       # Comparison FOMO
    3: 'Low FOMO'         # Balanced but Exhausted
}

df['FOMO_Level_Inferred_RF'] = df['Cluster'].map(cluster_to_fomo_label)

fomo_counts_rf = df['FOMO_Level_Inferred_RF'].value_counts()
print("\n FOMO Level Distribution (Random Forest Mapping):")
print(fomo_counts_rf)

def binary_fomo(label):
    return 'FOMO' if label in ['High FOMO', 'Medium FOMO'] else 'No FOMO'

df['FOMO_Binary_RF'] = df['FOMO_Level_Inferred_RF'].apply(binary_fomo)
binary_counts_rf = df['FOMO_Binary_RF'].value_counts()
print("\n FOMO vs No FOMO Distribution (Random Forest Mapping):")
print(binary_counts_rf)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='FOMO_Binary_RF', palette='Set1')
plt.title('FOMO vs No FOMO Students (Random Forest)')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig("outputs/fomo_vs_no_fomo_rf_countplot.png", dpi=300)
plt.show()
