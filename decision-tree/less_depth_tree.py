# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load dataset (use the dataset that includes FOMO_Level column)
# df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# # =====================
# # Step 1: Feature selection
# # =====================
# features = [
#     'social_comparison', 'peer_pressure', 'academic_burnout',
#     'life_satisfaction', 'loneliness', 'perceived_support',
#     'Gender', 'Department', 'Year', 'User_Type', 'heavy_user', 'active_user'
# ]
# target = 'FOMO_Level'

# # Drop rows with missing FOMO_Level if any
# df = df.dropna(subset=[target])

# # =====================
# # Step 2: Encode categorical variables
# # =====================
# categorical_cols = ['Gender', 'Department', 'Year', 'User_Type']
# df_encoded = df.copy()
# le_dict = {}

# for col in categorical_cols:
#     le = LabelEncoder()
#     df_encoded[col] = le.fit_transform(df[col])
#     le_dict[col] = le

# # =====================
# # Step 3: Train-test split
# # =====================
# X = df_encoded[features]
# y = df_encoded[target]

# # Encode the target labels (High/Moderate/Low FOMO)
# le_target = LabelEncoder()
# y = le_target.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # =====================
# # Step 4: Train decision tree
# # =====================
# clf = DecisionTreeClassifier(max_depth=5, random_state=42)
# clf.fit(X_train, y_train)

# # =====================
# # Step 5: Evaluation
# # =====================
# y_pred = clf.predict(X_test)
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # =====================
# # Step 6: Visualize the tree
# # =====================
# plt.figure(figsize=(18, 10))
# plot_tree(clf, feature_names=X.columns, class_names=le_target.classes_, filled=True, rounded=True)
# plt.title("Decision Tree for FOMO Level Prediction")
# plt.tight_layout()
# plt.savefig("fomo_decision_tree.png")
# plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import graphviz
from sklearn.tree import export_graphviz


# Load dataset
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# Step 1: Feature selection
features = [
    'social_comparison', 'peer_pressure', 'academic_burnout',
    'life_satisfaction', 'loneliness', 'perceived_support',
    'Gender', 'Department', 'Year', 'User_Type', 'heavy_user', 'active_user'
]
target = 'FOMO_Level'

# Drop rows with missing target
df = df.dropna(subset=[target])

# Step 2: Encode categorical variables
categorical_cols = ['Gender', 'Department', 'Year', 'User_Type']
df_encoded = df.copy()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
df_encoded[target] = le_target.fit_transform(df_encoded[target])

X = df_encoded[features]
y = df_encoded[target]

# Step 3: Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Define Decision Tree with class_weight balanced
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Step 4: Train a shallow Decision Tree directly
param_grid = {
    'max_depth': [1, 2, 3],  # Try depths 3–5 only
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 3]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    clf,
    param_grid,
    scoring='f1_macro',  # Use macro F1 to balance classes
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-val macro F1: {grid_search.best_score_:.4f}")

# Step 6: Train best estimator on full training data
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Step 7: Evaluate on test set
y_pred = best_clf.predict(X_test)

print("\nClassification Report on Test Set:\n")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print("Confusion Matrix on Test Set:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Feature importance
importances = best_clf.feature_importances_
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:\n", fi_df)

plt.figure(figsize=(10,6))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis', hue=None)
plt.title('Feature Importance from Decision Tree')
plt.tight_layout()
plt.show()

# Step 9: Visualize the decision tree
plt.figure(figsize=(24, 16))  # Larger figure size
plot_tree(
    best_clf,
    feature_names=X.columns,
    class_names=le_target.classes_,
    filled=True,
    rounded=True,
    fontsize=10  # Increase font size for readability
)
plt.title("Optimized Decision Tree for FOMO Level Prediction")
plt.tight_layout()
plt.savefig("fomo_decision_tree_best_less_depth.png", dpi=300)  # Increase DPI for clarity
plt.show()

# plt.tight_layout()
# plt.savefig("fomo_decision_tree_best.png")
# plt.show()


dot_data = export_graphviz(
    best_clf,
    out_file=None,
    feature_names=X.columns,
    class_names=le_target.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("fomo_decision_tree_best_less_depth", format='pdf')  # Saves as PDF file
graph.view()  # Opens the PDF automatically (if supported)
