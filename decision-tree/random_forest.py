import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Step 4: Random Forest with class_weight
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Step 5: Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-val macro F1: {grid_search.best_score_:.4f}")

# Step 6: Evaluate best RF model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("\nClassification Report on Test Set:\n")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print("Confusion Matrix on Test Set:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest - Confusion Matrix')
plt.show()

# Step 7: Feature importance
importances = best_rf.feature_importances_
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:\n", fi_df)

plt.figure(figsize=(10,6))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='crest', hue=None)
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()
