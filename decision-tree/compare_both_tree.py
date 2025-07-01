import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load dataset ===
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# === Feature selection ===
features = [
    'social_comparison', 'peer_pressure', 'academic_burnout',
    'life_satisfaction', 'loneliness', 'perceived_support',
    'Gender', 'Department', 'Year', 'User_Type', 'heavy_user', 'active_user'
]
target = 'Cluster'  # Your cluster labels column

# Drop rows with missing target
df = df.dropna(subset=[target])

# Encode categorical features
categorical_cols = ['Gender', 'Department', 'Year', 'User_Type']
df_encoded = df.copy()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target if not numeric
if df_encoded[target].dtype == 'object':
    le_target = LabelEncoder()
    df_encoded[target] = le_target.fit_transform(df_encoded[target])
    cluster_labels = le_target.classes_
else:
    le_target = None
    cluster_labels = np.unique(df_encoded[target])

X = df_encoded[features]
y = df_encoded[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Helper function for training and evaluation ===
def train_evaluate_model(clf, param_grid, model_name):
    print(f"\n===== Training {model_name} =====")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    print(f"Best parameters ({model_name}): {grid_search.best_params_}")
    print(f"Best cross-val macro F1 ({model_name}): {grid_search.best_score_:.4f}")
    
    y_pred = best_clf.predict(X_test)
    print(f"\nClassification Report ({model_name}):\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(cluster_labels),
                yticklabels=sorted(cluster_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()
    
    # Feature importances plot
    importances = best_clf.feature_importances_
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fi_df['Importance'] = fi_df['Importance'].round(4)
    print(f"\nTop Feature Importances ({model_name}):\n", fi_df)
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='mako' if model_name=='Decision Tree' else 'viridis')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f"outputs/feature_importance_{model_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()
    
    # Visualize tree for decision tree only
    if model_name == 'Decision Tree':
        plt.figure(figsize=(24,16))
        plot_tree(
            best_clf,
            feature_names=X.columns,
            class_names=[str(c) for c in sorted(np.unique(y))],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f"Decision Tree - Cluster Prediction")
        plt.tight_layout()
        plt.savefig(f"outputs/cluster_decision_tree.png", dpi=300)
        plt.show()
    
    return best_clf, y_pred

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# === Decision Tree ===
dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 5]
}
best_dt_clf, y_pred_dt = train_evaluate_model(dt_clf, dt_param_grid, "Decision Tree")

# === Random Forest ===
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
rf_param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 5],
    'n_estimators': [100, 200]
}
best_rf_clf, y_pred_rf = train_evaluate_model(rf_clf, rf_param_grid, "Random Forest")

# === Mapping cluster predictions to FOMO levels ===
cluster_to_fomo_label = {
    1: 'Medium FOMO',
    2: 'High FOMO',
    3: 'Low FOMO'
}

def map_predictions_to_fomo(y_pred):
    return pd.Series(y_pred).map(cluster_to_fomo_label)

predicted_fomo_dt = map_predictions_to_fomo(y_pred_dt)
predicted_fomo_rf = map_predictions_to_fomo(y_pred_rf)

print("\nDecision Tree predicted FOMO distribution:")
print(predicted_fomo_dt.value_counts())

print("\nRandom Forest predicted FOMO distribution:")
print(predicted_fomo_rf.value_counts())

# Binary FOMO conversion function
def to_binary_fomo(label):
    return 'FOMO' if label in ['High FOMO', 'Medium FOMO'] else 'No FOMO'

predicted_binary_dt = predicted_fomo_dt.apply(to_binary_fomo)
predicted_binary_rf = predicted_fomo_rf.apply(to_binary_fomo)

print("\nDecision Tree predicted binary FOMO counts:")
print(predicted_binary_dt.value_counts())

print("\nRandom Forest predicted binary FOMO counts:")
print(predicted_binary_rf.value_counts())

# === Plot comparison charts ===
import matplotlib.pyplot as plt
import numpy as np

# FOMO levels comparison
labels = ['High FOMO', 'Medium FOMO', 'Low FOMO']
dt_counts = [predicted_fomo_dt.value_counts().get(l, 0) for l in labels]
rf_counts = [predicted_fomo_rf.value_counts().get(l, 0) for l in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, dt_counts, width, label='Decision Tree')
rects2 = ax.bar(x + width/2, rf_counts, width, label='Random Forest')

ax.set_ylabel('Number of Students')
ax.set_title('Predicted FOMO Level Distribution Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0,3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("outputs/fomo_level_distribution_comparison.png", dpi=300)
plt.show()

# Binary FOMO comparison
binary_labels = ['FOMO', 'No FOMO']
dt_binary_counts = [predicted_binary_dt.value_counts().get(l, 0) for l in binary_labels]
rf_binary_counts = [predicted_binary_rf.value_counts().get(l, 0) for l in binary_labels]

fig, ax = plt.subplots(figsize=(6,4))
rects1 = ax.bar(np.arange(len(binary_labels)) - width/2, dt_binary_counts, width, label='Decision Tree')
rects2 = ax.bar(np.arange(len(binary_labels)) + width/2, rf_binary_counts, width, label='Random Forest')

ax.set_ylabel('Number of Students')
ax.set_title('Predicted Binary FOMO Distribution Comparison')
ax.set_xticks(np.arange(len(binary_labels)))
ax.set_xticklabels(binary_labels)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0,3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("outputs/binary_fomo_distribution_comparison.png", dpi=300)
plt.show()
