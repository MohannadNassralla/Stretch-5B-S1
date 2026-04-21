import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]

def load_telecom_data(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification."""
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES]
    y = df['churned']
    return X,y

X, y = load_telecom_data() 

# 1. Define the Parameter Grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# 2. Setup Inner CV
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Execute GridSearchCV
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid_rf, 
    scoring='f1', 
    cv=cv_inner, 
    n_jobs=-1
)
grid_search.fit(X, y)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Inner F1 Score: {grid_search.best_score_:.4f}")

# 4. Visualization: Heatmap
results_df = pd.DataFrame(grid_search.cv_results_)
# Fix min_samples_split at the best value to create a 2D grid
best_split = grid_search.best_params_['min_samples_split']
subset = results_df[results_df['param_min_samples_split'] == best_split]

pivot_table = subset.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".3f")
plt.title(f'RF F1 Score: Depth vs Estimators (min_split={best_split})')
plt.ylabel('Max Depth')
plt.xlabel('N Estimators')
plt.show()

# 1. Setup Outer CV (different random_state than inner)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

def run_nested_comparison(model, grid, X, y):
    # Inner Loop
    gs = GridSearchCV(model, grid, scoring='f1', cv=cv_inner, n_jobs=-1)
    
    # Outer Loop: Estimates the "true" performance
    outer_scores = cross_val_score(gs, X, y, scoring='f1', cv=cv_outer)
    
    # Get the average inner score for comparison
    gs.fit(X, y)
    return gs.best_score_, outer_scores.mean()

# Random Forest Comparison
rf_inner, rf_outer = run_nested_comparison(
    RandomForestClassifier(class_weight='balanced', random_state=42), 
    param_grid_rf, X, y
)

# Decision Tree Comparison
param_grid_dt = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}
dt_inner, dt_outer = run_nested_comparison(
    DecisionTreeClassifier(class_weight='balanced', random_state=42), 
    param_grid_dt, X, y
)

# 2. Report Results
comparison_data = {
    "Metric": ["Inner best_score_", "Outer nested CV score", "Gap (Bias)"],
    "Random Forest": [rf_inner, rf_outer, rf_inner - rf_outer],
    "Decision Tree": [dt_inner, dt_outer, dt_inner - dt_outer]
}
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)