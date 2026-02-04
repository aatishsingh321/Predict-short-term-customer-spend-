"""
================================================================================
MODEL TRAINING SCRIPT
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Train and evaluate regression models for 30-day customer spend prediction

Models Trained:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor

Evaluation Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

Input:  data/processed/customer_features.csv
Output: models/linear_regression.joblib
        models/random_forest.joblib
        models/xgboost_model.joblib
        models/model_comparison.csv
        plots/linear_regression_evaluation.png
        plots/random_forest_evaluation.png
        plots/xgboost_evaluation.png
        docs/05_model_evaluation.md
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Run: pip install xgboost")

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/Test split ratio (time-based)
TRAIN_RATIO = 0.80

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def print_metrics(metrics, dataset_name):
    """Print metrics in a formatted way"""
    print(f"  {dataset_name}:")
    print(f"    RMSE: ${metrics['RMSE']:.2f}")
    print(f"    MAE:  ${metrics['MAE']:.2f}")
    print(f"    R²:   {metrics['R2']:.4f}")


# ============================================================
# 1. LOAD DATA
# ============================================================
print_section("1. LOADING CUSTOMER FEATURES DATA")

# Load customer features
df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "customer_features.csv"))
print(f"Loaded dataset: {df.shape[0]} customers, {df.shape[1]} columns")

# Display target variable stats
TARGET = 'future_spend_30d'
print(f"\nTarget Variable ({TARGET}):")
print(f"  Mean:   ${df[TARGET].mean():.2f}")
print(f"  Median: ${df[TARGET].median():.2f}")
print(f"  Std:    ${df[TARGET].std():.2f}")
print(f"  Min:    ${df[TARGET].min():.2f}")
print(f"  Max:    ${df[TARGET].max():.2f}")


# ============================================================
# 2. DATA SPLITTING (TIME-BASED)
# ============================================================
print_section("2. DATA SPLITTING (TIME-BASED)")

print_subsection("2.1 Define Features and Target")

# Columns to exclude from features (identifiers, dates, target, leakage)
EXCLUDE_COLS = [
    'customer_id',              # Identifier
    'future_spend_30d',         # Target variable
    'future_transactions_30d',  # Leakage - future info
    'will_purchase_30d',        # Leakage - derived from future
    'first_purchase_date',      # Raw date column
    'transaction_id'            # Identifier (if present)
]

# Define feature columns
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

# Separate numeric and categorical features
categorical_cols = ['top_category', 'primary_store_id', 'primary_region', 
                    'loyalty_status', 'segment_id', 'preferred_time_of_day', 'preferred_day']
categorical_cols = [col for col in categorical_cols if col in feature_cols]

numeric_cols = [col for col in feature_cols if col not in categorical_cols]

print(f"Total features: {len(feature_cols)}")
print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")


print_subsection("2.2 Time-Based Train/Test Split")

# Convert first_purchase_date to datetime for sorting
df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'], errors='coerce')

# Sort by first_purchase_date (temporal ordering) - DO NOT SHUFFLE
df_sorted = df.sort_values('first_purchase_date', ascending=True).reset_index(drop=True)

print(f"Sorted {len(df_sorted)} customers by first_purchase_date")
print(f"  Earliest: {df_sorted['first_purchase_date'].min()}")
print(f"  Latest:   {df_sorted['first_purchase_date'].max()}")

# Calculate split index using TRAIN_RATIO
split_idx = int(len(df_sorted) * TRAIN_RATIO)

# Split temporally - earliest 80% for train, latest 20% for test
train_df = df_sorted.iloc[:split_idx].copy()
test_df = df_sorted.iloc[split_idx:].copy()

print(f"\nTrain set: {len(train_df)} customers (earliest {TRAIN_RATIO*100:.0f}%)")
print(f"Test set:  {len(test_df)} customers (latest {(1-TRAIN_RATIO)*100:.0f}%)")

# Get cutoff dates for validation
train_max_date = train_df['first_purchase_date'].max()
test_min_date = test_df['first_purchase_date'].min()

print(f"\nTemporal split boundary:")
print(f"  Train max date: {train_max_date}")
print(f"  Test min date:  {test_min_date}")

# Extract features (X) and target (y) AFTER temporal split
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()

# Fill remaining missing numeric values with 0
X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


print_subsection("2.3 Validate No Data Leakage")

# Validation checks
assert TARGET not in feature_cols, "Target variable in features!"
assert 'future_transactions_30d' not in feature_cols, "Future transactions in features!"
assert 'will_purchase_30d' not in feature_cols, "Future indicator in features!"
assert train_max_date < test_min_date, "Temporal split violation!"

train_customers = set(train_df['customer_id'].unique())
test_customers = set(test_df['customer_id'].unique())
assert len(train_customers.intersection(test_customers)) == 0, "Customer overlap!"

print("✅ All data leakage checks passed")


# ============================================================
# 3. PREPROCESSING
# ============================================================
print_section("3. PREPROCESSING")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit on training data only, transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed X_train shape: {X_train_processed.shape}")
print(f"Processed X_test shape:  {X_test_processed.shape}")

# Get feature names after preprocessing
try:
    feature_names_out = list(preprocessor.get_feature_names_out())
    print(f"Total features after encoding: {len(feature_names_out)}")
except:
    feature_names_out = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    print(f"Total features after encoding: {len(feature_names_out)}")


# ============================================================
# 4. MODEL TRAINING
# ============================================================
print_section("4. MODEL TRAINING")

# Store results for comparison
all_results = []
trained_models = {}

# ------------------------------
# 4.1 LINEAR REGRESSION (Baseline)
# ------------------------------
print_subsection("4.1 Linear Regression (Baseline)")

lr_model = LinearRegression()
lr_model.fit(X_train_processed, y_train)

# Predictions (clip to non-negative)
y_train_pred_lr = np.clip(lr_model.predict(X_train_processed), 0, None)
y_test_pred_lr = np.clip(lr_model.predict(X_test_processed), 0, None)

# Evaluate
lr_train_metrics = evaluate_model(y_train, y_train_pred_lr)
lr_test_metrics = evaluate_model(y_test, y_test_pred_lr)

print("\nLinear Regression Performance:")
print_metrics(lr_train_metrics, "Train")
print_metrics(lr_test_metrics, "Test")

# Check overfitting
overfit_ratio_lr = lr_train_metrics['RMSE'] / lr_test_metrics['RMSE']
print(f"\n  Train/Test RMSE Ratio: {overfit_ratio_lr:.3f}")
if overfit_ratio_lr < 0.8:
    print("  ⚠️ Possible underfitting")
elif overfit_ratio_lr > 1.2:
    print("  ⚠️ Possible overfitting")
else:
    print("  ✓ Good generalization")

trained_models['Linear Regression'] = lr_model
all_results.append({
    'Model': 'Linear Regression',
    'Train_RMSE': lr_train_metrics['RMSE'],
    'Test_RMSE': lr_test_metrics['RMSE'],
    'Train_MAE': lr_train_metrics['MAE'],
    'Test_MAE': lr_test_metrics['MAE'],
    'Train_R2': lr_train_metrics['R2'],
    'Test_R2': lr_test_metrics['R2']
})


# ------------------------------
# 4.2 RANDOM FOREST
# ------------------------------
print_subsection("4.2 Random Forest Regressor")

rf_model = RandomForestRegressor(
    n_estimators=200,           # Sufficiently large number of trees
    max_depth=12,               # Controlled depth to reduce overfitting
    min_samples_split=20,       # Minimum samples to split
    min_samples_leaf=10,        # Minimum samples per leaf
    max_features='sqrt',        # Feature subsampling
    random_state=RANDOM_STATE,  # Reproducibility
    n_jobs=-1                   # Use all cores
)
rf_model.fit(X_train_processed, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train_processed)
y_test_pred_rf = rf_model.predict(X_test_processed)

# Evaluate
rf_train_metrics = evaluate_model(y_train, y_train_pred_rf)
rf_test_metrics = evaluate_model(y_test, y_test_pred_rf)

print("\nRandom Forest Performance:")
print_metrics(rf_train_metrics, "Train")
print_metrics(rf_test_metrics, "Test")

# Check overfitting
overfit_ratio_rf = rf_train_metrics['RMSE'] / rf_test_metrics['RMSE']
print(f"\n  Train/Test RMSE Ratio: {overfit_ratio_rf:.3f}")
if overfit_ratio_rf < 0.8:
    print("  ⚠️ Possible underfitting")
elif overfit_ratio_rf > 1.2:
    print("  ⚠️ Possible overfitting")
else:
    print("  ✓ Good generalization")

trained_models['Random Forest'] = rf_model
all_results.append({
    'Model': 'Random Forest',
    'Train_RMSE': rf_train_metrics['RMSE'],
    'Test_RMSE': rf_test_metrics['RMSE'],
    'Train_MAE': rf_train_metrics['MAE'],
    'Test_MAE': rf_test_metrics['MAE'],
    'Train_R2': rf_train_metrics['R2'],
    'Test_R2': rf_test_metrics['R2']
})


# ------------------------------
# 4.3 XGBOOST
# ------------------------------
if XGBOOST_AVAILABLE:
    print_subsection("4.3 XGBoost Regressor")
    
    xgb_model = XGBRegressor(
        objective='reg:squarederror',  # Regression objective
        n_estimators=200,              # Number of trees
        max_depth=6,                   # Conservative depth
        learning_rate=0.05,            # Conservative learning rate
        subsample=0.8,                 # Row subsampling
        colsample_bytree=0.8,          # Column subsampling
        min_child_weight=10,           # Minimum sum of instance weight
        reg_alpha=0.1,                 # L1 regularization
        reg_lambda=1.0,                # L2 regularization
        random_state=RANDOM_STATE,     # Reproducibility
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_processed, y_train)
    
    # Predictions
    y_train_pred_xgb = xgb_model.predict(X_train_processed)
    y_test_pred_xgb = xgb_model.predict(X_test_processed)
    
    # Evaluate
    xgb_train_metrics = evaluate_model(y_train, y_train_pred_xgb)
    xgb_test_metrics = evaluate_model(y_test, y_test_pred_xgb)
    
    print("\nXGBoost Performance:")
    print_metrics(xgb_train_metrics, "Train")
    print_metrics(xgb_test_metrics, "Test")
    
    # Check overfitting
    overfit_ratio_xgb = xgb_train_metrics['RMSE'] / xgb_test_metrics['RMSE']
    print(f"\n  Train/Test RMSE Ratio: {overfit_ratio_xgb:.3f}")
    if overfit_ratio_xgb < 0.8:
        print("  ⚠️ Possible underfitting")
    elif overfit_ratio_xgb > 1.2:
        print("  ⚠️ Possible overfitting")
    else:
        print("  ✓ Good generalization")
    
    trained_models['XGBoost'] = xgb_model
    all_results.append({
        'Model': 'XGBoost',
        'Train_RMSE': xgb_train_metrics['RMSE'],
        'Test_RMSE': xgb_test_metrics['RMSE'],
        'Train_MAE': xgb_train_metrics['MAE'],
        'Test_MAE': xgb_test_metrics['MAE'],
        'Train_R2': xgb_train_metrics['R2'],
        'Test_R2': xgb_test_metrics['R2']
    })
else:
    print("\n⚠️ Skipping XGBoost (not installed)")


# ============================================================
# 5. MODEL COMPARISON
# ============================================================
print_section("5. MODEL COMPARISON")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Test_MAE')

print("\n" + "=" * 90)
print(f"{'Model':<20} {'Train RMSE':>12} {'Test RMSE':>12} {'Train MAE':>12} {'Test MAE':>12} {'Test R²':>10}")
print("=" * 90)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<20} ${row['Train_RMSE']:>10.2f} ${row['Test_RMSE']:>10.2f} ${row['Train_MAE']:>10.2f} ${row['Test_MAE']:>10.2f} {row['Test_R2']:>10.4f}")
print("=" * 90)

# Determine best model
best_model_name = results_df.iloc[0]['Model']
best_test_mae = results_df.iloc[0]['Test_MAE']
print(f"\n🏆 Best Model: {best_model_name} (Test MAE: ${best_test_mae:.2f})")

# Save comparison results
comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
results_df.to_csv(comparison_path, index=False)
print(f"\n✓ Saved model comparison to: {comparison_path}")


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print_section("6. FEATURE IMPORTANCE")

# ------------------------------
# 6.1 Random Forest Feature Importance (Gini)
# ------------------------------
print_subsection("6.1 Random Forest Feature Importance (Gini Impurity)")

rf_importance = pd.DataFrame({
    'feature': feature_names_out,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Features (Random Forest - Gini Importance):")
print("-" * 60)
for i, row in rf_importance.head(20).iterrows():
    bar = "█" * int(row['importance'] * 200)
    print(f"  {row['feature'][:40]:<40} {row['importance']:.4f} {bar}")

# Save RF importance
rf_importance_path = os.path.join(MODELS_DIR, "rf_feature_importance.csv")
rf_importance.to_csv(rf_importance_path, index=False)
print(f"\n✓ Saved to: {rf_importance_path}")


# ------------------------------
# 6.2 XGBoost Feature Importance (Gain)
# ------------------------------
if XGBOOST_AVAILABLE:
    print_subsection("6.2 XGBoost Feature Importance (Gain)")
    
    xgb_importance = pd.DataFrame({
        'feature': feature_names_out,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features (XGBoost - Gain Importance):")
    print("-" * 60)
    for i, row in xgb_importance.head(20).iterrows():
        bar = "█" * int(row['importance'] * 200)
        print(f"  {row['feature'][:40]:<40} {row['importance']:.4f} {bar}")
    
    # Save XGB importance
    xgb_importance_path = os.path.join(MODELS_DIR, "xgb_feature_importance.csv")
    xgb_importance.to_csv(xgb_importance_path, index=False)
    print(f"\n✓ Saved to: {xgb_importance_path}")


# ============================================================
# 7. VISUALIZATION - SEPARATE PLOTS FOR EACH MODEL
# ============================================================
print_section("7. GENERATING EVALUATION PLOTS")

# ------------------------------
# 7.1 Linear Regression Plot
# ------------------------------
print_subsection("7.1 Linear Regression Evaluation Plot")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Linear Regression Model Evaluation', fontsize=16, fontweight='bold')

# Actual vs Predicted (Test)
ax1 = axes[0, 0]
ax1.scatter(y_test, y_test_pred_lr, alpha=0.5, s=20, c='blue')
ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Spend ($)')
ax1.set_ylabel('Predicted Spend ($)')
ax1.set_title('Actual vs Predicted (Test Set)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residual Distribution
ax2 = axes[0, 1]
residuals_lr = y_test - y_test_pred_lr
ax2.hist(residuals_lr, bins=50, edgecolor='black', alpha=0.7, color='blue')
ax2.axvline(x=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Residual ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution (Test Set)')
ax2.grid(True, alpha=0.3)

# Residuals vs Predicted
ax3 = axes[1, 0]
ax3.scatter(y_test_pred_lr, residuals_lr, alpha=0.5, s=20, c='blue')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Spend ($)')
ax3.set_ylabel('Residual ($)')
ax3.set_title('Residuals vs Predicted')
ax3.grid(True, alpha=0.3)

# Metrics Summary
ax4 = axes[1, 1]
ax4.axis('off')
metrics_text = f"""
LINEAR REGRESSION METRICS
{'='*40}

TRAINING SET:
  RMSE:  ${lr_train_metrics['RMSE']:.2f}
  MAE:   ${lr_train_metrics['MAE']:.2f}
  R²:    {lr_train_metrics['R2']:.4f}

TEST SET:
  RMSE:  ${lr_test_metrics['RMSE']:.2f}
  MAE:   ${lr_test_metrics['MAE']:.2f}
  R²:    {lr_test_metrics['R2']:.4f}

{'='*40}
Train/Test RMSE Ratio: {overfit_ratio_lr:.3f}
"""
ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
lr_plot_path = os.path.join(PLOTS_DIR, "linear_regression_evaluation.png")
plt.savefig(lr_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {lr_plot_path}")


# ------------------------------
# 7.2 Random Forest Plot
# ------------------------------
print_subsection("7.2 Random Forest Evaluation Plot")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Random Forest Model Evaluation', fontsize=16, fontweight='bold')

# Actual vs Predicted (Test)
ax1 = axes[0, 0]
ax1.scatter(y_test, y_test_pred_rf, alpha=0.5, s=20, c='green')
ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Spend ($)')
ax1.set_ylabel('Predicted Spend ($)')
ax1.set_title('Actual vs Predicted (Test Set)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residual Distribution
ax2 = axes[0, 1]
residuals_rf = y_test - y_test_pred_rf
ax2.hist(residuals_rf, bins=50, edgecolor='black', alpha=0.7, color='green')
ax2.axvline(x=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Residual ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution (Test Set)')
ax2.grid(True, alpha=0.3)

# Feature Importance (Top 15)
ax3 = axes[1, 0]
top_15_rf = rf_importance.head(15)
ax3.barh(range(len(top_15_rf)), top_15_rf['importance'].values, color='green', alpha=0.7)
ax3.set_yticks(range(len(top_15_rf)))
ax3.set_yticklabels([f[:30] for f in top_15_rf['feature'].values], fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel('Gini Importance')
ax3.set_title('Top 15 Feature Importance')
ax3.grid(True, alpha=0.3, axis='x')

# Metrics Summary
ax4 = axes[1, 1]
ax4.axis('off')
metrics_text = f"""
RANDOM FOREST METRICS
{'='*40}

TRAINING SET:
  RMSE:  ${rf_train_metrics['RMSE']:.2f}
  MAE:   ${rf_train_metrics['MAE']:.2f}
  R²:    {rf_train_metrics['R2']:.4f}

TEST SET:
  RMSE:  ${rf_test_metrics['RMSE']:.2f}
  MAE:   ${rf_test_metrics['MAE']:.2f}
  R²:    {rf_test_metrics['R2']:.4f}

{'='*40}
Train/Test RMSE Ratio: {overfit_ratio_rf:.3f}
n_estimators: 200, max_depth: 12
"""
ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
rf_plot_path = os.path.join(PLOTS_DIR, "random_forest_evaluation.png")
plt.savefig(rf_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {rf_plot_path}")


# ------------------------------
# 7.3 XGBoost Plot
# ------------------------------
if XGBOOST_AVAILABLE:
    print_subsection("7.3 XGBoost Evaluation Plot")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('XGBoost Model Evaluation', fontsize=16, fontweight='bold')
    
    # Actual vs Predicted (Test)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_xgb, alpha=0.5, s=20, c='orange')
    ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Spend ($)')
    ax1.set_ylabel('Predicted Spend ($)')
    ax1.set_title('Actual vs Predicted (Test Set)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual Distribution
    ax2 = axes[0, 1]
    residuals_xgb = y_test - y_test_pred_xgb
    ax2.hist(residuals_xgb, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residual ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution (Test Set)')
    ax2.grid(True, alpha=0.3)
    
    # Feature Importance (Top 15)
    ax3 = axes[1, 0]
    top_15_xgb = xgb_importance.head(15)
    ax3.barh(range(len(top_15_xgb)), top_15_xgb['importance'].values, color='orange', alpha=0.7)
    ax3.set_yticks(range(len(top_15_xgb)))
    ax3.set_yticklabels([f[:30] for f in top_15_xgb['feature'].values], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('Gain Importance')
    ax3.set_title('Top 15 Feature Importance')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
XGBOOST METRICS
{'='*40}

TRAINING SET:
  RMSE:  ${xgb_train_metrics['RMSE']:.2f}
  MAE:   ${xgb_train_metrics['MAE']:.2f}
  R²:    {xgb_train_metrics['R2']:.4f}

TEST SET:
  RMSE:  ${xgb_test_metrics['RMSE']:.2f}
  MAE:   ${xgb_test_metrics['MAE']:.2f}
  R²:    {xgb_test_metrics['R2']:.4f}

{'='*40}
Train/Test RMSE Ratio: {overfit_ratio_xgb:.3f}
n_estimators: 200, max_depth: 6, lr: 0.05
"""
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.5))
    
    plt.tight_layout()
    xgb_plot_path = os.path.join(PLOTS_DIR, "xgboost_evaluation.png")
    plt.savefig(xgb_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {xgb_plot_path}")


# ------------------------------
# 7.4 Model Comparison Plot
# ------------------------------
print_subsection("7.4 Model Comparison Plot")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

models_list = results_df['Model'].tolist()
x = np.arange(len(models_list))
width = 0.35

# RMSE Comparison
ax1 = axes[0]
ax1.bar(x - width/2, results_df['Train_RMSE'], width, label='Train', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, results_df['Test_RMSE'], width, label='Test', color='coral', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE ($)')
ax1.set_title('RMSE Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models_list, rotation=15)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# MAE Comparison
ax2 = axes[1]
ax2.bar(x - width/2, results_df['Train_MAE'], width, label='Train', color='steelblue', alpha=0.8)
ax2.bar(x + width/2, results_df['Test_MAE'], width, label='Test', color='coral', alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('MAE ($)')
ax2.set_title('MAE Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(models_list, rotation=15)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# R² Comparison
ax3 = axes[2]
ax3.bar(x - width/2, results_df['Train_R2'], width, label='Train', color='steelblue', alpha=0.8)
ax3.bar(x + width/2, results_df['Test_R2'], width, label='Test', color='coral', alpha=0.8)
ax3.set_xlabel('Model')
ax3.set_ylabel('R²')
ax3.set_title('R² Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(models_list, rotation=15)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
comparison_plot_path = os.path.join(PLOTS_DIR, "model_comparison.png")
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {comparison_plot_path}")


# ============================================================
# 8. SAVE MODELS
# ============================================================
print_section("8. SAVING MODELS")

# Save each model separately
for model_name, model in trained_models.items():
    model_filename = model_name.lower().replace(' ', '_') + '.joblib'
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"✓ Saved: {model_path}")

# Save best model with all metadata
best_model = trained_models[best_model_name]
final_model_path = os.path.join(MODELS_DIR, "final_model.joblib")
joblib.dump({
    'model': best_model,
    'model_name': best_model_name,
    'preprocessor': preprocessor,
    'feature_cols': feature_cols,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'metrics': results_df[results_df['Model'] == best_model_name].iloc[0].to_dict(),
    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}, final_model_path)
print(f"✓ Saved best model package: {final_model_path}")


# ============================================================
# 9. GENERATE DOCUMENTATION
# ============================================================
print_section("9. GENERATING DOCUMENTATION")

# Build documentation content
doc_content = f"""# Model Evaluation Report

## Overview

This document compares the performance of three regression models trained to predict customer spend in the next 30 days.

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Models Trained

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline model for interpretability and benchmarking |
| **Random Forest** | Ensemble of 200 decision trees with controlled depth |
| **XGBoost** | Gradient boosting with conservative hyperparameters |

---

## 2. Dataset Summary

| Metric | Value |
|--------|-------|
| Total Customers | {len(df):,} |
| Training Set | {len(train_df):,} ({TRAIN_RATIO*100:.0f}%) |
| Test Set | {len(test_df):,} ({(1-TRAIN_RATIO)*100:.0f}%) |
| Features | {len(feature_cols)} |
| Split Method | Time-based (by first_purchase_date) |

### Target Variable: `future_spend_30d`

| Statistic | Value |
|-----------|-------|
| Mean | ${df[TARGET].mean():.2f} |
| Median | ${df[TARGET].median():.2f} |
| Std Dev | ${df[TARGET].std():.2f} |
| Min | ${df[TARGET].min():.2f} |
| Max | ${df[TARGET].max():.2f} |

---

## 3. Model Performance Comparison

### 3.1 Metrics Summary

| Model | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² |
|-------|------------|-----------|-----------|----------|----------|---------|
| Linear Regression | ${lr_train_metrics['RMSE']:.2f} | ${lr_test_metrics['RMSE']:.2f} | ${lr_train_metrics['MAE']:.2f} | ${lr_test_metrics['MAE']:.2f} | {lr_train_metrics['R2']:.4f} | {lr_test_metrics['R2']:.4f} |
| Random Forest | ${rf_train_metrics['RMSE']:.2f} | ${rf_test_metrics['RMSE']:.2f} | ${rf_train_metrics['MAE']:.2f} | ${rf_test_metrics['MAE']:.2f} | {rf_train_metrics['R2']:.4f} | {rf_test_metrics['R2']:.4f} |
"""

if XGBOOST_AVAILABLE:
    doc_content += f"""| XGBoost | ${xgb_train_metrics['RMSE']:.2f} | ${xgb_test_metrics['RMSE']:.2f} | ${xgb_train_metrics['MAE']:.2f} | ${xgb_test_metrics['MAE']:.2f} | {xgb_train_metrics['R2']:.4f} | {xgb_test_metrics['R2']:.4f} |
"""

doc_content += f"""
### 3.2 Overfitting Analysis

| Model | Train/Test RMSE Ratio | Assessment |
|-------|----------------------|------------|
| Linear Regression | {overfit_ratio_lr:.3f} | {'Good' if 0.8 <= overfit_ratio_lr <= 1.2 else 'Overfitting' if overfit_ratio_lr > 1.2 else 'Underfitting'} |
| Random Forest | {overfit_ratio_rf:.3f} | {'Good' if 0.8 <= overfit_ratio_rf <= 1.2 else 'Overfitting' if overfit_ratio_rf > 1.2 else 'Underfitting'} |
"""

if XGBOOST_AVAILABLE:
    doc_content += f"""| XGBoost | {overfit_ratio_xgb:.3f} | {'Good' if 0.8 <= overfit_ratio_xgb <= 1.2 else 'Overfitting' if overfit_ratio_xgb > 1.2 else 'Underfitting'} |
"""

doc_content += f"""
> A Train/Test RMSE ratio between 0.8-1.2 indicates good generalization.
> Ratio > 1.2 suggests overfitting (model memorizes training data).
> Ratio < 0.8 suggests underfitting (model too simple).

---

## 4. Model Selection Decision

### 🏆 Selected Model: **{best_model_name}**

### Rationale

"""

# Add specific rationale based on best model
if best_model_name == "XGBoost":
    doc_content += f"""
1. **Lowest Test MAE (${best_test_mae:.2f})**: XGBoost achieves the best predictive accuracy on unseen data, meaning predictions are closest to actual customer spending.

2. **Balanced Generalization**: The train/test RMSE ratio of {overfit_ratio_xgb:.3f} indicates the model generalizes well without overfitting.

3. **Feature Handling**: XGBoost's gradient boosting naturally handles feature interactions and non-linear relationships in customer behavior.

4. **Regularization**: Built-in L1/L2 regularization (reg_alpha=0.1, reg_lambda=1.0) prevents overfitting on our relatively small dataset.

5. **Conservative Hyperparameters**: Using learning_rate=0.05 and max_depth=6 ensures stable predictions without chasing noise.
"""
elif best_model_name == "Random Forest":
    doc_content += f"""
1. **Lowest Test MAE (${best_test_mae:.2f})**: Random Forest achieves the best predictive accuracy on unseen data.

2. **Robust to Overfitting**: The ensemble of 200 trees with controlled depth (max_depth=12) and minimum samples per leaf (10) prevents overfitting.

3. **No Feature Scaling Required**: Random Forest is invariant to feature scales, making it robust to our mixed feature types.

4. **Interpretable Feature Importance**: Gini importance provides clear insights into which features drive predictions.

5. **Handles Non-linear Relationships**: Decision tree ensembles capture complex patterns in customer purchasing behavior.
"""
else:  # Linear Regression
    doc_content += f"""
1. **Lowest Test MAE (${best_test_mae:.2f})**: Despite its simplicity, Linear Regression achieves competitive accuracy.

2. **Perfect Interpretability**: Coefficients directly show how each feature impacts predicted spend.

3. **No Overfitting Risk**: Linear models are inherently simple and don't overfit on small datasets.

4. **Fast Inference**: Prediction is a simple dot product, making it ideal for real-time applications.

5. **Baseline Reliability**: Serves as a reliable baseline that more complex models should beat.
"""

doc_content += f"""
### Why Not Other Models?

"""

# Add comparison with non-selected models
for _, row in results_df.iterrows():
    if row['Model'] != best_model_name:
        if row['Model'] == "Linear Regression":
            doc_content += f"""
**Linear Regression**: While interpretable, it achieved Test MAE of ${row['Test_MAE']:.2f} (vs ${best_test_mae:.2f} for {best_model_name}). Linear models cannot capture non-linear relationships in customer behavior.
"""
        elif row['Model'] == "Random Forest":
            doc_content += f"""
**Random Forest**: Achieved Test MAE of ${row['Test_MAE']:.2f}. While robust, it {'underperformed' if row['Test_MAE'] > best_test_mae else 'slightly underperformed'} compared to {best_model_name} on this dataset.
"""
        elif row['Model'] == "XGBoost":
            doc_content += f"""
**XGBoost**: Achieved Test MAE of ${row['Test_MAE']:.2f}. While powerful, it {'underperformed' if row['Test_MAE'] > best_test_mae else 'slightly underperformed'} compared to {best_model_name} on this dataset.
"""

doc_content += f"""
---

## 5. Feature Importance

### Top 10 Features (Random Forest - Gini Importance)

| Rank | Feature | Importance |
|------|---------|------------|
"""

for idx, (i, row) in enumerate(rf_importance.head(10).iterrows()):
    doc_content += f"| {idx+1} | {row['feature']} | {row['importance']:.4f} |\n"

if XGBOOST_AVAILABLE:
    doc_content += f"""
### Top 10 Features (XGBoost - Gain Importance)

| Rank | Feature | Importance |
|------|---------|------------|
"""
    for idx, (i, row) in enumerate(xgb_importance.head(10).iterrows()):
        doc_content += f"| {idx+1} | {row['feature']} | {row['importance']:.4f} |\n"

doc_content += f"""
### Interpretation

The most important features relate to:
1. **Monetary features** (total spend, average order value) - Past spending strongly predicts future spending
2. **Frequency features** (transaction counts) - Purchase frequency indicates engagement
3. **Recency features** (days since last purchase) - Recent buyers are more likely to buy again

---

## 6. Visualizations

| Plot | Location |
|------|----------|
| Linear Regression Evaluation | `plots/linear_regression_evaluation.png` |
| Random Forest Evaluation | `plots/random_forest_evaluation.png` |
| XGBoost Evaluation | `plots/xgboost_evaluation.png` |
| Model Comparison | `plots/model_comparison.png` |

---

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `models/linear_regression.joblib` | Trained Linear Regression model |
| `models/random_forest.joblib` | Trained Random Forest model |
| `models/xgboost.joblib` | Trained XGBoost model |
| `models/final_model.joblib` | Best model with preprocessor and metadata |
| `models/model_comparison.csv` | Performance metrics for all models |
| `models/rf_feature_importance.csv` | Random Forest feature importance |
| `models/xgb_feature_importance.csv` | XGBoost feature importance |

---

## 8. Recommendations

1. **Deploy {best_model_name}** as the production model for 30-day spend prediction.

2. **Monitor Performance**: Track prediction accuracy over time and retrain if performance degrades.

3. **Feature Updates**: Regularly update customer features to capture recent behavior.

4. **Business Actions**: Use predictions to:
   - Target high-value customers with retention campaigns
   - Identify at-risk customers (low predicted spend)
   - Optimize marketing budget allocation

---

*Report generated automatically by model_training.py*
"""

# Save documentation
doc_path = os.path.join(DOCS_DIR, "05_model_evaluation.md")
with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(doc_content)
print(f"✓ Saved documentation: {doc_path}")


# ============================================================
# 10. SUMMARY
# ============================================================
print_section("TRAINING COMPLETE - SUMMARY")

print(f"""
Dataset:
  Total Customers: {len(df):,}
  Training Set: {len(train_df):,}
  Test Set: {len(test_df):,}
  Features: {len(feature_cols)}

Models Trained:
  1. Linear Regression (Baseline)
  2. Random Forest (200 trees, max_depth=12)
  3. XGBoost (200 trees, max_depth=6, lr=0.05)

Best Model: 🏆 {best_model_name}
  Test MAE:  ${best_test_mae:.2f}
  Test RMSE: ${results_df[results_df['Model']==best_model_name].iloc[0]['Test_RMSE']:.2f}
  Test R²:   {results_df[results_df['Model']==best_model_name].iloc[0]['Test_R2']:.4f}

Files Generated:
  Models:
    - models/linear_regression.joblib
    - models/random_forest.joblib
    - models/xgboost.joblib (if available)
    - models/final_model.joblib
  
  Plots:
    - plots/linear_regression_evaluation.png
    - plots/random_forest_evaluation.png
    - plots/xgboost_evaluation.png (if available)
    - plots/model_comparison.png
  
  Documentation:
    - docs/05_model_evaluation.md

Next Steps:
  1. Create inference function
  2. Build Streamlit UI
  3. Test end-to-end prediction
""")
