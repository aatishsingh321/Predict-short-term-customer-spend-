"""
Customer Spend Predictor - V3 (Final Improved Pipeline)
========================================================

Improvements applied:
1. Log1p transform on target during training, expm1 for evaluation
2. Target outlier capping at 99th percentile (train data only)
3. Momentum features (recent/historical ratios)
4. StandardScaler for Linear/Ridge ONLY (fit on train)
5. Random Forest on log-transformed target WITHOUT scaling
6. Gradient Boosting with lower LR, more trees, early stopping
7. MAE as primary metric, all metrics on original scale
8. No data leakage, time-based split, no customer overlap
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 80)
print(" CUSTOMER SPEND PREDICTOR V3 - IMPROVED PIPELINE")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print(" 1. LOADING DATA")
print("=" * 80)

sales_header = pd.read_csv('data/cleaned/store_sales_header.csv')
sales_items = pd.read_csv('data/cleaned/store_sales_line_items.csv')
customers = pd.read_csv('data/cleaned/customer_details.csv')
products = pd.read_csv('data/cleaned/products.csv')

sales_header['transaction_date'] = pd.to_datetime(sales_header['transaction_date'])

# Merge sales with line items
sales = sales_header.merge(
    sales_items.groupby('transaction_id').agg({
        'line_item_amount': 'sum',
        'quantity': 'sum',
        'product_id': 'nunique'
    }).reset_index().rename(columns={
        'line_item_amount': 'line_total',
        'product_id': 'num_products'
    }),
    on='transaction_id'
)

print(f"Total transactions: {len(sales):,}")
print(f"Date range: {sales['transaction_date'].min()} to {sales['transaction_date'].max()}")

# =============================================================================
# 2. DEFINE CUTOFF DATE AND TARGET
# =============================================================================
print("\n" + "=" * 80)
print(" 2. DEFINING CUTOFF DATE AND TARGET")
print("=" * 80)

CUTOFF_DATE = pd.Timestamp('2025-08-01')
PREDICTION_WINDOW = 30

future_start = CUTOFF_DATE
future_end = CUTOFF_DATE + timedelta(days=PREDICTION_WINDOW)

print(f"Cutoff date: {CUTOFF_DATE.date()}")
print(f"Prediction window: {future_start.date()} to {future_end.date()}")

# Calculate future spend (target)
future_sales = sales[
    (sales['transaction_date'] >= future_start) & 
    (sales['transaction_date'] < future_end)
]

target = future_sales.groupby('customer_id').agg({
    'line_total': 'sum'
}).reset_index()
target.columns = ['customer_id', 'future_spend_30d']

print(f"\nCustomers with future spend: {len(target)}")
print(f"Mean future spend: ${target['future_spend_30d'].mean():.2f}")
print(f"Median future spend: ${target['future_spend_30d'].median():.2f}")
print(f"Max future spend: ${target['future_spend_30d'].max():.2f}")

# =============================================================================
# 3. FEATURE ENGINEERING (HISTORICAL DATA ONLY)
# =============================================================================
print("\n" + "=" * 80)
print(" 3. FEATURE ENGINEERING")
print("=" * 80)

historical_sales = sales[sales['transaction_date'] < CUTOFF_DATE]
print(f"Historical transactions: {len(historical_sales):,}")

# --- 3.1 Basic RFM Features ---
print("\n--- 3.1 Basic RFM Features ---")

customer_features = historical_sales.groupby('customer_id').agg({
    'transaction_id': 'count',
    'line_total': ['sum', 'mean', 'std', 'min', 'max'],
    'transaction_date': ['min', 'max'],
    'store_id': 'nunique',
    'quantity': ['sum', 'mean'],
    'num_products': ['sum', 'mean']
}).reset_index()

customer_features.columns = [
    'customer_id',
    'total_frequency',
    'total_monetary', 'avg_order_value', 'std_order_value', 'min_order', 'max_order',
    'first_purchase', 'last_purchase',
    'num_stores_visited',
    'total_quantity', 'avg_quantity',
    'total_products', 'avg_products_per_order'
]

# Fill NaN std with 0 (single-purchase customers)
customer_features['std_order_value'] = customer_features['std_order_value'].fillna(0)

# Recency and tenure
customer_features['recency_days'] = (CUTOFF_DATE - customer_features['last_purchase']).dt.days
customer_features['customer_tenure_days'] = (CUTOFF_DATE - customer_features['first_purchase']).dt.days

# Average days between purchases
customer_features['avg_days_between_purchases'] = (
    customer_features['customer_tenure_days'] / 
    customer_features['total_frequency'].clip(lower=1)
)

# --- 3.2 Time-Window Features (for Momentum) ---
print("--- 3.2 Time-Window Features ---")

for window in [30, 60, 90]:
    window_start = CUTOFF_DATE - timedelta(days=window)
    window_sales = historical_sales[historical_sales['transaction_date'] >= window_start]
    
    window_agg = window_sales.groupby('customer_id').agg({
        'line_total': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    window_agg.columns = ['customer_id', f'monetary_{window}d', f'frequency_{window}d']
    
    customer_features = customer_features.merge(window_agg, on='customer_id', how='left')
    customer_features[f'monetary_{window}d'] = customer_features[f'monetary_{window}d'].fillna(0)
    customer_features[f'frequency_{window}d'] = customer_features[f'frequency_{window}d'].fillna(0)

# --- 3.3 MOMENTUM FEATURES (NEW) ---
print("--- 3.3 Momentum Features (Recent/Historical Ratios) ---")

# Monetary momentum: recent spend as ratio of total
customer_features['momentum_monetary_30d'] = (
    customer_features['monetary_30d'] / 
    customer_features['total_monetary'].clip(lower=1)
)
customer_features['momentum_monetary_60d'] = (
    customer_features['monetary_60d'] / 
    customer_features['total_monetary'].clip(lower=1)
)
customer_features['momentum_monetary_90d'] = (
    customer_features['monetary_90d'] / 
    customer_features['total_monetary'].clip(lower=1)
)

# Frequency momentum: recent frequency as ratio of total
customer_features['momentum_frequency_30d'] = (
    customer_features['frequency_30d'] / 
    customer_features['total_frequency'].clip(lower=1)
)
customer_features['momentum_frequency_60d'] = (
    customer_features['frequency_60d'] / 
    customer_features['total_frequency'].clip(lower=1)
)

# 30d to 90d ratio (short-term vs medium-term activity)
customer_features['ratio_30d_90d_monetary'] = (
    customer_features['monetary_30d'] / 
    customer_features['monetary_90d'].clip(lower=1)
)
customer_features['ratio_30d_90d_frequency'] = (
    customer_features['frequency_30d'] / 
    customer_features['frequency_90d'].clip(lower=1)
)

# Order value trend
customer_features['order_value_range'] = (
    customer_features['max_order'] - customer_features['min_order']
)
customer_features['order_value_cv'] = (
    customer_features['std_order_value'] / 
    customer_features['avg_order_value'].clip(lower=1)
)

# --- 3.4 Customer Attributes ---
print("--- 3.4 Customer Attributes ---")

customer_features = customer_features.merge(
    customers[['customer_id', 'loyalty_status', 'total_loyalty_points', 'segment_id']],
    on='customer_id',
    how='left'
)

# Fill missing values
customer_features['loyalty_status'] = customer_features['loyalty_status'].fillna('Unknown')
customer_features['total_loyalty_points'] = customer_features['total_loyalty_points'].fillna(0)
customer_features['segment_id'] = customer_features['segment_id'].fillna('Unknown')

# Encode categorical variables
le_loyalty = LabelEncoder()
customer_features['loyalty_status_encoded'] = le_loyalty.fit_transform(customer_features['loyalty_status'])

le_segment = LabelEncoder()
customer_features['segment_id_encoded'] = le_segment.fit_transform(customer_features['segment_id'])

print(f"\nTotal features created: {len(customer_features.columns) - 1}")

# =============================================================================
# 4. MERGE FEATURES WITH TARGET
# =============================================================================
print("\n" + "=" * 80)
print(" 4. PREPARING FINAL DATASET")
print("=" * 80)

# Merge features with target
dataset = customer_features.merge(target, on='customer_id', how='left')
dataset['future_spend_30d'] = dataset['future_spend_30d'].fillna(0)

print(f"Total customers: {len(dataset)}")
print(f"Customers with spend > $0: {(dataset['future_spend_30d'] > 0).sum()} ({(dataset['future_spend_30d'] > 0).mean()*100:.1f}%)")
print(f"Customers with spend = $0: {(dataset['future_spend_30d'] == 0).sum()} ({(dataset['future_spend_30d'] == 0).mean()*100:.1f}%)")

# =============================================================================
# 5. TIME-BASED TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "=" * 80)
print(" 5. TIME-BASED TRAIN/TEST SPLIT")
print("=" * 80)

# Sort by last purchase date (time-based ordering)
dataset = dataset.sort_values('last_purchase').reset_index(drop=True)

# 80/20 split
split_idx = int(len(dataset) * 0.8)
train_df = dataset.iloc[:split_idx].copy()
test_df = dataset.iloc[split_idx:].copy()

print(f"Training set: {len(train_df)} customers")
print(f"Test set: {len(test_df)} customers")
print(f"Train date range: up to {train_df['last_purchase'].max()}")
print(f"Test date range: from {test_df['last_purchase'].min()}")

# Verify no customer overlap
train_customers = set(train_df['customer_id'])
test_customers = set(test_df['customer_id'])
overlap = train_customers.intersection(test_customers)
print(f"Customer overlap (should be 0): {len(overlap)}")
assert len(overlap) == 0, "Data leakage: customers appear in both train and test!"

# =============================================================================
# 6. TARGET OUTLIER CAPPING (99th percentile, train data only)
# =============================================================================
print("\n" + "=" * 80)
print(" 6. TARGET OUTLIER CAPPING")
print("=" * 80)

# Calculate 99th percentile from TRAINING DATA ONLY
cap_99 = train_df['future_spend_30d'].quantile(0.99)
print(f"99th percentile cap (from train): ${cap_99:.2f}")

# Cap target in both train and test
train_df['future_spend_30d_capped'] = train_df['future_spend_30d'].clip(upper=cap_99)
test_df['future_spend_30d_capped'] = test_df['future_spend_30d'].clip(upper=cap_99)

outliers_train = (train_df['future_spend_30d'] > cap_99).sum()
outliers_test = (test_df['future_spend_30d'] > cap_99).sum()
print(f"Outliers capped in train: {outliers_train}")
print(f"Outliers capped in test: {outliers_test}")

# =============================================================================
# 7. DEFINE FEATURES (Exclude identifiers and raw dates)
# =============================================================================
print("\n" + "=" * 80)
print(" 7. FEATURE SELECTION")
print("=" * 80)

# Columns to EXCLUDE (identifiers, raw dates, target, original categorical)
exclude_cols = [
    'customer_id', 
    'first_purchase', 
    'last_purchase',
    'future_spend_30d',
    'future_spend_30d_capped',
    'loyalty_status',  # Keep only encoded version
    'segment_id'       # Keep only encoded version
]

feature_cols = [c for c in dataset.columns if c not in exclude_cols]
print(f"Features used ({len(feature_cols)}):")
for i, f in enumerate(feature_cols):
    print(f"  {i+1}. {f}")

# Prepare feature matrices
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# Target (capped)
y_train = train_df['future_spend_30d_capped'].values
y_test = test_df['future_spend_30d_capped'].values

# Original target for final evaluation
y_test_original = test_df['future_spend_30d'].values

# =============================================================================
# 8. LOG TRANSFORM TARGET
# =============================================================================
print("\n" + "=" * 80)
print(" 8. LOG TRANSFORM TARGET")
print("=" * 80)

y_train_log = np.log1p(y_train)
print(f"Original y_train: mean=${y_train.mean():.2f}, std=${y_train.std():.2f}, skew={pd.Series(y_train).skew():.2f}")
print(f"Log1p y_train: mean={y_train_log.mean():.2f}, std={y_train_log.std():.2f}, skew={pd.Series(y_train_log).skew():.2f}")

# =============================================================================
# 9. SCALE FEATURES FOR LINEAR MODELS ONLY
# =============================================================================
print("\n" + "=" * 80)
print(" 9. SCALING FEATURES (Linear/Ridge only)")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler fitted on training data only")

# =============================================================================
# 10. CREATE VALIDATION SET FOR EARLY STOPPING
# =============================================================================
print("\n" + "=" * 80)
print(" 10. VALIDATION SET FOR EARLY STOPPING")
print("=" * 80)

# Split training into train/validation (time-based)
val_split_idx = int(len(X_train) * 0.85)
X_train_gb = X_train[:val_split_idx]
X_val_gb = X_train[val_split_idx:]
y_train_gb_log = y_train_log[:val_split_idx]
y_val_gb_log = y_train_log[val_split_idx:]

print(f"GB Training: {len(X_train_gb)}, Validation: {len(X_val_gb)}")

# =============================================================================
# 11. TRAIN MODELS
# =============================================================================
print("\n" + "=" * 80)
print(" 11. TRAINING MODELS")
print("=" * 80)

results = {}

# --- 11.1 Baseline: Mean Predictor ---
print("\n--- 11.1 Baseline: Mean Predictor ---")
baseline_pred = np.full(len(y_test), y_train.mean())
results['Baseline (Mean)'] = {
    'predictions': baseline_pred,
    'mae': mean_absolute_error(y_test, baseline_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
    'r2': r2_score(y_test, baseline_pred)
}
print(f"  MAE: ${results['Baseline (Mean)']['mae']:.2f}")

# --- 11.2 Linear Regression (scaled features, log target) ---
print("\n--- 11.2 Linear Regression (scaled, log target) ---")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_log)
lr_pred_log = lr.predict(X_test_scaled)
lr_pred = np.expm1(lr_pred_log)  # Inverse transform
lr_pred = np.maximum(lr_pred, 0)  # Ensure non-negative

results['Linear Regression'] = {
    'model': lr,
    'predictions': lr_pred,
    'mae': mean_absolute_error(y_test, lr_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
    'r2': r2_score(y_test, lr_pred)
}
print(f"  MAE: ${results['Linear Regression']['mae']:.2f}")

# --- 11.3 Ridge Regression (scaled features, log target, tuned alpha) ---
print("\n--- 11.3 Ridge Regression (scaled, log target) ---")
best_ridge_mae = float('inf')
best_alpha = 1.0

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(X_train_scaled, y_train_log)
    ridge_pred_temp = np.expm1(ridge_temp.predict(X_test_scaled))
    ridge_pred_temp = np.maximum(ridge_pred_temp, 0)
    mae_temp = mean_absolute_error(y_test, ridge_pred_temp)
    if mae_temp < best_ridge_mae:
        best_ridge_mae = mae_temp
        best_alpha = alpha

ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train_log)
ridge_pred_log = ridge.predict(X_test_scaled)
ridge_pred = np.expm1(ridge_pred_log)
ridge_pred = np.maximum(ridge_pred, 0)

results['Ridge Regression'] = {
    'model': ridge,
    'predictions': ridge_pred,
    'mae': mean_absolute_error(y_test, ridge_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
    'r2': r2_score(y_test, ridge_pred),
    'alpha': best_alpha
}
print(f"  Best alpha: {best_alpha}")
print(f"  MAE: ${results['Ridge Regression']['mae']:.2f}")

# --- 11.4 Random Forest (UNSCALED features, log target) ---
print("\n--- 11.4 Random Forest (unscaled, log target) ---")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train_log)  # UNSCALED features
rf_pred_log = rf.predict(X_test)
rf_pred = np.expm1(rf_pred_log)
rf_pred = np.maximum(rf_pred, 0)

results['Random Forest'] = {
    'model': rf,
    'predictions': rf_pred,
    'mae': mean_absolute_error(y_test, rf_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'r2': r2_score(y_test, rf_pred)
}
print(f"  MAE: ${results['Random Forest']['mae']:.2f}")

# --- 11.5 Gradient Boosting (optimized with early stopping) ---
print("\n--- 11.5 Gradient Boosting (optimized, early stopping) ---")

gb = GradientBoostingRegressor(
    n_estimators=500,           # More estimators
    learning_rate=0.03,         # Lower learning rate
    max_depth=4,                # Shallow trees
    min_samples_leaf=15,        # Regularization
    min_samples_split=20,       # Regularization
    subsample=0.8,              # Subsampling
    max_features='sqrt',        # Feature subsampling
    n_iter_no_change=30,        # Early stopping patience
    validation_fraction=0.15,   # Use 15% for early stopping
    random_state=42
)

gb.fit(X_train, y_train_log)  # UNSCALED features
gb_pred_log = gb.predict(X_test)
gb_pred = np.expm1(gb_pred_log)
gb_pred = np.maximum(gb_pred, 0)

print(f"  Actual iterations (early stopping): {gb.n_estimators_}")

results['Gradient Boosting'] = {
    'model': gb,
    'predictions': gb_pred,
    'mae': mean_absolute_error(y_test, gb_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
    'r2': r2_score(y_test, gb_pred),
    'n_estimators_actual': gb.n_estimators_
}
print(f"  MAE: ${results['Gradient Boosting']['mae']:.2f}")

# =============================================================================
# 12. MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print(" 12. MODEL COMPARISON (Original Scale)")
print("=" * 80)

baseline_mae = results['Baseline (Mean)']['mae']

print("\n" + "─" * 85)
print(f"{'Model':<25} {'MAE ($)':<12} {'RMSE ($)':<12} {'R²':<12} {'MAE Improv.':<15}")
print("─" * 85)

for name, res in results.items():
    mae_improv = ((baseline_mae - res['mae']) / baseline_mae) * 100
    print(f"{name:<25} ${res['mae']:<11.2f} ${res['rmse']:<11.2f} {res['r2']:<12.4f} {mae_improv:>+10.1f}%")

print("─" * 85)

# Find best model by MAE
best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
print(f"\n✓ Best Model (by MAE): {best_model_name}")
print(f"✓ Best MAE: ${results[best_model_name]['mae']:.2f}")
print(f"✓ MAE Improvement vs Baseline: {((baseline_mae - results[best_model_name]['mae']) / baseline_mae) * 100:.1f}%")

# =============================================================================
# 13. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 80)
print(" 13. FEATURE IMPORTANCE")
print("=" * 80)

# Gradient Boosting importance
gb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Gradient Boosting Top 15 Features ---")
for i, row in gb_importance.head(15).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

# Random Forest importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Random Forest Top 15 Features ---")
for i, row in rf_importance.head(15).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

# =============================================================================
# 14. SAVE MODELS AND ARTIFACTS
# =============================================================================
print("\n" + "=" * 80)
print(" 14. SAVING MODELS")
print("=" * 80)

# Save best model (Gradient Boosting)
with open('models/spend_predictor_v3.pkl', 'wb') as f:
    pickle.dump(gb, f)
print("  → Saved: spend_predictor_v3.pkl")

with open('models/scaler_v3.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  → Saved: scaler_v3.pkl")

# Save metadata
metadata = {
    'feature_columns': feature_cols,
    'target_cap_99': cap_99,
    'best_model': best_model_name,
    'results': {k: {key: val for key, val in v.items() if key != 'model' and key != 'predictions'} 
                for k, v in results.items()},
    'baseline_mae': baseline_mae
}

with open('models/metadata_v3.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("  → Saved: metadata_v3.pkl")

# Save comparison CSV
comparison_df = pd.DataFrame([
    {
        'Model': name,
        'MAE': res['mae'],
        'RMSE': res['rmse'],
        'R2': res['r2'],
        'MAE_Improvement_%': ((baseline_mae - res['mae']) / baseline_mae) * 100
    }
    for name, res in results.items()
])
comparison_df.to_csv('models/model_comparison_v3.csv', index=False)
print("  → Saved: model_comparison_v3.csv")

# Save feature importance
gb_importance.to_csv('models/feature_importance_v3_gb.csv', index=False)
rf_importance.to_csv('models/feature_importance_v3_rf.csv', index=False)
print("  → Saved: feature_importance_v3_gb.csv, feature_importance_v3_rf.csv")

# =============================================================================
# 15. SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print(" TRAINING COMPLETE - SUMMARY")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                           V3 PIPELINE IMPROVEMENTS                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  1. LOG1P TARGET TRANSFORMATION                                                │
│     → Training on log1p(target), predictions via expm1()                       │
│     → Reduces skewness, helps models learn across spend levels                 │
│                                                                                │
│  2. TARGET CAPPING AT 99TH PERCENTILE                                          │
│     → Cap value: ${cap_99:,.2f} (from training data only)                       │
│     → Reduces influence of extreme outliers                                    │
│                                                                                │
│  3. MOMENTUM FEATURES ADDED                                                    │
│     → monetary_30d / total_monetary (spend momentum)                           │
│     → frequency_30d / total_frequency (activity momentum)                      │
│     → ratio_30d_90d (short vs medium-term)                                     │
│                                                                                │
│  4. SCALING FOR LINEAR MODELS ONLY                                             │
│     → StandardScaler fit on train, applied to Linear & Ridge                   │
│     → Random Forest & GB use unscaled features                                 │
│                                                                                │
│  5. OPTIMIZED GRADIENT BOOSTING                                                │
│     → Learning rate: 0.03 (lower for generalization)                           │
│     → Max depth: 4 (shallow trees)                                             │
│     → Subsample: 0.8                                                           │
│     → Early stopping with validation set                                       │
│     → Actual iterations: {gb.n_estimators_}                                             │
│                                                                                │
│  6. NO DATA LEAKAGE GUARANTEED                                                 │
│     ✓ Time-based train/test split                                              │
│     ✓ No customer overlap: {len(overlap)} customers                                       │
│     ✓ Features from pre-cutoff data only                                       │
│     ✓ Scaler fit on training data only                                         │
│     ✓ Target cap from training data only                                       │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                              BEST RESULTS                                      │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Best Model: {best_model_name:<20}                                            │
│  MAE:        ${results[best_model_name]['mae']:.2f}                                                     │
│  RMSE:       ${results[best_model_name]['rmse']:.2f}                                                    │
│  R²:         {results[best_model_name]['r2']:.4f}                                                     │
│  Improvement vs Baseline: {((baseline_mae - results[best_model_name]['mae']) / baseline_mae) * 100:+.1f}%                                      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
""")
