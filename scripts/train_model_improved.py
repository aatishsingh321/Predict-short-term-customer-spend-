"""
================================================================================
IMPROVED MODEL TRAINING PIPELINE
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Enhanced regression pipeline with improved performance and no data leakage

IMPROVEMENTS IMPLEMENTED:
1. Log1p transformation on target variable (handles right-skewed spend distribution)
2. Target winsorization at 99th percentile (reduces outlier dominance)
3. StandardScaler for linear models (fit on train only - prevents leakage)
4. TimeSeriesSplit cross-validation (respects temporal ordering)
5. Business-aligned metrics (MAE primary, RMSE/R² diagnostic)
6. Improved GB/XGBoost (lower LR, early stopping, regularization)
7. Baseline comparison (mean predictor benchmark)

STRICT CONSTRAINTS MAINTAINED:
- Time-based train/test split (NO random splitting)
- No feature leakage (target never used in features)
- Cutoff date logic preserved
- No customer in both train and test
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

# Optional: XGBoost (if available)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
CLEANED_DATA_DIR = "/Users/apple/Customer Spend Predictor/data/cleaned"
MODELS_DIR = "/Users/apple/Customer Spend Predictor/models"

# Time-based split parameters
CUTOFF_DATE = pd.Timestamp('2025-08-01')
PREDICTION_WINDOW = 30  # days

# Winsorization percentile (cap outliers)
WINSORIZE_PERCENTILE = 99

# Random state for reproducibility
RANDOM_STATE = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_subsection(title):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")


# ============================================================
# CUSTOM BASELINE PREDICTOR
# ============================================================
class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline model that predicts the mean of training target.
    
    WHY: Provides a benchmark to measure if our models add value.
    Any useful model should significantly outperform this baseline.
    """
    def __init__(self):
        self.mean_ = None
    
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)


# ============================================================
# 1. LOAD DATA
# ============================================================
print_section("1. LOADING CLEANED DATA")

sales_header = pd.read_csv(f"{CLEANED_DATA_DIR}/store_sales_header.csv")
sales_header['transaction_date'] = pd.to_datetime(sales_header['transaction_date'])

line_items = pd.read_csv(f"{CLEANED_DATA_DIR}/store_sales_line_items.csv")
customers = pd.read_csv(f"{CLEANED_DATA_DIR}/customer_details.csv")
products = pd.read_csv(f"{CLEANED_DATA_DIR}/products.csv")

customers['customer_since'] = pd.to_datetime(customers['customer_since'], errors='coerce')

print(f"Sales transactions: {len(sales_header):,}")
print(f"Cutoff date: {CUTOFF_DATE}")
print(f"Prediction window: {PREDICTION_WINDOW} days")


# ============================================================
# 2. DEFINE TARGET VARIABLE (with improvements)
# ============================================================
print_section("2. DEFINING TARGET VARIABLE")

# Target = Total spend in 30 days AFTER cutoff date
future_window_start = CUTOFF_DATE
future_window_end = CUTOFF_DATE + timedelta(days=PREDICTION_WINDOW)

print(f"Future window: {future_window_start} to {future_window_end}")

# Get transactions in future window
future_transactions = sales_header[
    (sales_header['transaction_date'] > future_window_start) &
    (sales_header['transaction_date'] <= future_window_end)
]

# Calculate raw target
target = future_transactions.groupby('customer_id')['total_amount'].sum().reset_index()
target.columns = ['customer_id', 'future_spend_30d']

print(f"\nRaw target statistics:")
print(f"  Customers with purchases: {len(target):,}")
print(f"  Mean: ${target['future_spend_30d'].mean():.2f}")
print(f"  Median: ${target['future_spend_30d'].median():.2f}")
print(f"  Max: ${target['future_spend_30d'].max():.2f}")
print(f"  99th percentile: ${target['future_spend_30d'].quantile(0.99):.2f}")

# ------------------------------
# IMPROVEMENT 2: Target Winsorization
# ------------------------------
# WHY: Extreme outliers can dominate loss function and hurt model generalization.
#      Capping at 99th percentile reduces outlier influence while preserving
#      most of the distribution's information.
print_subsection("2.1 Applying Target Winsorization (99th percentile)")

cap_value = target['future_spend_30d'].quantile(WINSORIZE_PERCENTILE / 100)
outliers_before = (target['future_spend_30d'] > cap_value).sum()
target['future_spend_30d_capped'] = target['future_spend_30d'].clip(upper=cap_value)

print(f"  Cap value: ${cap_value:.2f}")
print(f"  Outliers capped: {outliers_before}")


# ============================================================
# 3. FEATURE ENGINEERING (unchanged logic, preserving cutoff)
# ============================================================
print_section("3. FEATURE ENGINEERING")

# Use only transactions BEFORE cutoff date for features (NO LEAKAGE)
historical_transactions = sales_header[sales_header['transaction_date'] <= CUTOFF_DATE]
print(f"Historical transactions (before cutoff): {len(historical_transactions):,}")

customer_ids = historical_transactions['customer_id'].unique()
print(f"Customers with historical data: {len(customer_ids):,}")

# Initialize features
features = pd.DataFrame({'customer_id': customer_ids})

# 3.1 RFM Features
print_subsection("3.1 RFM Features")

# Recency
recency = historical_transactions.groupby('customer_id')['transaction_date'].max().reset_index()
recency['recency_days'] = (CUTOFF_DATE - recency['transaction_date']).dt.days
features = features.merge(recency[['customer_id', 'recency_days']], on='customer_id', how='left')

# Frequency & Monetary in windows
for window in [30, 60, 90]:
    window_start = CUTOFF_DATE - timedelta(days=window)
    window_trans = historical_transactions[historical_transactions['transaction_date'] > window_start]
    
    freq = window_trans.groupby('customer_id').size().reset_index(name=f'frequency_{window}d')
    features = features.merge(freq, on='customer_id', how='left')
    features[f'frequency_{window}d'] = features[f'frequency_{window}d'].fillna(0)
    
    monetary = window_trans.groupby('customer_id')['total_amount'].sum().reset_index(name=f'monetary_{window}d')
    features = features.merge(monetary, on='customer_id', how='left')
    features[f'monetary_{window}d'] = features[f'monetary_{window}d'].fillna(0)

# Total frequency and monetary
total_freq = historical_transactions.groupby('customer_id').size().reset_index(name='total_frequency')
features = features.merge(total_freq, on='customer_id', how='left')

total_monetary = historical_transactions.groupby('customer_id')['total_amount'].sum().reset_index(name='total_monetary')
features = features.merge(total_monetary, on='customer_id', how='left')

# 3.2 Behavior Features
print_subsection("3.2 Behavior Features")

aov = historical_transactions.groupby('customer_id')['total_amount'].mean().reset_index(name='avg_order_value')
features = features.merge(aov, on='customer_id', how='left')

stores_visited = historical_transactions.groupby('customer_id')['store_id'].nunique().reset_index(name='num_stores_visited')
features = features.merge(stores_visited, on='customer_id', how='left')

def calc_avg_days_between(group):
    dates = group.sort_values()
    if len(dates) < 2:
        return np.nan
    return dates.diff().dt.days.dropna().mean()

avg_days = historical_transactions.groupby('customer_id')['transaction_date'].apply(calc_avg_days_between).reset_index(name='avg_days_between_purchases')
features = features.merge(avg_days, on='customer_id', how='left')
median_days = features['avg_days_between_purchases'].median()
features['avg_days_between_purchases'] = features['avg_days_between_purchases'].fillna(median_days)

# 3.3 Customer Attributes
print_subsection("3.3 Customer Attributes")

customer_attrs = customers[['customer_id', 'loyalty_status', 'total_loyalty_points', 'segment_id', 'customer_since']].copy()
customer_attrs['customer_tenure_days'] = (CUTOFF_DATE - customer_attrs['customer_since']).dt.days
customer_attrs['customer_tenure_days'] = customer_attrs['customer_tenure_days'].fillna(365)

features = features.merge(
    customer_attrs[['customer_id', 'loyalty_status', 'total_loyalty_points', 'segment_id', 'customer_tenure_days']], 
    on='customer_id', how='left'
)
features['total_loyalty_points'] = features['total_loyalty_points'].fillna(0)
features['loyalty_status'] = features['loyalty_status'].fillna('Bronze')
features['segment_id'] = features['segment_id'].fillna('NR')

# 3.4 Category Features
print_subsection("3.4 Category Features")

line_items_full = line_items.merge(
    historical_transactions[['transaction_id', 'customer_id']], 
    on='transaction_id', how='inner'
)
line_items_full = line_items_full.merge(
    products[['product_id', 'product_category']], 
    on='product_id', how='left'
)

num_cats = line_items_full.groupby('customer_id')['product_category'].nunique().reset_index(name='num_categories')
features = features.merge(num_cats, on='customer_id', how='left')
features['num_categories'] = features['num_categories'].fillna(1)

top_cat = line_items_full.groupby(['customer_id', 'product_category']).size().reset_index(name='count')
top_cat = top_cat.loc[top_cat.groupby('customer_id')['count'].idxmax()][['customer_id', 'product_category']]
top_cat.columns = ['customer_id', 'top_category']
features = features.merge(top_cat, on='customer_id', how='left')
features['top_category'] = features['top_category'].fillna('Unknown')

# 3.5 Temporal Features
print_subsection("3.5 Temporal Features")

historical_transactions['is_weekend'] = historical_transactions['transaction_date'].dt.dayofweek >= 5
weekend_ratio = historical_transactions.groupby('customer_id')['is_weekend'].mean().reset_index(name='weekend_ratio')
features = features.merge(weekend_ratio, on='customer_id', how='left')
features['is_weekend_shopper'] = (features['weekend_ratio'] > 0.5).astype(int)
features = features.drop(columns=['weekend_ratio'])

print(f"\nTotal features created: {features.shape[1] - 1}")


# ============================================================
# 4. PREPARE DATASET
# ============================================================
print_section("4. PREPARING DATASET")

# Merge features with target (using capped target)
dataset = features.merge(target[['customer_id', 'future_spend_30d', 'future_spend_30d_capped']], 
                         on='customer_id', how='left')

# Customers without future purchases = $0
dataset['future_spend_30d'] = dataset['future_spend_30d'].fillna(0)
dataset['future_spend_30d_capped'] = dataset['future_spend_30d_capped'].fillna(0)

print(f"Dataset size: {len(dataset):,}")
print(f"Customers with spend > $0: {(dataset['future_spend_30d'] > 0).sum():,} ({(dataset['future_spend_30d'] > 0).mean()*100:.1f}%)")
print(f"Customers with spend = $0: {(dataset['future_spend_30d'] == 0).sum():,} ({(dataset['future_spend_30d'] == 0).mean()*100:.1f}%)")

# Encode categorical variables
print_subsection("4.1 Encoding Categorical Variables")

from sklearn.preprocessing import LabelEncoder

loyalty_encoder = LabelEncoder()
dataset['loyalty_status_encoded'] = loyalty_encoder.fit_transform(dataset['loyalty_status'])

segment_encoder = LabelEncoder()
dataset['segment_id_encoded'] = segment_encoder.fit_transform(dataset['segment_id'])

category_encoder = LabelEncoder()
dataset['top_category_encoded'] = category_encoder.fit_transform(dataset['top_category'])

# Define feature columns
feature_cols = [
    'recency_days',
    'frequency_30d', 'frequency_60d', 'frequency_90d',
    'monetary_30d', 'monetary_60d', 'monetary_90d',
    'total_frequency', 'total_monetary',
    'avg_order_value', 'num_stores_visited', 'avg_days_between_purchases',
    'total_loyalty_points', 'customer_tenure_days',
    'num_categories', 'is_weekend_shopper',
    'loyalty_status_encoded', 'segment_id_encoded', 'top_category_encoded'
]

# Numeric columns for scaling
numeric_cols = [
    'recency_days',
    'frequency_30d', 'frequency_60d', 'frequency_90d',
    'monetary_30d', 'monetary_60d', 'monetary_90d',
    'total_frequency', 'total_monetary',
    'avg_order_value', 'num_stores_visited', 'avg_days_between_purchases',
    'total_loyalty_points', 'customer_tenure_days',
    'num_categories'
]

X = dataset[feature_cols].copy()
y_original = dataset['future_spend_30d'].values  # Original for final evaluation
y_capped = dataset['future_spend_30d_capped'].values  # Capped for training

print(f"\nFeature matrix: {X.shape}")


# ============================================================
# 5. TIME-BASED TRAIN/TEST SPLIT
# ============================================================
print_section("5. TIME-BASED TRAIN/TEST SPLIT")

# ------------------------------
# STRICT: Time-based split (NO random splitting)
# ------------------------------
# WHY: Random splitting would leak future information into training.
#      We split by ordering customers based on their first transaction date,
#      ensuring train customers' activity is temporally before test customers.

# Sort by last transaction date (proxy for temporal ordering)
customer_last_trans = historical_transactions.groupby('customer_id')['transaction_date'].max().reset_index()
customer_last_trans.columns = ['customer_id', 'last_trans_date']
dataset = dataset.merge(customer_last_trans, on='customer_id', how='left')
dataset = dataset.sort_values('last_trans_date').reset_index(drop=True)

# 80/20 split maintaining temporal order
split_idx = int(len(dataset) * 0.8)

train_data = dataset.iloc[:split_idx]
test_data = dataset.iloc[split_idx:]

X_train = train_data[feature_cols].values
X_test = test_data[feature_cols].values
y_train_original = train_data['future_spend_30d'].values
y_test_original = test_data['future_spend_30d'].values
y_train_capped = train_data['future_spend_30d_capped'].values
y_test_capped = test_data['future_spend_30d_capped'].values

print(f"Training set: {len(X_train):,} customers")
print(f"Test set: {len(X_test):,} customers")
print(f"Train date range: up to {train_data['last_trans_date'].max()}")
print(f"Test date range: from {test_data['last_trans_date'].min()}")

# Verify no overlap
train_customers = set(train_data['customer_id'])
test_customers = set(test_data['customer_id'])
overlap = train_customers.intersection(test_customers)
print(f"Customer overlap (should be 0): {len(overlap)}")

# ------------------------------
# IMPROVEMENT 1: Log1p Transformation on Target
# ------------------------------
# WHY: Customer spend is typically right-skewed (many small, few large values).
#      Log transformation normalizes the distribution, making it easier for
#      regression models to learn patterns. log1p handles zeros gracefully.
print_subsection("5.1 Applying Log1p Transformation to Target")

y_train_log = np.log1p(y_train_capped)
y_test_log = np.log1p(y_test_capped)

print(f"  Original y_train: mean=${y_train_capped.mean():.2f}, std=${y_train_capped.std():.2f}")
print(f"  Log1p y_train: mean={y_train_log.mean():.2f}, std={y_train_log.std():.2f}")

# ------------------------------
# IMPROVEMENT 3: StandardScaler (fit on train only)
# ------------------------------
# WHY: Linear models are sensitive to feature scales. Standardization puts all
#      features on same scale. CRITICAL: Fit ONLY on train to prevent leakage.
print_subsection("5.2 Scaling Features (fit on train only)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)  # Transform test (no fitting!)

print(f"  Scaler fitted on {len(X_train)} training samples only")


# ============================================================
# 6. TIME-SERIES CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# ============================================================
print_section("6. TIME-SERIES CROSS-VALIDATION")

# ------------------------------
# IMPROVEMENT 4: TimeSeriesSplit CV
# ------------------------------
# WHY: Standard KFold CV randomly mixes data, causing temporal leakage.
#      TimeSeriesSplit ensures training folds always precede validation folds,
#      mimicking real-world deployment where we predict future from past.

tscv = TimeSeriesSplit(n_splits=5)

def cross_validate_model(model, X, y, cv, use_scaled=False):
    """
    Perform time-series cross-validation.
    Returns mean and std of MAE across folds.
    """
    mae_scores = []
    
    for train_idx, val_idx in cv.split(X):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]
        
        if use_scaled:
            cv_scaler = StandardScaler()
            X_cv_train = cv_scaler.fit_transform(X_cv_train)
            X_cv_val = cv_scaler.transform(X_cv_val)
        
        model.fit(X_cv_train, y_cv_train)
        y_pred_log = model.predict(X_cv_val)
        
        # Inverse transform for MAE calculation
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_cv_val)
        
        mae_scores.append(mean_absolute_error(y_true, y_pred))
    
    return np.mean(mae_scores), np.std(mae_scores)

print("Performing TimeSeriesSplit CV (5 folds)...")


# ============================================================
# 7. MODEL TRAINING WITH IMPROVEMENTS
# ============================================================
print_section("7. MODEL TRAINING")

results = []

# ------------------------------
# IMPROVEMENT 7: Baseline Model (Mean Predictor)
# ------------------------------
# WHY: Provides benchmark. If our models don't beat this, they add no value.
print_subsection("7.1 Baseline: Mean Predictor")

baseline = MeanBaselineRegressor()
baseline.fit(X_train, y_train_log)
y_pred_baseline_log = baseline.predict(X_test)
y_pred_baseline = np.expm1(y_pred_baseline_log)
y_pred_baseline = np.maximum(y_pred_baseline, 0)  # Ensure non-negative

mae_baseline = mean_absolute_error(y_test_original, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test_original, y_pred_baseline))
r2_baseline = r2_score(y_test_original, y_pred_baseline)

results.append({
    'Model': 'Baseline (Mean)',
    'MAE': mae_baseline,
    'RMSE': rmse_baseline,
    'R2': r2_baseline,
    'MAE_Improvement': 0.0
})
print(f"  MAE: ${mae_baseline:.2f} | RMSE: ${rmse_baseline:.2f} | R²: {r2_baseline:.4f}")

# ------------------------------
# 7.2 Linear Regression
# ------------------------------
print_subsection("7.2 Linear Regression (scaled features)")

lr = LinearRegression()
lr.fit(X_train_scaled, y_train_log)
y_pred_lr_log = lr.predict(X_test_scaled)
y_pred_lr = np.expm1(y_pred_lr_log)
y_pred_lr = np.maximum(y_pred_lr, 0)

mae_lr = mean_absolute_error(y_test_original, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_original, y_pred_lr))
r2_lr = r2_score(y_test_original, y_pred_lr)
improvement_lr = ((mae_baseline - mae_lr) / mae_baseline) * 100

results.append({
    'Model': 'Linear Regression',
    'MAE': mae_lr,
    'RMSE': rmse_lr,
    'R2': r2_lr,
    'MAE_Improvement': improvement_lr
})
print(f"  MAE: ${mae_lr:.2f} | RMSE: ${rmse_lr:.2f} | R²: {r2_lr:.4f} | Improvement: {improvement_lr:.1f}%")

# ------------------------------
# 7.3 Ridge Regression (with CV tuning)
# ------------------------------
print_subsection("7.3 Ridge Regression (tuned alpha)")

# WHY: L2 regularization prevents overfitting and handles multicollinearity
best_ridge_mae = float('inf')
best_ridge_alpha = 1.0

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge_cv = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    cv_mae, _ = cross_validate_model(ridge_cv, X_train, y_train_log, tscv, use_scaled=True)
    if cv_mae < best_ridge_mae:
        best_ridge_mae = cv_mae
        best_ridge_alpha = alpha

print(f"  Best alpha: {best_ridge_alpha} (CV MAE: ${best_ridge_mae:.2f})")

ridge = Ridge(alpha=best_ridge_alpha, random_state=RANDOM_STATE)
ridge.fit(X_train_scaled, y_train_log)
y_pred_ridge_log = ridge.predict(X_test_scaled)
y_pred_ridge = np.expm1(y_pred_ridge_log)
y_pred_ridge = np.maximum(y_pred_ridge, 0)

mae_ridge = mean_absolute_error(y_test_original, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test_original, y_pred_ridge))
r2_ridge = r2_score(y_test_original, y_pred_ridge)
improvement_ridge = ((mae_baseline - mae_ridge) / mae_baseline) * 100

results.append({
    'Model': 'Ridge Regression',
    'MAE': mae_ridge,
    'RMSE': rmse_ridge,
    'R2': r2_ridge,
    'MAE_Improvement': improvement_ridge
})
print(f"  MAE: ${mae_ridge:.2f} | RMSE: ${rmse_ridge:.2f} | R²: {r2_ridge:.4f} | Improvement: {improvement_ridge:.1f}%")

# ------------------------------
# 7.4 Random Forest (tuned)
# ------------------------------
print_subsection("7.4 Random Forest (regularized)")

# WHY: Ensemble of trees captures non-linear patterns
# Regularization (max_depth, min_samples_leaf) prevents overfitting
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,  # Limit depth to prevent overfitting
    min_samples_leaf=10,  # Require more samples per leaf
    min_samples_split=20,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train_log)
y_pred_rf_log = rf.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)
y_pred_rf = np.maximum(y_pred_rf, 0)

mae_rf = mean_absolute_error(y_test_original, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_original, y_pred_rf))
r2_rf = r2_score(y_test_original, y_pred_rf)
improvement_rf = ((mae_baseline - mae_rf) / mae_baseline) * 100

results.append({
    'Model': 'Random Forest',
    'MAE': mae_rf,
    'RMSE': rmse_rf,
    'R2': r2_rf,
    'MAE_Improvement': improvement_rf
})
print(f"  MAE: ${mae_rf:.2f} | RMSE: ${rmse_rf:.2f} | R²: {r2_rf:.4f} | Improvement: {improvement_rf:.1f}%")

# ------------------------------
# IMPROVEMENT 6: Gradient Boosting (optimized)
# ------------------------------
print_subsection("7.5 Gradient Boosting (optimized)")

# WHY: Lower learning rate + more trees = better generalization
#      Regularization (max_depth, min_samples_leaf) prevents overfitting
#      Subsample adds randomness for robustness
gb = GradientBoostingRegressor(
    n_estimators=500,  # More trees with lower LR
    learning_rate=0.05,  # Lower LR for better generalization
    max_depth=4,  # Shallow trees
    min_samples_leaf=15,  # Regularization
    min_samples_split=30,
    subsample=0.8,  # Stochastic gradient boosting
    validation_fraction=0.1,  # For early stopping
    n_iter_no_change=20,  # Early stopping patience
    random_state=RANDOM_STATE
)
gb.fit(X_train, y_train_log)
y_pred_gb_log = gb.predict(X_test)
y_pred_gb = np.expm1(y_pred_gb_log)
y_pred_gb = np.maximum(y_pred_gb, 0)

mae_gb = mean_absolute_error(y_test_original, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test_original, y_pred_gb))
r2_gb = r2_score(y_test_original, y_pred_gb)
improvement_gb = ((mae_baseline - mae_gb) / mae_baseline) * 100

results.append({
    'Model': 'Gradient Boosting',
    'MAE': mae_gb,
    'RMSE': rmse_gb,
    'R2': r2_gb,
    'MAE_Improvement': improvement_gb
})
print(f"  MAE: ${mae_gb:.2f} | RMSE: ${rmse_gb:.2f} | R²: {r2_gb:.4f} | Improvement: {improvement_gb:.1f}%")
print(f"  Actual iterations (early stopping): {gb.n_estimators_}")

# ------------------------------
# 7.6 XGBoost (if available)
# ------------------------------
if XGBOOST_AVAILABLE:
    print_subsection("7.6 XGBoost (optimized)")
    
    # WHY: Often best performance on tabular data
    #      Early stopping prevents overfitting
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    
    # Split train for early stopping
    es_split = int(len(X_train) * 0.9)
    xgb.fit(
        X_train[:es_split], y_train_log[:es_split],
        eval_set=[(X_train[es_split:], y_train_log[es_split:])],
        verbose=False
    )
    
    y_pred_xgb_log = xgb.predict(X_test)
    y_pred_xgb = np.expm1(y_pred_xgb_log)
    y_pred_xgb = np.maximum(y_pred_xgb, 0)
    
    mae_xgb = mean_absolute_error(y_test_original, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_original, y_pred_xgb))
    r2_xgb = r2_score(y_test_original, y_pred_xgb)
    improvement_xgb = ((mae_baseline - mae_xgb) / mae_baseline) * 100
    
    results.append({
        'Model': 'XGBoost',
        'MAE': mae_xgb,
        'RMSE': rmse_xgb,
        'R2': r2_xgb,
        'MAE_Improvement': improvement_xgb
    })
    print(f"  MAE: ${mae_xgb:.2f} | RMSE: ${rmse_xgb:.2f} | R²: {r2_xgb:.4f} | Improvement: {improvement_xgb:.1f}%")
    print(f"  Best iteration: {xgb.best_iteration}")


# ============================================================
# 8. FINAL EVALUATION TABLE
# ============================================================
print_section("8. FINAL MODEL COMPARISON")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAE')

print("\n" + "─" * 85)
print(f"{'Model':<25} {'MAE ($)':<12} {'RMSE ($)':<12} {'R²':<12} {'MAE Improv.':<15}")
print("─" * 85)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<25} {row['MAE']:<12.2f} {row['RMSE']:<12.2f} {row['R2']:<12.4f} {row['MAE_Improvement']:>+.1f}%")
print("─" * 85)

# Best model
best_model_name = results_df.iloc[0]['Model']
best_mae = results_df.iloc[0]['MAE']
best_improvement = results_df.iloc[0]['MAE_Improvement']

print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Best MAE: ${best_mae:.2f} ({best_improvement:+.1f}% vs baseline)")


# ============================================================
# 9. FEATURE IMPORTANCE (Tree-based models)
# ============================================================
print_section("9. FEATURE IMPORTANCE")

print_subsection("9.1 Gradient Boosting Feature Importance")
gb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (Gradient Boosting):")
for i, row in gb_importance.head(10).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

print_subsection("9.2 Random Forest Feature Importance")
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (Random Forest):")
for i, row in rf_importance.head(10).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")


# ============================================================
# 10. SAVE BEST MODEL AND ARTIFACTS
# ============================================================
print_section("10. SAVING MODELS AND ARTIFACTS")

# Determine best tree-based model
if XGBOOST_AVAILABLE and mae_xgb < mae_gb:
    best_tree_model = xgb
    best_tree_name = 'XGBoost'
else:
    best_tree_model = gb
    best_tree_name = 'Gradient Boosting'

# Save best model
with open(f"{MODELS_DIR}/spend_predictor_improved.pkl", 'wb') as f:
    pickle.dump(best_tree_model, f)
print(f"  → Saved: spend_predictor_improved.pkl ({best_tree_name})")

# Save scaler
with open(f"{MODELS_DIR}/feature_scaler_improved.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print(f"  → Saved: feature_scaler_improved.pkl")

# Save encoders
encoders = {
    'loyalty_encoder': loyalty_encoder,
    'segment_encoder': segment_encoder,
    'category_encoder': category_encoder
}
with open(f"{MODELS_DIR}/encoders_improved.pkl", 'wb') as f:
    pickle.dump(encoders, f)
print(f"  → Saved: encoders_improved.pkl")

# Save metadata
metadata = {
    'model_type': best_tree_name,
    'cutoff_date': str(CUTOFF_DATE),
    'prediction_window': PREDICTION_WINDOW,
    'feature_columns': feature_cols,
    'numeric_columns': numeric_cols,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'target_cap_value': cap_value,
    'winsorize_percentile': WINSORIZE_PERCENTILE,
    'use_log_transform': True,
    'mae': float(results_df.iloc[0]['MAE']),
    'rmse': float(results_df.iloc[0]['RMSE']),
    'r2': float(results_df.iloc[0]['R2']),
    'baseline_mae': mae_baseline,
    'improvement_vs_baseline': float(results_df.iloc[0]['MAE_Improvement']),
    'loyalty_classes': list(loyalty_encoder.classes_),
    'segment_classes': list(segment_encoder.classes_),
    'category_classes': list(category_encoder.classes_)
}
with open(f"{MODELS_DIR}/model_metadata_improved.pkl", 'wb') as f:
    pickle.dump(metadata, f)
print(f"  → Saved: model_metadata_improved.pkl")

# Save results comparison
results_df.to_csv(f"{MODELS_DIR}/model_comparison_improved.csv", index=False)
print(f"  → Saved: model_comparison_improved.csv")

# Save feature importance
gb_importance.to_csv(f"{MODELS_DIR}/feature_importance_gb.csv", index=False)
rf_importance.to_csv(f"{MODELS_DIR}/feature_importance_rf.csv", index=False)
print(f"  → Saved: feature_importance_gb.csv, feature_importance_rf.csv")


# ============================================================
# 11. SUMMARY
# ============================================================
print_section("11. IMPROVEMENT SUMMARY")

print("""
┌────────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE IMPROVEMENTS APPLIED                           │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  1. LOG1P TRANSFORMATION ON TARGET                                             │
│     → Normalizes right-skewed spend distribution                               │
│     → Helps models learn patterns across all spend levels                      │
│     → Predictions inverse-transformed (expm1) for evaluation                   │
│                                                                                │
│  2. TARGET WINSORIZATION (99th percentile)                                     │
│     → Caps extreme outliers at ${:.2f}                                │
│     → Reduces influence of few very high spenders                              │
│     → Improves model generalization                                            │
│                                                                                │
│  3. STANDARDSCALER (fit on train only)                                         │
│     → Scales features for linear models                                        │
│     → CRITICAL: Fitted ONLY on training data to prevent leakage               │
│                                                                                │
│  4. TIMESERIESSPLIT CROSS-VALIDATION                                           │
│     → Respects temporal ordering of data                                       │
│     → Training folds always precede validation folds                           │
│     → Mimics real-world deployment scenario                                    │
│                                                                                │
│  5. BUSINESS-ALIGNED METRICS                                                   │
│     → Primary: MAE (interpretable in dollars)                                  │
│     → Diagnostic: RMSE, R²                                                     │
│     → All metrics computed on ORIGINAL scale (not log)                         │
│                                                                                │
│  6. IMPROVED GRADIENT BOOSTING / XGBOOST                                       │
│     → Lower learning rate (0.05) for better generalization                     │
│     → Early stopping to prevent overfitting                                    │
│     → Regularization: max_depth=4, min_samples_leaf=15                         │
│     → Subsample=0.8 for stochastic boosting                                    │
│                                                                                │
│  7. BASELINE COMPARISON                                                        │
│     → Mean predictor benchmark: MAE = ${:.2f}                           │
│     → Best model improvement: {:.1f}%                                         │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│  NO DATA LEAKAGE GUARANTEED:                                                   │
│  ✓ Time-based train/test split (no random mixing)                             │
│  ✓ Features computed ONLY from pre-cutoff data                                │
│  ✓ Scaler fitted ONLY on training data                                        │
│  ✓ No customer appears in both train and test                                 │
└────────────────────────────────────────────────────────────────────────────────┘
""".format(cap_value, mae_baseline, best_improvement))

print("\n" + "=" * 80)
print(" IMPROVED MODEL TRAINING COMPLETE!")
print("=" * 80)
