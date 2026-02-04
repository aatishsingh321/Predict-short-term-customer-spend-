"""
================================================================================
FEATURE ENGINEERING AND MODEL TRAINING SCRIPT
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Create customer-level features, define target variable, train model

Input:  data/cleaned/*.csv
Output: models/spend_predictor.pkl (trained model)
        models/feature_scaler.pkl (feature scaler)
        data/cleaned/customer_features_full.csv (feature dataset)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================
CLEANED_DATA_DIR = "/Users/apple/Customer Spend Predictor/data/cleaned"
MODELS_DIR = "/Users/apple/Customer Spend Predictor/models"

# Cutoff date for train/test split (30 days before last transaction)
# Features are computed BEFORE cutoff, target is AFTER cutoff
CUTOFF_DATE = pd.Timestamp('2025-08-01')
PREDICTION_WINDOW = 30  # days

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subsection(title):
    print(f"\n--- {title} ---")


# ============================================================
# 1. LOAD DATA
# ============================================================
print_section("1. LOADING CLEANED DATA")

sales_header = pd.read_csv(f"{CLEANED_DATA_DIR}/store_sales_header.csv")
sales_header['transaction_date'] = pd.to_datetime(sales_header['transaction_date'])

line_items = pd.read_csv(f"{CLEANED_DATA_DIR}/store_sales_line_items.csv")
customers = pd.read_csv(f"{CLEANED_DATA_DIR}/customer_details.csv")
products = pd.read_csv(f"{CLEANED_DATA_DIR}/products.csv")
stores = pd.read_csv(f"{CLEANED_DATA_DIR}/stores.csv")

customers['customer_since'] = pd.to_datetime(customers['customer_since'], errors='coerce')

print(f"Sales transactions: {len(sales_header):,}")
print(f"Line items: {len(line_items):,}")
print(f"Customers: {len(customers):,}")
print(f"Date range: {sales_header['transaction_date'].min()} to {sales_header['transaction_date'].max()}")
print(f"Cutoff date: {CUTOFF_DATE}")


# ============================================================
# 2. DEFINE TARGET VARIABLE
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

# Calculate target: sum of spend per customer in future window
target = future_transactions.groupby('customer_id')['total_amount'].sum().reset_index()
target.columns = ['customer_id', 'future_spend_30d']

print(f"Customers with future purchases: {len(target):,}")
print(f"Average future spend: ${target['future_spend_30d'].mean():.2f}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print_section("3. FEATURE ENGINEERING")

# Use only transactions BEFORE cutoff date for features
historical_transactions = sales_header[sales_header['transaction_date'] <= CUTOFF_DATE]
print(f"Historical transactions (before cutoff): {len(historical_transactions):,}")

# Get unique customers with historical transactions
customer_ids = historical_transactions['customer_id'].unique()
print(f"Customers with historical data: {len(customer_ids):,}")

# Initialize feature dataframe
features = pd.DataFrame({'customer_id': customer_ids})

# ------------------------------
# 3.1 RFM Features
# ------------------------------
print_subsection("3.1 Computing RFM Features")

# Recency: Days since last purchase (before cutoff)
recency = historical_transactions.groupby('customer_id')['transaction_date'].max().reset_index()
recency['recency_days'] = (CUTOFF_DATE - recency['transaction_date']).dt.days
features = features.merge(recency[['customer_id', 'recency_days']], on='customer_id', how='left')

# Frequency: Number of transactions in different windows
for window in [30, 60, 90]:
    window_start = CUTOFF_DATE - timedelta(days=window)
    window_trans = historical_transactions[historical_transactions['transaction_date'] > window_start]
    freq = window_trans.groupby('customer_id').size().reset_index(name=f'frequency_{window}d')
    features = features.merge(freq, on='customer_id', how='left')
    features[f'frequency_{window}d'] = features[f'frequency_{window}d'].fillna(0)

# Monetary: Total spend in different windows
for window in [30, 60, 90]:
    window_start = CUTOFF_DATE - timedelta(days=window)
    window_trans = historical_transactions[historical_transactions['transaction_date'] > window_start]
    monetary = window_trans.groupby('customer_id')['total_amount'].sum().reset_index(name=f'monetary_{window}d')
    features = features.merge(monetary, on='customer_id', how='left')
    features[f'monetary_{window}d'] = features[f'monetary_{window}d'].fillna(0)

# Total historical frequency and monetary
total_freq = historical_transactions.groupby('customer_id').size().reset_index(name='total_frequency')
features = features.merge(total_freq, on='customer_id', how='left')

total_monetary = historical_transactions.groupby('customer_id')['total_amount'].sum().reset_index(name='total_monetary')
features = features.merge(total_monetary, on='customer_id', how='left')

print(f"  RFM features computed")

# ------------------------------
# 3.2 Transaction Behavior Features
# ------------------------------
print_subsection("3.2 Computing Transaction Behavior Features")

# Average order value
aov = historical_transactions.groupby('customer_id')['total_amount'].mean().reset_index(name='avg_order_value')
features = features.merge(aov, on='customer_id', how='left')

# Number of unique stores visited
stores_visited = historical_transactions.groupby('customer_id')['store_id'].nunique().reset_index(name='num_stores_visited')
features = features.merge(stores_visited, on='customer_id', how='left')

# Average days between purchases
def calc_avg_days_between(group):
    dates = group.sort_values()
    if len(dates) < 2:
        return np.nan
    diffs = dates.diff().dt.days.dropna()
    return diffs.mean()

avg_days = historical_transactions.groupby('customer_id')['transaction_date'].apply(calc_avg_days_between).reset_index(name='avg_days_between_purchases')
features = features.merge(avg_days, on='customer_id', how='left')
features['avg_days_between_purchases'] = features['avg_days_between_purchases'].fillna(features['avg_days_between_purchases'].median())

print(f"  Transaction behavior features computed")

# ------------------------------
# 3.3 Customer Attributes
# ------------------------------
print_subsection("3.3 Adding Customer Attributes")

# Merge customer attributes
customer_attrs = customers[['customer_id', 'loyalty_status', 'total_loyalty_points', 'segment_id', 'customer_since']].copy()

# Calculate tenure
customer_attrs['customer_tenure_days'] = (CUTOFF_DATE - customer_attrs['customer_since']).dt.days
customer_attrs['customer_tenure_days'] = customer_attrs['customer_tenure_days'].fillna(365)  # default

features = features.merge(customer_attrs[['customer_id', 'loyalty_status', 'total_loyalty_points', 'segment_id', 'customer_tenure_days']], 
                          on='customer_id', how='left')

# Fill missing values
features['total_loyalty_points'] = features['total_loyalty_points'].fillna(0)
features['loyalty_status'] = features['loyalty_status'].fillna('Bronze')
features['segment_id'] = features['segment_id'].fillna('NR')

print(f"  Customer attributes added")

# ------------------------------
# 3.4 Category Preferences
# ------------------------------
print_subsection("3.4 Computing Category Preferences")

# Merge line items with products and transactions
line_items_full = line_items.merge(
    historical_transactions[['transaction_id', 'customer_id']], 
    on='transaction_id', 
    how='inner'
)
line_items_full = line_items_full.merge(
    products[['product_id', 'product_category']], 
    on='product_id', 
    how='left'
)

# Number of unique categories
num_cats = line_items_full.groupby('customer_id')['product_category'].nunique().reset_index(name='num_categories')
features = features.merge(num_cats, on='customer_id', how='left')
features['num_categories'] = features['num_categories'].fillna(1)

# Top category (most purchased)
top_cat = line_items_full.groupby(['customer_id', 'product_category']).size().reset_index(name='count')
top_cat = top_cat.loc[top_cat.groupby('customer_id')['count'].idxmax()][['customer_id', 'product_category']]
top_cat.columns = ['customer_id', 'top_category']
features = features.merge(top_cat, on='customer_id', how='left')
features['top_category'] = features['top_category'].fillna('Unknown')

print(f"  Category preferences computed")

# ------------------------------
# 3.5 Temporal Features
# ------------------------------
print_subsection("3.5 Computing Temporal Features")

# Is weekend shopper (>50% of transactions on Sat/Sun)
historical_transactions['is_weekend'] = historical_transactions['transaction_date'].dt.dayofweek >= 5
weekend_ratio = historical_transactions.groupby('customer_id')['is_weekend'].mean().reset_index(name='weekend_ratio')
features = features.merge(weekend_ratio, on='customer_id', how='left')
features['is_weekend_shopper'] = (features['weekend_ratio'] > 0.5).astype(int)
features = features.drop(columns=['weekend_ratio'])

print(f"  Temporal features computed")

# ------------------------------
# 3.6 Final Feature Set
# ------------------------------
print_subsection("3.6 Final Feature Set")

print(f"\nFeature columns: {features.shape[1] - 1}")  # exclude customer_id
print(features.columns.tolist())


# ============================================================
# 4. PREPARE TRAINING DATA
# ============================================================
print_section("4. PREPARING TRAINING DATA")

# Merge features with target
dataset = features.merge(target, on='customer_id', how='left')

# Customers without future purchases have target = 0
dataset['future_spend_30d'] = dataset['future_spend_30d'].fillna(0)

print(f"Total customers in dataset: {len(dataset):,}")
print(f"Customers with future spend > 0: {(dataset['future_spend_30d'] > 0).sum():,}")
print(f"Customers with future spend = 0: {(dataset['future_spend_30d'] == 0).sum():,}")

# Encode categorical variables
print_subsection("4.1 Encoding Categorical Variables")

# Loyalty Status
loyalty_encoder = LabelEncoder()
dataset['loyalty_status_encoded'] = loyalty_encoder.fit_transform(dataset['loyalty_status'])

# Segment ID
segment_encoder = LabelEncoder()
dataset['segment_id_encoded'] = segment_encoder.fit_transform(dataset['segment_id'])

# Top Category
category_encoder = LabelEncoder()
dataset['top_category_encoded'] = category_encoder.fit_transform(dataset['top_category'])

print(f"  Loyalty status classes: {list(loyalty_encoder.classes_)}")
print(f"  Segment classes: {list(segment_encoder.classes_)}")
print(f"  Category classes: {len(category_encoder.classes_)} unique")

# Define feature columns for modeling
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

X = dataset[feature_cols]
y = dataset['future_spend_30d']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================
print_section("5. TRAIN/TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. MODEL TRAINING
# ============================================================
print_section("6. MODEL TRAINING")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []

for name, model in models.items():
    print_subsection(f"Training {name}")
    
    # Use scaled data for linear models, original for tree-based
    if 'Linear' in name or 'Ridge' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })
    
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

# Results summary
print_subsection("Model Comparison")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select best model (lowest MAE)
best_model_name = results_df.loc[results_df['MAE'].idxmin(), 'Model']
print(f"\n✓ Best Model: {best_model_name}")


# ============================================================
# 7. SAVE MODEL AND ARTIFACTS
# ============================================================
print_section("7. SAVING MODEL AND ARTIFACTS")

# Use Gradient Boosting as final model (typically best for this type of problem)
final_model = models['Gradient Boosting']

# Save model
with open(f"{MODELS_DIR}/spend_predictor.pkl", 'wb') as f:
    pickle.dump(final_model, f)
print(f"  → Saved: spend_predictor.pkl")

# Save scaler
with open(f"{MODELS_DIR}/feature_scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print(f"  → Saved: feature_scaler.pkl")

# Save encoders
encoders = {
    'loyalty_encoder': loyalty_encoder,
    'segment_encoder': segment_encoder,
    'category_encoder': category_encoder
}
with open(f"{MODELS_DIR}/encoders.pkl", 'wb') as f:
    pickle.dump(encoders, f)
print(f"  → Saved: encoders.pkl")

# Save feature columns
with open(f"{MODELS_DIR}/feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"  → Saved: feature_columns.pkl")

# Save feature dataset
dataset.to_csv(f"{CLEANED_DATA_DIR}/customer_features_full.csv", index=False)
print(f"  → Saved: customer_features_full.csv ({len(dataset)} rows)")

# Save model metadata
metadata = {
    'model_type': 'GradientBoostingRegressor',
    'cutoff_date': str(CUTOFF_DATE),
    'prediction_window': PREDICTION_WINDOW,
    'feature_columns': feature_cols,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'mae': results_df[results_df['Model'] == 'Gradient Boosting']['MAE'].values[0],
    'rmse': results_df[results_df['Model'] == 'Gradient Boosting']['RMSE'].values[0],
    'r2': results_df[results_df['Model'] == 'Gradient Boosting']['R2'].values[0],
    'loyalty_classes': list(loyalty_encoder.classes_),
    'segment_classes': list(segment_encoder.classes_),
    'category_classes': list(category_encoder.classes_)
}

with open(f"{MODELS_DIR}/model_metadata.pkl", 'wb') as f:
    pickle.dump(metadata, f)
print(f"  → Saved: model_metadata.pkl")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
print_section("8. FEATURE IMPORTANCE")

# Get feature importance from Gradient Boosting
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10).to_string(index=False))

# Save feature importance
importance.to_csv(f"{MODELS_DIR}/feature_importance.csv", index=False)


print("\n" + "=" * 70)
print(" MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"""
Model saved to: {MODELS_DIR}/spend_predictor.pkl

Performance Metrics:
  MAE:  ${metadata['mae']:.2f}
  RMSE: ${metadata['rmse']:.2f}
  R²:   {metadata['r2']:.4f}

Ready for deployment!
""")
