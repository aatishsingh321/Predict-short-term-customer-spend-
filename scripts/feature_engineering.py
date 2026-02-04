"""
================================================================================
FEATURE ENGINEERING SCRIPT
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Create customer-level features from cleaned transaction data

Features Created:
1. RFM Features (Recency, Frequency, Monetary)
2. Product/Category Preferences
3. Channel/Store Usage Features
4. Customer Attributes (loyalty, segment, tenure)

Input:  data/cleaned/*.csv
Output: data/processed/customer_features.csv
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
# Get the project root directory (parent of scripts folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Cutoff date for train/test split (as defined in architecture doc)
CUTOFF_DATE = pd.Timestamp("2025-01-01")
PREDICTION_HORIZON_DAYS = 30

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


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================
print_section("1. LOADING CLEANED DATA")

print("Loading cleaned tables...")

# Load all cleaned tables
sales_header = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "store_sales_header.csv"))
line_items = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "store_sales_line_items.csv"))
customers = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "customer_details.csv"))
products = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "products.csv"))
stores = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "stores.csv"))

print(f"  → Sales header: {len(sales_header):,} transactions")
print(f"  → Line items: {len(line_items):,} items")
print(f"  → Customers: {len(customers):,} customers")
print(f"  → Products: {len(products):,} products")
print(f"  → Stores: {len(stores):,} stores")

# Convert transaction_date to datetime
sales_header['transaction_date'] = pd.to_datetime(sales_header['transaction_date'])

# Convert customer_since to datetime
customers['customer_since'] = pd.to_datetime(customers['customer_since'], errors='coerce')

print(f"\nTransaction date range: {sales_header['transaction_date'].min()} to {sales_header['transaction_date'].max()}")
print(f"Cutoff date: {CUTOFF_DATE}")


# ============================================================
# 2. SPLIT DATA BY CUTOFF DATE
# ============================================================
print_section("2. SPLITTING DATA BY CUTOFF DATE")

# Historical data (for features) - transactions BEFORE cutoff
historical_transactions = sales_header[sales_header['transaction_date'] < CUTOFF_DATE].copy()

# Future data (for target) - transactions in 30 days AFTER cutoff
future_end = CUTOFF_DATE + timedelta(days=PREDICTION_HORIZON_DAYS)
future_transactions = sales_header[
    (sales_header['transaction_date'] >= CUTOFF_DATE) & 
    (sales_header['transaction_date'] < future_end)
].copy()

print(f"Historical transactions (before {CUTOFF_DATE.date()}): {len(historical_transactions):,}")
print(f"Future transactions ({CUTOFF_DATE.date()} to {future_end.date()}): {len(future_transactions):,}")

# Get customers who had at least one transaction before cutoff
active_customers = historical_transactions['customer_id'].unique()
print(f"\nCustomers with historical transactions: {len(active_customers):,}")


# ============================================================
# 3. MERGE LINE ITEMS WITH PRODUCTS FOR CATEGORY INFO
# ============================================================
print_section("3. PREPARING PRODUCT CATEGORY DATA")

# Merge line items with product info
line_items_with_products = line_items.merge(
    products[['product_id', 'product_category']], 
    on='product_id', 
    how='left'
)

# Merge with sales header to get customer_id and date
transaction_details = historical_transactions.merge(
    line_items_with_products,
    on='transaction_id',
    how='left'
)

print(f"Transaction details prepared: {len(transaction_details):,} line items")


# ============================================================
# 4. CREATE RFM FEATURES
# ============================================================
print_section("4. CREATING RFM FEATURES")

print_subsection("4.1 Recency Features")

# Calculate recency for each customer (days since last purchase before cutoff)
recency = historical_transactions.groupby('customer_id').agg(
    last_purchase_date=('transaction_date', 'max')
).reset_index()

recency['recency_days'] = (CUTOFF_DATE - recency['last_purchase_date']).dt.days
recency = recency[['customer_id', 'recency_days']]

print(f"  → recency_days: Days since last purchase before cutoff")
print(f"     Mean: {recency['recency_days'].mean():.1f} days")
print(f"     Median: {recency['recency_days'].median():.1f} days")


print_subsection("4.2 Frequency Features")

# Time windows for frequency calculation
windows = {
    '30d': 30,
    '60d': 60,
    '90d': 90,
    '180d': 180,
    '365d': 365
}

frequency_dfs = []

for window_name, days in windows.items():
    window_start = CUTOFF_DATE - timedelta(days=days)
    window_transactions = historical_transactions[
        historical_transactions['transaction_date'] >= window_start
    ]
    
    freq = window_transactions.groupby('customer_id').agg(
        **{f'frequency_{window_name}': ('transaction_id', 'count')}
    ).reset_index()
    
    frequency_dfs.append(freq)
    print(f"  → frequency_{window_name}: Transactions in last {days} days")

# Also calculate total frequency
total_freq = historical_transactions.groupby('customer_id').agg(
    frequency_total=('transaction_id', 'count'),
    first_purchase_date=('transaction_date', 'min')
).reset_index()

print(f"  → frequency_total: Total transactions ever")


print_subsection("4.3 Monetary Features")

monetary_dfs = []

for window_name, days in windows.items():
    window_start = CUTOFF_DATE - timedelta(days=days)
    window_transactions = historical_transactions[
        historical_transactions['transaction_date'] >= window_start
    ]
    
    mon = window_transactions.groupby('customer_id').agg(
        **{
            f'monetary_{window_name}': ('total_amount', 'sum'),
            f'avg_order_value_{window_name}': ('total_amount', 'mean')
        }
    ).reset_index()
    
    monetary_dfs.append(mon)
    print(f"  → monetary_{window_name}: Total spend in last {days} days")
    print(f"  → avg_order_value_{window_name}: Average order value in last {days} days")

# Also calculate total monetary and average
total_monetary = historical_transactions.groupby('customer_id').agg(
    monetary_total=('total_amount', 'sum'),
    avg_order_value=('total_amount', 'mean'),
    max_order_value=('total_amount', 'max'),
    min_order_value=('total_amount', 'min'),
    std_order_value=('total_amount', 'std')
).reset_index()

# Fill std with 0 for customers with single transaction
total_monetary['std_order_value'] = total_monetary['std_order_value'].fillna(0)

print(f"  → monetary_total: Total spend ever")
print(f"  → avg_order_value: Average order value")
print(f"  → max_order_value: Maximum order value")
print(f"  → min_order_value: Minimum order value")
print(f"  → std_order_value: Std dev of order values")


# ============================================================
# 5. CREATE PRODUCT/CATEGORY PREFERENCE FEATURES
# ============================================================
print_section("5. CREATING PRODUCT/CATEGORY PREFERENCE FEATURES")

# Category diversity
category_features = transaction_details.groupby('customer_id').agg(
    unique_categories=('product_category', 'nunique'),
    unique_products=('product_id', 'nunique'),
    total_items_purchased=('quantity', 'sum')
).reset_index()

print(f"  → unique_categories: Number of distinct categories purchased")
print(f"  → unique_products: Number of distinct products purchased")
print(f"  → total_items_purchased: Total quantity of items purchased")

# Most purchased category per customer
category_counts = transaction_details.groupby(['customer_id', 'product_category']).size().reset_index(name='count')
idx = category_counts.groupby('customer_id')['count'].idxmax()
top_category = category_counts.loc[idx, ['customer_id', 'product_category']].copy()
top_category.columns = ['customer_id', 'top_category']

print(f"  → top_category: Most frequently purchased category")

# Category spend distribution (what % of spend in each category)
category_spend = transaction_details.groupby(['customer_id', 'product_category'])['line_item_amount'].sum().reset_index()
total_spend_by_customer = category_spend.groupby('customer_id')['line_item_amount'].sum().reset_index()
total_spend_by_customer.columns = ['customer_id', 'total_category_spend']

category_spend = category_spend.merge(total_spend_by_customer, on='customer_id')
category_spend['spend_pct'] = category_spend['line_item_amount'] / category_spend['total_category_spend']

# Pivot to get category spend percentages
category_pivot = category_spend.pivot_table(
    index='customer_id', 
    columns='product_category', 
    values='spend_pct', 
    fill_value=0
).reset_index()

# Rename columns to be more descriptive
category_pivot.columns = ['customer_id'] + [f'pct_spend_{col.lower().replace(" ", "_")}' for col in category_pivot.columns[1:]]

print(f"  → Category spend percentages: {len(category_pivot.columns) - 1} categories")


# ============================================================
# 6. CREATE CHANNEL/STORE USAGE FEATURES
# ============================================================
print_section("6. CREATING CHANNEL/STORE USAGE FEATURES")

# Merge transactions with store info
transactions_with_stores = historical_transactions.merge(
    stores[['store_id', 'store_region']], 
    on='store_id', 
    how='left'
)

store_features = transactions_with_stores.groupby('customer_id').agg(
    unique_stores=('store_id', 'nunique'),
    unique_regions=('store_region', 'nunique')
).reset_index()

print(f"  → unique_stores: Number of distinct stores visited")
print(f"  → unique_regions: Number of distinct regions shopped in")

# Most frequent store per customer
store_counts = transactions_with_stores.groupby(['customer_id', 'store_id']).size().reset_index(name='count')
idx = store_counts.groupby('customer_id')['count'].idxmax()
top_store = store_counts.loc[idx, ['customer_id', 'store_id']].copy()
top_store.columns = ['customer_id', 'primary_store_id']

print(f"  → primary_store_id: Most frequently visited store")

# Most frequent region per customer
region_counts = transactions_with_stores.groupby(['customer_id', 'store_region']).size().reset_index(name='count')
idx = region_counts.groupby('customer_id')['count'].idxmax()
top_region = region_counts.loc[idx, ['customer_id', 'store_region']].copy()
top_region.columns = ['customer_id', 'primary_region']

print(f"  → primary_region: Most frequent shopping region")


# ============================================================
# 7. CREATE CUSTOMER ATTRIBUTE FEATURES
# ============================================================
print_section("7. CREATING CUSTOMER ATTRIBUTE FEATURES")

# Select relevant customer attributes
customer_attributes = customers[[
    'customer_id', 
    'loyalty_status', 
    'total_loyalty_points', 
    'segment_id',
    'customer_since'
]].copy()

# Calculate customer tenure (days since first becoming a customer)
customer_attributes['tenure_days'] = (CUTOFF_DATE - customer_attributes['customer_since']).dt.days

# Handle negative tenure (customer_since after cutoff - shouldn't happen in clean data)
customer_attributes.loc[customer_attributes['tenure_days'] < 0, 'tenure_days'] = 0

# Drop the original date column
customer_attributes = customer_attributes.drop('customer_since', axis=1)

# Fill missing loyalty status
customer_attributes['loyalty_status'] = customer_attributes['loyalty_status'].fillna('Unknown')

# Fill missing loyalty points with 0
customer_attributes['total_loyalty_points'] = customer_attributes['total_loyalty_points'].fillna(0)

print(f"  → loyalty_status: Customer loyalty tier (Bronze/Silver/Gold/Platinum)")
print(f"  → total_loyalty_points: Accumulated loyalty points")
print(f"  → segment_id: Customer segment")
print(f"  → tenure_days: Days since customer registration")

# Print loyalty status distribution
print(f"\nLoyalty Status Distribution:")
print(customer_attributes['loyalty_status'].value_counts().to_string())


# ============================================================
# 8. CREATE BEHAVIORAL FEATURES
# ============================================================
print_section("8. CREATING BEHAVIORAL FEATURES")

# Days between purchases (purchase cadence)
def calculate_purchase_cadence(group):
    if len(group) < 2:
        return pd.Series({
            'avg_days_between_purchases': np.nan,
            'std_days_between_purchases': np.nan
        })
    
    sorted_dates = group['transaction_date'].sort_values()
    days_between = sorted_dates.diff().dt.days.dropna()
    
    return pd.Series({
        'avg_days_between_purchases': days_between.mean(),
        'std_days_between_purchases': days_between.std() if len(days_between) > 1 else 0
    })

purchase_cadence = historical_transactions.groupby('customer_id').apply(calculate_purchase_cadence).reset_index()

# Fill NaN for customers with single purchase
purchase_cadence['avg_days_between_purchases'] = purchase_cadence['avg_days_between_purchases'].fillna(0)
purchase_cadence['std_days_between_purchases'] = purchase_cadence['std_days_between_purchases'].fillna(0)

print(f"  → avg_days_between_purchases: Average days between transactions")
print(f"  → std_days_between_purchases: Variability in purchase timing")

# Time of day preferences (hour of purchase)
historical_transactions['purchase_hour'] = historical_transactions['transaction_date'].dt.hour

time_features = historical_transactions.groupby('customer_id').agg(
    avg_purchase_hour=('purchase_hour', 'mean'),
    most_common_hour=('purchase_hour', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean())
).reset_index()

# Categorize into time of day
def categorize_hour(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

time_features['preferred_time_of_day'] = time_features['most_common_hour'].apply(categorize_hour)

print(f"  → avg_purchase_hour: Average hour of day for purchases")
print(f"  → preferred_time_of_day: Preferred shopping time (morning/afternoon/evening/night)")

# Day of week preferences
historical_transactions['purchase_dow'] = historical_transactions['transaction_date'].dt.dayofweek

dow_features = historical_transactions.groupby('customer_id').agg(
    most_common_dow=('purchase_dow', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean()),
    weekend_transaction_pct=('purchase_dow', lambda x: (x >= 5).sum() / len(x))
).reset_index()

# Map day of week to name
dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
           4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
dow_features['preferred_day'] = dow_features['most_common_dow'].map(dow_map)

print(f"  → preferred_day: Most common shopping day")
print(f"  → weekend_transaction_pct: Percentage of transactions on weekends")


# ============================================================
# 9. MERGE ALL FEATURES
# ============================================================
print_section("9. MERGING ALL FEATURES")

# Start with recency (all customers who have historical transactions)
customer_features = recency.copy()
print(f"Starting with {len(customer_features):,} customers")

# Merge frequency features
customer_features = customer_features.merge(total_freq, on='customer_id', how='left')
for freq_df in frequency_dfs:
    customer_features = customer_features.merge(freq_df, on='customer_id', how='left')

# Merge monetary features
customer_features = customer_features.merge(total_monetary, on='customer_id', how='left')
for mon_df in monetary_dfs:
    customer_features = customer_features.merge(mon_df, on='customer_id', how='left')

# Merge category features
customer_features = customer_features.merge(category_features, on='customer_id', how='left')
customer_features = customer_features.merge(top_category, on='customer_id', how='left')
customer_features = customer_features.merge(category_pivot, on='customer_id', how='left')

# Merge store features
customer_features = customer_features.merge(store_features, on='customer_id', how='left')
customer_features = customer_features.merge(top_store, on='customer_id', how='left')
customer_features = customer_features.merge(top_region, on='customer_id', how='left')

# Merge customer attributes
customer_features = customer_features.merge(customer_attributes, on='customer_id', how='left')

# Merge behavioral features
customer_features = customer_features.merge(purchase_cadence, on='customer_id', how='left')
customer_features = customer_features.merge(time_features[['customer_id', 'avg_purchase_hour', 'preferred_time_of_day']], on='customer_id', how='left')
customer_features = customer_features.merge(dow_features[['customer_id', 'preferred_day', 'weekend_transaction_pct']], on='customer_id', how='left')

print(f"Merged features: {len(customer_features):,} customers, {len(customer_features.columns)} columns")


# ============================================================
# 10. CALCULATE TARGET VARIABLE
# ============================================================
print_section("10. CALCULATING TARGET VARIABLE")

# Calculate future spend for each customer in the 30-day window after cutoff
target = future_transactions.groupby('customer_id').agg(
    future_spend_30d=('total_amount', 'sum'),
    future_transactions_30d=('transaction_id', 'count')
).reset_index()

# Merge target with features
customer_features = customer_features.merge(target, on='customer_id', how='left')

# Customers with no future transactions have $0 spend
customer_features['future_spend_30d'] = customer_features['future_spend_30d'].fillna(0)
customer_features['future_transactions_30d'] = customer_features['future_transactions_30d'].fillna(0)

# Create binary target for classification (optional - for future use)
customer_features['will_purchase_30d'] = (customer_features['future_spend_30d'] > 0).astype(int)

print(f"Target variable: future_spend_30d")
print(f"  Customers who will purchase: {(customer_features['future_spend_30d'] > 0).sum():,}")
print(f"  Customers who won't purchase: {(customer_features['future_spend_30d'] == 0).sum():,}")
print(f"  Mean future spend: ${customer_features['future_spend_30d'].mean():.2f}")
print(f"  Median future spend: ${customer_features['future_spend_30d'].median():.2f}")


# ============================================================
# 11. FILL MISSING VALUES
# ============================================================
print_section("11. HANDLING MISSING VALUES")

# Fill missing numeric features with 0 (appropriate for count/sum features)
numeric_cols = customer_features.select_dtypes(include=[np.number]).columns
missing_before = customer_features[numeric_cols].isnull().sum().sum()

customer_features[numeric_cols] = customer_features[numeric_cols].fillna(0)

missing_after = customer_features[numeric_cols].isnull().sum().sum()
print(f"  Numeric missing values filled: {missing_before} → {missing_after}")

# Fill missing categorical features
categorical_cols = ['loyalty_status', 'segment_id', 'top_category', 'primary_store_id', 
                    'primary_region', 'preferred_time_of_day', 'preferred_day']

for col in categorical_cols:
    if col in customer_features.columns:
        customer_features[col] = customer_features[col].fillna('Unknown')

print(f"  Categorical missing values filled with 'Unknown'")


# ============================================================
# 12. FEATURE SUMMARY
# ============================================================
print_section("12. FEATURE SUMMARY")

feature_groups = {
    'RFM Features': [
        'recency_days',
        'frequency_30d', 'frequency_60d', 'frequency_90d', 'frequency_180d', 'frequency_365d', 'frequency_total',
        'monetary_30d', 'monetary_60d', 'monetary_90d', 'monetary_180d', 'monetary_365d', 'monetary_total',
        'avg_order_value', 'avg_order_value_30d', 'avg_order_value_60d', 'avg_order_value_90d',
        'max_order_value', 'min_order_value', 'std_order_value'
    ],
    'Product/Category Features': [
        'unique_categories', 'unique_products', 'total_items_purchased', 'top_category'
    ] + [col for col in customer_features.columns if col.startswith('pct_spend_')],
    'Store/Channel Features': [
        'unique_stores', 'unique_regions', 'primary_store_id', 'primary_region'
    ],
    'Customer Attributes': [
        'loyalty_status', 'total_loyalty_points', 'segment_id', 'tenure_days'
    ],
    'Behavioral Features': [
        'avg_days_between_purchases', 'std_days_between_purchases',
        'avg_purchase_hour', 'preferred_time_of_day', 'preferred_day', 'weekend_transaction_pct'
    ],
    'Target Variables': [
        'future_spend_30d', 'future_transactions_30d', 'will_purchase_30d'
    ]
}

total_features = 0
for group_name, features in feature_groups.items():
    existing_features = [f for f in features if f in customer_features.columns]
    total_features += len(existing_features)
    print(f"\n{group_name}: {len(existing_features)} features")
    for f in existing_features[:5]:  # Show first 5
        print(f"  - {f}")
    if len(existing_features) > 5:
        print(f"  ... and {len(existing_features) - 5} more")

print(f"\n{'='*50}")
print(f"TOTAL FEATURES: {total_features}")
print(f"TOTAL CUSTOMERS: {len(customer_features):,}")


# ============================================================
# 13. SAVE PROCESSED DATA
# ============================================================
print_section("13. SAVING PROCESSED DATA")

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Save customer features
output_path = os.path.join(PROCESSED_DATA_DIR, "customer_features.csv")
customer_features.to_csv(output_path, index=False)
print(f"  → Saved: {output_path}")
print(f"     Shape: {customer_features.shape}")

# Save a feature list for reference
feature_list_path = os.path.join(PROCESSED_DATA_DIR, "feature_list.txt")
with open(feature_list_path, 'w') as f:
    f.write("Customer Features for 30-Day Spend Prediction\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Cutoff Date: {CUTOFF_DATE}\n")
    f.write(f"Total Customers: {len(customer_features)}\n")
    f.write(f"Total Features: {len(customer_features.columns)}\n")
    f.write("\n" + "="*50 + "\n\n")
    
    for group_name, features in feature_groups.items():
        existing_features = [feat for feat in features if feat in customer_features.columns]
        f.write(f"{group_name}:\n")
        for feat in existing_features:
            f.write(f"  - {feat}\n")
        f.write("\n")

print(f"  → Saved: {feature_list_path}")

print("\n" + "="*70)
print(" FEATURE ENGINEERING COMPLETE!")
print("="*70)
print(f"\nNext steps:")
print(f"  1. Run EDA on customer_features.csv")
print(f"  2. Perform train/test split")
print(f"  3. Train models")
