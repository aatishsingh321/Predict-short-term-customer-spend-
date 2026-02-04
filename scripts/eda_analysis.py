"""
================================================================================
EXPLORATORY DATA ANALYSIS (EDA) SCRIPT
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Analyze cleaned data, generate statistics, create visualizations,
         and document initial hypotheses for modeling

Input:  data/cleaned/*.csv
Output: outputs/eda_plots/*.png (visualizations)
        Console output with statistics and insights
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
CLEANED_DATA_DIR = "/Users/apple/Customer Spend Predictor/data/cleaned"
OUTPUT_DIR = "/Users/apple/Customer Spend Predictor/outputs/eda_plots"

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'─' * 40}")
    print(f" {title}")
    print(f"{'─' * 40}")

def save_plot(fig, filename):
    """Save plot to output directory"""
    filepath = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  → Saved: {filename}")


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================
print_section("1. LOADING CLEANED DATA")

print("Loading all cleaned tables...")
stores = pd.read_csv(f"{CLEANED_DATA_DIR}/stores.csv")
products = pd.read_csv(f"{CLEANED_DATA_DIR}/products.csv")
customers = pd.read_csv(f"{CLEANED_DATA_DIR}/customer_details.csv")
promotions = pd.read_csv(f"{CLEANED_DATA_DIR}/promotion_details.csv")
loyalty_rules = pd.read_csv(f"{CLEANED_DATA_DIR}/loyalty_rules.csv")
sales_header = pd.read_csv(f"{CLEANED_DATA_DIR}/store_sales_header.csv")
line_items = pd.read_csv(f"{CLEANED_DATA_DIR}/store_sales_line_items.csv")

# Convert date columns
sales_header['transaction_date'] = pd.to_datetime(sales_header['transaction_date'])
customers['last_purchase_date'] = pd.to_datetime(customers['last_purchase_date'], errors='coerce')
customers['customer_since'] = pd.to_datetime(customers['customer_since'], errors='coerce')

print(f"""
Loaded tables:
  - stores:       {len(stores):,} rows
  - products:     {len(products):,} rows
  - customers:    {len(customers):,} rows
  - promotions:   {len(promotions):,} rows
  - loyalty_rules: {len(loyalty_rules):,} rows
  - sales_header: {len(sales_header):,} rows
  - line_items:   {len(line_items):,} rows
""")


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================
print_section("2. DESCRIPTIVE STATISTICS")

# ------------------------------
# 2.1 Transaction Statistics
# ------------------------------
print_subsection("2.1 Transaction Amount Statistics")

trans_stats = sales_header['total_amount'].describe()
print(f"""
Transaction Amount (total_amount):
  Count:    {trans_stats['count']:,.0f}
  Mean:     ${trans_stats['mean']:,.2f}
  Std Dev:  ${trans_stats['std']:,.2f}
  Min:      ${trans_stats['min']:,.2f}
  25%:      ${trans_stats['25%']:,.2f}
  Median:   ${trans_stats['50%']:,.2f}
  75%:      ${trans_stats['75%']:,.2f}
  Max:      ${trans_stats['max']:,.2f}
""")

# ------------------------------
# 2.2 Customer Statistics
# ------------------------------
print_subsection("2.2 Customer Statistics")

# Transactions per customer
trans_per_customer = sales_header.groupby('customer_id').size()
print(f"""
Transactions per Customer:
  Mean:     {trans_per_customer.mean():.2f}
  Median:   {trans_per_customer.median():.0f}
  Min:      {trans_per_customer.min()}
  Max:      {trans_per_customer.max()}
  Std Dev:  {trans_per_customer.std():.2f}
""")

# Total spend per customer
spend_per_customer = sales_header.groupby('customer_id')['total_amount'].sum()
print(f"""
Total Spend per Customer:
  Mean:     ${spend_per_customer.mean():,.2f}
  Median:   ${spend_per_customer.median():,.2f}
  Min:      ${spend_per_customer.min():,.2f}
  Max:      ${spend_per_customer.max():,.2f}
  Std Dev:  ${spend_per_customer.std():,.2f}
""")

# Loyalty status distribution
print("\nLoyalty Status Distribution:")
loyalty_dist = customers['loyalty_status'].value_counts()
for status, count in loyalty_dist.items():
    pct = count / len(customers) * 100
    print(f"  {status}: {count:,} ({pct:.1f}%)")

# ------------------------------
# 2.3 Product Statistics
# ------------------------------
print_subsection("2.3 Product Statistics")

price_stats = products['unit_price'].describe()
print(f"""
Product Unit Price:
  Mean:     ${price_stats['mean']:,.2f}
  Median:   ${price_stats['50%']:,.2f}
  Min:      ${price_stats['min']:,.2f}
  Max:      ${price_stats['max']:,.2f}
""")

print("\nProducts by Category:")
cat_dist = products['product_category'].value_counts()
for cat, count in cat_dist.items():
    print(f"  {cat}: {count}")

# ------------------------------
# 2.4 Time-based Statistics
# ------------------------------
print_subsection("2.4 Time-based Statistics")

print(f"""
Transaction Date Range:
  First Transaction: {sales_header['transaction_date'].min()}
  Last Transaction:  {sales_header['transaction_date'].max()}
  Total Days:        {(sales_header['transaction_date'].max() - sales_header['transaction_date'].min()).days}
""")

# Transactions by month
sales_header['month'] = sales_header['transaction_date'].dt.to_period('M')
monthly_trans = sales_header.groupby('month').agg({
    'transaction_id': 'count',
    'total_amount': 'sum'
}).rename(columns={'transaction_id': 'num_transactions', 'total_amount': 'total_revenue'})

print("\nMonthly Transaction Summary (sample):")
print(monthly_trans.head(6).to_string())

# ------------------------------
# 2.5 Store Statistics
# ------------------------------
print_subsection("2.5 Store Statistics")

store_stats = sales_header.groupby('store_id').agg({
    'transaction_id': 'count',
    'total_amount': ['sum', 'mean']
}).round(2)
store_stats.columns = ['num_transactions', 'total_revenue', 'avg_transaction']
store_stats = store_stats.sort_values('total_revenue', ascending=False)

print("\nTop 5 Stores by Revenue:")
print(store_stats.head().to_string())


# ============================================================
# 3. DISTRIBUTION PLOTS
# ============================================================
print_section("3. GENERATING DISTRIBUTION PLOTS")

# ------------------------------
# 3.1 Transaction Amount Distribution
# ------------------------------
print_subsection("3.1 Transaction Amount Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(sales_header['total_amount'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(sales_header['total_amount'].mean(), color='red', linestyle='--', label=f'Mean: ${sales_header["total_amount"].mean():.2f}')
axes[0].axvline(sales_header['total_amount'].median(), color='green', linestyle='--', label=f'Median: ${sales_header["total_amount"].median():.2f}')
axes[0].set_xlabel('Transaction Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Transaction Amounts')
axes[0].legend()

# Box plot
axes[1].boxplot(sales_header['total_amount'], vert=True)
axes[1].set_ylabel('Transaction Amount ($)')
axes[1].set_title('Transaction Amount Box Plot')

plt.tight_layout()
save_plot(fig, '01_transaction_amount_distribution.png')

# ------------------------------
# 3.2 Customer Spend Distribution
# ------------------------------
print_subsection("3.2 Customer Total Spend Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of total spend per customer
axes[0].hist(spend_per_customer, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0].axvline(spend_per_customer.mean(), color='red', linestyle='--', label=f'Mean: ${spend_per_customer.mean():.2f}')
axes[0].axvline(spend_per_customer.median(), color='orange', linestyle='--', label=f'Median: ${spend_per_customer.median():.2f}')
axes[0].set_xlabel('Total Customer Spend ($)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Distribution of Total Customer Spend')
axes[0].legend()

# Log-transformed for better visualization
axes[1].hist(np.log1p(spend_per_customer), bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1].set_xlabel('Log(Total Customer Spend + 1)')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Log-Transformed Customer Spend Distribution')

plt.tight_layout()
save_plot(fig, '02_customer_spend_distribution.png')

# ------------------------------
# 3.3 Transactions per Customer Distribution
# ------------------------------
print_subsection("3.3 Transactions per Customer Distribution")

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(trans_per_customer, bins=range(1, trans_per_customer.max()+2), edgecolor='black', alpha=0.7, color='coral')
ax.axvline(trans_per_customer.mean(), color='red', linestyle='--', label=f'Mean: {trans_per_customer.mean():.1f}')
ax.set_xlabel('Number of Transactions')
ax.set_ylabel('Number of Customers')
ax.set_title('Distribution of Transactions per Customer')
ax.legend()

plt.tight_layout()
save_plot(fig, '03_transactions_per_customer.png')

# ------------------------------
# 3.4 Loyalty Status Distribution
# ------------------------------
print_subsection("3.4 Loyalty Status Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
axes[0].pie(loyalty_dist.values, labels=loyalty_dist.index, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Customer Distribution by Loyalty Status')

# Bar chart with spend
loyalty_spend = sales_header.merge(customers[['customer_id', 'loyalty_status']], on='customer_id')
loyalty_avg_spend = loyalty_spend.groupby('loyalty_status')['total_amount'].mean().sort_values(ascending=False)
axes[1].bar(loyalty_avg_spend.index, loyalty_avg_spend.values, color=['gold', 'silver', '#cd7f32', 'lightgray'])
axes[1].set_xlabel('Loyalty Status')
axes[1].set_ylabel('Average Transaction Amount ($)')
axes[1].set_title('Average Transaction Amount by Loyalty Status')

plt.tight_layout()
save_plot(fig, '04_loyalty_status_analysis.png')

# ------------------------------
# 3.5 Product Category Analysis
# ------------------------------
print_subsection("3.5 Product Category Analysis")

# Merge line items with products to get category
line_items_with_cat = line_items.merge(products[['product_id', 'product_category', 'unit_price']], on='product_id', how='left')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Revenue by category
cat_revenue = line_items_with_cat.groupby('product_category')['line_item_amount'].sum().sort_values(ascending=True)
axes[0].barh(cat_revenue.index, cat_revenue.values, color='steelblue')
axes[0].set_xlabel('Total Revenue ($)')
axes[0].set_title('Revenue by Product Category')

# Transaction count by category
cat_count = line_items_with_cat['product_category'].value_counts().sort_values(ascending=True)
axes[1].barh(cat_count.index, cat_count.values, color='seagreen')
axes[1].set_xlabel('Number of Items Sold')
axes[1].set_title('Items Sold by Product Category')

plt.tight_layout()
save_plot(fig, '05_product_category_analysis.png')

# ------------------------------
# 3.6 Time Series Analysis
# ------------------------------
print_subsection("3.6 Time Series Analysis")

# Daily transactions
daily_stats = sales_header.groupby(sales_header['transaction_date'].dt.date).agg({
    'transaction_id': 'count',
    'total_amount': 'sum'
}).rename(columns={'transaction_id': 'num_transactions', 'total_amount': 'daily_revenue'})
daily_stats.index = pd.to_datetime(daily_stats.index)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Daily transaction count
axes[0].plot(daily_stats.index, daily_stats['num_transactions'], color='blue', alpha=0.7)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Number of Transactions')
axes[0].set_title('Daily Transaction Count Over Time')
axes[0].tick_params(axis='x', rotation=45)

# Daily revenue
axes[1].plot(daily_stats.index, daily_stats['daily_revenue'], color='green', alpha=0.7)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Daily Revenue ($)')
axes[1].set_title('Daily Revenue Over Time')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_plot(fig, '06_time_series_analysis.png')

# ------------------------------
# 3.7 Day of Week Analysis
# ------------------------------
print_subsection("3.7 Day of Week Analysis")

sales_header['day_of_week'] = sales_header['transaction_date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Transactions by day of week
day_trans = sales_header['day_of_week'].value_counts().reindex(day_order)
axes[0].bar(day_trans.index, day_trans.values, color='coral')
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Number of Transactions')
axes[0].set_title('Transactions by Day of Week')
axes[0].tick_params(axis='x', rotation=45)

# Average transaction by day of week
day_avg = sales_header.groupby('day_of_week')['total_amount'].mean().reindex(day_order)
axes[1].bar(day_avg.index, day_avg.values, color='teal')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Average Transaction Amount ($)')
axes[1].set_title('Average Transaction Amount by Day of Week')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_plot(fig, '07_day_of_week_analysis.png')

# ------------------------------
# 3.8 Hour of Day Analysis
# ------------------------------
print_subsection("3.8 Hour of Day Analysis")

sales_header['hour'] = sales_header['transaction_date'].dt.hour

fig, ax = plt.subplots(figsize=(12, 5))

hour_trans = sales_header['hour'].value_counts().sort_index()
ax.bar(hour_trans.index, hour_trans.values, color='purple', alpha=0.7)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Transactions')
ax.set_title('Transaction Distribution by Hour of Day')
ax.set_xticks(range(0, 24))

plt.tight_layout()
save_plot(fig, '08_hour_of_day_analysis.png')

# ------------------------------
# 3.9 Store Performance Analysis
# ------------------------------
print_subsection("3.9 Store Performance Analysis")

# Merge with store info
sales_with_store = sales_header.merge(stores[['store_id', 'store_name', 'store_region']], on='store_id', how='left')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Revenue by region
region_revenue = sales_with_store.groupby('store_region')['total_amount'].sum().sort_values(ascending=True)
axes[0].barh(region_revenue.index, region_revenue.values, color='darkorange')
axes[0].set_xlabel('Total Revenue ($)')
axes[0].set_title('Revenue by Store Region')

# Average transaction by region
region_avg = sales_with_store.groupby('store_region')['total_amount'].mean().sort_values(ascending=True)
axes[1].barh(region_avg.index, region_avg.values, color='darkgreen')
axes[1].set_xlabel('Average Transaction Amount ($)')
axes[1].set_title('Average Transaction by Store Region')

plt.tight_layout()
save_plot(fig, '09_store_region_analysis.png')

# ------------------------------
# 3.10 Customer Tenure Analysis
# ------------------------------
print_subsection("3.10 Customer Tenure Analysis")

# Calculate customer tenure in days
customers['tenure_days'] = (pd.Timestamp('2025-12-01') - customers['customer_since']).dt.days

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tenure distribution
axes[0].hist(customers['tenure_days'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_xlabel('Customer Tenure (Days)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Distribution of Customer Tenure')

# Tenure vs Total Spend
customer_spend = sales_header.groupby('customer_id')['total_amount'].sum().reset_index()
customer_spend.columns = ['customer_id', 'total_spend']
customers_with_spend = customers.merge(customer_spend, on='customer_id', how='left')
customers_with_spend['total_spend'] = customers_with_spend['total_spend'].fillna(0)

axes[1].scatter(customers_with_spend['tenure_days'], customers_with_spend['total_spend'], alpha=0.3, s=10)
axes[1].set_xlabel('Customer Tenure (Days)')
axes[1].set_ylabel('Total Spend ($)')
axes[1].set_title('Customer Tenure vs Total Spend')

plt.tight_layout()
save_plot(fig, '10_customer_tenure_analysis.png')

# ------------------------------
# 3.11 Loyalty Points vs Spend
# ------------------------------
print_subsection("3.11 Loyalty Points vs Spend Analysis")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(customers_with_spend['total_loyalty_points'], customers_with_spend['total_spend'], alpha=0.3, s=10, c='purple')
ax.set_xlabel('Total Loyalty Points')
ax.set_ylabel('Total Spend ($)')
ax.set_title('Loyalty Points vs Total Customer Spend')

plt.tight_layout()
save_plot(fig, '11_loyalty_points_vs_spend.png')

# ------------------------------
# 3.12 Correlation Heatmap
# ------------------------------
print_subsection("3.12 Correlation Analysis")

# Create customer-level features for correlation
customer_features = sales_header.groupby('customer_id').agg({
    'total_amount': ['sum', 'mean', 'count'],
    'store_id': 'nunique'
}).reset_index()
customer_features.columns = ['customer_id', 'total_spend', 'avg_transaction', 'num_transactions', 'num_stores']

# Merge with customer attributes
customer_corr_data = customer_features.merge(
    customers[['customer_id', 'total_loyalty_points', 'tenure_days']], 
    on='customer_id', 
    how='left'
)

# Calculate correlation matrix
corr_matrix = customer_corr_data[['total_spend', 'avg_transaction', 'num_transactions', 
                                   'num_stores', 'total_loyalty_points', 'tenure_days']].corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix of Customer Features')

plt.tight_layout()
save_plot(fig, '12_correlation_heatmap.png')

print("\nCorrelation Matrix:")
print(corr_matrix.round(3).to_string())


# ============================================================
# 4. CUSTOMER SEGMENTATION ANALYSIS (RFM Preview)
# ============================================================
print_section("4. CUSTOMER SEGMENTATION ANALYSIS (RFM Preview)")

# Calculate RFM metrics
reference_date = sales_header['transaction_date'].max() + timedelta(days=1)

rfm = sales_header.groupby('customer_id').agg({
    'transaction_date': lambda x: (reference_date - x.max()).days,  # Recency
    'transaction_id': 'count',  # Frequency
    'total_amount': 'sum'  # Monetary
}).reset_index()
rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

print(f"""
RFM Metrics Summary:

RECENCY (days since last purchase):
  Mean:   {rfm['recency'].mean():.1f} days
  Median: {rfm['recency'].median():.0f} days
  Min:    {rfm['recency'].min()} days
  Max:    {rfm['recency'].max()} days

FREQUENCY (number of transactions):
  Mean:   {rfm['frequency'].mean():.1f}
  Median: {rfm['frequency'].median():.0f}
  Min:    {rfm['frequency'].min()}
  Max:    {rfm['frequency'].max()}

MONETARY (total spend):
  Mean:   ${rfm['monetary'].mean():,.2f}
  Median: ${rfm['monetary'].median():,.2f}
  Min:    ${rfm['monetary'].min():,.2f}
  Max:    ${rfm['monetary'].max():,.2f}
""")

# RFM Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(rfm['recency'], bins=50, edgecolor='black', alpha=0.7, color='red')
axes[0].set_xlabel('Recency (Days)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Recency Distribution')

axes[1].hist(rfm['frequency'], bins=50, edgecolor='black', alpha=0.7, color='blue')
axes[1].set_xlabel('Frequency (# Transactions)')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Frequency Distribution')

axes[2].hist(rfm['monetary'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[2].set_xlabel('Monetary (Total Spend $)')
axes[2].set_ylabel('Number of Customers')
axes[2].set_title('Monetary Distribution')

plt.tight_layout()
save_plot(fig, '13_rfm_distributions.png')


# ============================================================
# 5. INITIAL HYPOTHESES
# ============================================================
print_section("5. INITIAL HYPOTHESES FOR MODELING")

print("""
Based on the EDA, the following hypotheses will guide feature engineering and modeling:

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 1: Past Spend Predicts Future Spend                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Strong correlation between historical spending and total spend │
│ Hypothesis:  Customers who spent more in the past will spend more in the    │
│              next 30 days                                                   │
│ Features:    total_spend_30d, total_spend_60d, total_spend_90d              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 2: Purchase Frequency Indicates Engagement                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: High variance in transactions per customer (1 to 35+)          │
│ Hypothesis:  Customers with higher purchase frequency will continue         │
│              purchasing in the next 30 days                                 │
│ Features:    num_transactions_30d, num_transactions_60d, avg_days_between   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 3: Recency Affects Future Purchases                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Wide range of recency values (0 to 700+ days)                  │
│ Hypothesis:  Recently active customers are more likely to purchase again    │
│ Features:    days_since_last_purchase, is_active_30d, is_active_60d         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 4: Loyalty Status Correlates with Spend                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Platinum/Gold customers have higher avg transaction amounts    │
│ Hypothesis:  Higher loyalty tiers indicate higher future spending potential │
│ Features:    loyalty_status_encoded, total_loyalty_points                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 5: Customer Tenure Affects Behavior                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Positive correlation between tenure and total spend            │
│ Hypothesis:  Longer-tenured customers have established buying patterns      │
│ Features:    customer_tenure_days, tenure_bucket                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 6: Category Preferences Indicate Future Purchases                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Clear category preferences vary by customer                    │
│ Hypothesis:  Customers tend to buy from their preferred categories          │
│ Features:    top_category, num_categories, category_diversity_score         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 7: Store Behavior Reflects Shopping Patterns                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Customers visit different numbers of stores                    │
│ Hypothesis:  Multi-store shoppers may have different spend patterns         │
│ Features:    num_stores_visited, preferred_store, preferred_region          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 8: Temporal Patterns Exist                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Observation: Variations in transactions by day of week and hour             │
│ Hypothesis:  Shopping time preferences may indicate customer type           │
│ Features:    preferred_day_of_week, preferred_hour, weekend_shopper_flag    │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================================
# 6. KEY INSIGHTS SUMMARY
# ============================================================
print_section("6. KEY INSIGHTS SUMMARY")

# Calculate some key metrics for summary
active_customers = len(rfm[rfm['recency'] <= 30])
churned_customers = len(rfm[rfm['recency'] > 180])
high_value_customers = len(spend_per_customer[spend_per_customer > spend_per_customer.quantile(0.9)])

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KEY INSIGHTS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA VOLUME                                                                │
│  • Total Transactions: {len(sales_header):,}                                         │
│  • Total Customers: {len(customers):,}                                            │
│  • Total Products: {len(products)}                                               │
│  • Date Range: {(sales_header['transaction_date'].max() - sales_header['transaction_date'].min()).days} days                                           │
│                                                                             │
│  CUSTOMER BEHAVIOR                                                          │
│  • Avg Transactions/Customer: {trans_per_customer.mean():.1f}                              │
│  • Avg Spend/Customer: ${spend_per_customer.mean():,.2f}                            │
│  • Active Customers (30 days): {active_customers:,} ({active_customers/len(rfm)*100:.1f}%)                      │
│  • Churned Customers (>180 days): {churned_customers:,} ({churned_customers/len(rfm)*100:.1f}%)                  │
│  • High-Value Customers (top 10%): {high_value_customers:,}                            │
│                                                                             │
│  TRANSACTION PATTERNS                                                       │
│  • Avg Transaction Amount: ${sales_header['total_amount'].mean():.2f}                       │
│  • Median Transaction Amount: ${sales_header['total_amount'].median():.2f}                     │
│  • Transaction Skewness: {sales_header['total_amount'].skew():.2f} (right-skewed)                │
│                                                                             │
│  LOYALTY IMPACT                                                             │
│  • Bronze: {loyalty_dist.get('Bronze', 0)/len(customers)*100:.1f}% of customers                                    │
│  • Silver: {loyalty_dist.get('Silver', 0)/len(customers)*100:.1f}% of customers                                    │
│  • Gold: {loyalty_dist.get('Gold', 0)/len(customers)*100:.1f}% of customers                                      │
│  • Platinum: {loyalty_dist.get('Platinum', 0)/len(customers)*100:.1f}% of customers                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMMENDED FEATURES FOR MODELING:

1. RFM Features (Recency, Frequency, Monetary)
2. Historical spend in multiple windows (30d, 60d, 90d)
3. Loyalty status and points
4. Customer tenure
5. Category preferences
6. Store/region behavior
7. Temporal patterns (day of week, hour)
8. Average order value and basket size
""")


# ============================================================
# 7. SAVE EDA SUMMARY STATISTICS
# ============================================================
print_section("7. SAVING EDA RESULTS")

# Save RFM data for later use
rfm.to_csv(f"{CLEANED_DATA_DIR}/customer_rfm.csv", index=False)
print(f"  → Saved: customer_rfm.csv ({len(rfm)} rows)")

# Save customer features for later use
customer_features.to_csv(f"{CLEANED_DATA_DIR}/customer_features_basic.csv", index=False)
print(f"  → Saved: customer_features_basic.csv ({len(customer_features)} rows)")

print(f"\n  → All plots saved to: {OUTPUT_DIR}/")

print("\n" + "=" * 80)
print(" EDA COMPLETE!")
print("=" * 80)
