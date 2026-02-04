"""
================================================================================
DATA LOADING AND CLEANING SCRIPT
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Load raw data, clean it, handle missing values, fix data types,
         remove outliers, and save cleaned data

Input:  data/raw/*.csv (7 tables with quality issues)
Output: data/cleaned/*.csv (cleaned tables)
        data/rejected/*.csv (invalid/dropped records for audit)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RAW_DATA_DIR = "/Users/apple/Customer Spend Predictor/data/raw"
CLEANED_DATA_DIR = "/Users/apple/Customer Spend Predictor/data/cleaned"
REJECTED_DATA_DIR = "/Users/apple/Customer Spend Predictor/data/rejected"

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

def summarize_missing(df, table_name):
    """Print summary of missing values in a dataframe"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df) > 0:
        print(f"\nMissing values in {table_name}:")
        print(missing_df.to_string())
    else:
        print(f"\nNo missing values in {table_name}")
    return missing_df

def parse_date_flexible(date_str):
    """
    Parse dates with multiple possible formats.
    Returns None if parsing fails.
    
    Supported formats:
    - YYYY-MM-DD
    - YYYY/MM/DD
    - DD/MM/YYYY
    - MM-DD-YYYY
    - DD-MM-YYYY HH:MM
    - YYYY-MM-DD HH:MM:SS
    """
    if pd.isna(date_str) or date_str is None or str(date_str).strip() == '':
        return None
    
    date_str = str(date_str).strip()
    
    # List of date formats to try
    formats = [
        '%Y-%m-%d %H:%M:%S',  # 2023-01-15 10:30:00
        '%Y-%m-%d %H:%M',     # 2023-01-15 10:30
        '%Y-%m-%d',           # 2023-01-15
        '%Y/%m/%d',           # 2023/01/15
        '%d/%m/%Y',           # 15/01/2023
        '%m-%d-%Y',           # 01-15-2023
        '%d-%m-%Y %H:%M',     # 15-01-2023 10:30
        '%d-%m-%Y',           # 15-01-2023
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, return None
    return None

def validate_email(email):
    """
    Validate email format.
    Returns True if valid, False otherwise.
    """
    if pd.isna(email) or email is None:
        return False
    
    email = str(email).strip()
    
    # Check for placeholder values
    if email.lower() in ['n/a', 'na', 'null', 'none', 'unknown', '']:
        return False
    
    # Simple email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    """
    Validate phone number.
    Returns True if valid (at least 10 digits), False otherwise.
    """
    if pd.isna(phone) or phone is None:
        return False
    
    phone = str(phone).strip()
    
    # Check for placeholder values
    if phone.lower() in ['n/a', 'na', 'null', 'none', 'unknown', '']:
        return False
    
    # Extract digits only
    digits = re.sub(r'\D', '', phone)
    
    # Phone should have at least 10 digits
    return len(digits) >= 10

def detect_outliers_iqr(series, multiplier=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean mask where True = outlier
    
    Parameters:
    - series: pandas Series of numeric values
    - multiplier: IQR multiplier (default 1.5, use 3 for extreme outliers only)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
print_section("1. LOADING RAW DATA")

# Load all tables
print("Loading stores.csv...")
stores_raw = pd.read_csv(f"{RAW_DATA_DIR}/stores.csv")
print(f"  → {len(stores_raw)} rows loaded")

print("Loading products.csv...")
products_raw = pd.read_csv(f"{RAW_DATA_DIR}/products.csv")
print(f"  → {len(products_raw)} rows loaded")

print("Loading customer_details.csv...")
customers_raw = pd.read_csv(f"{RAW_DATA_DIR}/customer_details.csv")
print(f"  → {len(customers_raw)} rows loaded")

print("Loading promotion_details.csv...")
promotions_raw = pd.read_csv(f"{RAW_DATA_DIR}/promotion_details.csv")
print(f"  → {len(promotions_raw)} rows loaded")

print("Loading loyalty_rules.csv...")
loyalty_rules_raw = pd.read_csv(f"{RAW_DATA_DIR}/loyalty_rules.csv")
print(f"  → {len(loyalty_rules_raw)} rows loaded")

print("Loading store_sales_header.csv...")
sales_header_raw = pd.read_csv(f"{RAW_DATA_DIR}/store_sales_header.csv")
print(f"  → {len(sales_header_raw)} rows loaded")

print("Loading store_sales_line_items.csv...")
line_items_raw = pd.read_csv(f"{RAW_DATA_DIR}/store_sales_line_items.csv")
print(f"  → {len(line_items_raw)} rows loaded")

# ============================================================
# 2. ANALYZE DATA QUALITY ISSUES
# ============================================================
print_section("2. ANALYZING DATA QUALITY ISSUES")

print_subsection("Missing Values Summary")
summarize_missing(stores_raw, "stores")
summarize_missing(products_raw, "products")
summarize_missing(customers_raw, "customers")
summarize_missing(promotions_raw, "promotions")
summarize_missing(sales_header_raw, "sales_header")
summarize_missing(line_items_raw, "line_items")


# ============================================================
# 3. CLEAN STORES TABLE
# ============================================================
print_section("3. CLEANING STORES TABLE")

stores = stores_raw.copy()
stores_rejected = pd.DataFrame()

print(f"Initial rows: {len(stores)}")

# 3.1 Clean store_name - Fill missing with city name + " Store"
print_subsection("3.1 Fixing missing store names")
missing_names = stores['store_name'].isnull()
print(f"  Missing store names: {missing_names.sum()}")
stores.loc[missing_names, 'store_name'] = stores.loc[missing_names, 'store_city'] + ' Store'
print(f"  → Filled missing names using city name")

# 3.2 Clean opening_date - Parse to standard format
print_subsection("3.2 Standardizing date formats")
stores['opening_date'] = stores['opening_date'].apply(parse_date_flexible)

# Check for any dates that couldn't be parsed
invalid_dates = stores['opening_date'].isnull()
if invalid_dates.sum() > 0:
    print(f"  Could not parse {invalid_dates.sum()} dates - filling with default")
    stores.loc[invalid_dates, 'opening_date'] = datetime(2018, 1, 1)

# Convert to standard string format
stores['opening_date'] = pd.to_datetime(stores['opening_date']).dt.strftime('%Y-%m-%d')
print(f"  → All dates converted to YYYY-MM-DD format")

# 3.3 Handle missing city (critical field)
print_subsection("3.3 Handling missing cities")
missing_cities = stores['store_city'].isnull()
if missing_cities.sum() > 0:
    print(f"  Missing cities: {missing_cities.sum()}")
    # Extract city from store_name if possible
    for idx in stores[missing_cities].index:
        name = stores.loc[idx, 'store_name']
        if name and 'Store' in name:
            stores.loc[idx, 'store_city'] = name.replace(' Store', '')
    print(f"  → Extracted city from store name where possible")

print(f"\nFinal stores rows: {len(stores)}")


# ============================================================
# 4. CLEAN PRODUCTS TABLE
# ============================================================
print_section("4. CLEANING PRODUCTS TABLE")

products = products_raw.copy()
products_rejected = pd.DataFrame()

print(f"Initial rows: {len(products)}")

# 4.1 Clean product_category - Fix casing and trailing spaces
print_subsection("4.1 Standardizing product categories")
# Strip whitespace and standardize casing
products['product_category'] = products['product_category'].str.strip().str.title()
print(f"  → Stripped whitespace and applied title case")

# Count missing categories before filling
missing_cats = products['product_category'].isnull().sum()
print(f"  Missing categories: {missing_cats}")

# Fill missing categories with 'Unknown'
products['product_category'] = products['product_category'].fillna('Unknown')

# 4.2 Handle invalid prices (negative or extreme outliers)
print_subsection("4.2 Handling invalid prices")

# Identify negative prices
negative_prices = products['unit_price'] < 0
print(f"  Negative prices found: {negative_prices.sum()}")

# Convert negative prices to positive (assuming data entry error)
products.loc[negative_prices, 'unit_price'] = products.loc[negative_prices, 'unit_price'].abs()
print(f"  → Converted negative prices to positive")

# Identify extreme outliers (prices > 1000, likely 100x error)
extreme_prices = products['unit_price'] > 1000
print(f"  Extreme prices (>1000) found: {extreme_prices.sum()}")

# Store extreme price records for review
if extreme_prices.sum() > 0:
    products_rejected = pd.concat([products_rejected, products[extreme_prices].copy()])
    products_rejected['rejection_reason'] = 'Extreme price outlier (>1000)'
    
    # Fix by dividing by 100 (assuming 100x error)
    products.loc[extreme_prices, 'unit_price'] = products.loc[extreme_prices, 'unit_price'] / 100
    print(f"  → Corrected extreme prices by dividing by 100")

# 4.3 Handle missing prices
print_subsection("4.3 Handling missing prices")
missing_prices = products['unit_price'].isnull()
print(f"  Missing prices: {missing_prices.sum()}")

if missing_prices.sum() > 0:
    # Fill with median price of same category
    for idx in products[missing_prices].index:
        category = products.loc[idx, 'product_category']
        median_price = products[products['product_category'] == category]['unit_price'].median()
        if pd.isna(median_price):
            median_price = products['unit_price'].median()
        products.loc[idx, 'unit_price'] = median_price
    print(f"  → Filled missing prices with category median")

# 4.4 Handle negative stock levels
print_subsection("4.4 Handling invalid stock levels")
negative_stock = products['current_stock_level'] < 0
print(f"  Negative stock levels: {negative_stock.sum()}")
products.loc[negative_stock, 'current_stock_level'] = 0
print(f"  → Set negative stock levels to 0")

# Fill missing stock with 0
missing_stock = products['current_stock_level'].isnull()
print(f"  Missing stock levels: {missing_stock.sum()}")
products['current_stock_level'] = products['current_stock_level'].fillna(0)

# 4.5 Handle missing product names
print_subsection("4.5 Handling missing product names")
missing_names = products['product_name'].isnull()
print(f"  Missing product names: {missing_names.sum()}")
# Fill with product_id as name
products.loc[missing_names, 'product_name'] = 'Product_' + products.loc[missing_names, 'product_id']

print(f"\nFinal products rows: {len(products)}")
print(f"Rejected products (for review): {len(products_rejected)}")


# ============================================================
# 5. CLEAN CUSTOMERS TABLE
# ============================================================
print_section("5. CLEANING CUSTOMERS TABLE")

customers = customers_raw.copy()
customers_rejected = pd.DataFrame()

print(f"Initial rows: {len(customers)}")

# 5.1 Validate and clean emails
print_subsection("5.1 Validating emails")
customers['email_valid'] = customers['email'].apply(validate_email)
invalid_emails = ~customers['email_valid']
print(f"  Invalid emails: {invalid_emails.sum()}")

# Set invalid emails to None (will be handled as missing)
customers.loc[invalid_emails, 'email'] = None
print(f"  → Set invalid emails to NULL")

# Drop the helper column
customers = customers.drop(columns=['email_valid'])

# 5.2 Validate and clean phone numbers
print_subsection("5.2 Validating phone numbers")
customers['phone_valid'] = customers['customer_phone'].apply(validate_phone)
invalid_phones = ~customers['phone_valid']
print(f"  Invalid phones: {invalid_phones.sum()}")

# Set invalid phones to None
customers.loc[invalid_phones, 'customer_phone'] = None
print(f"  → Set invalid phones to NULL")

customers = customers.drop(columns=['phone_valid'])

# 5.3 Standardize date formats
print_subsection("5.3 Standardizing date formats")

# Parse last_purchase_date
customers['last_purchase_date'] = customers['last_purchase_date'].apply(parse_date_flexible)
invalid_lpd = customers['last_purchase_date'].isnull()
print(f"  Invalid last_purchase_date: {invalid_lpd.sum()}")

# Parse customer_since
customers['customer_since'] = customers['customer_since'].apply(parse_date_flexible)
invalid_cs = customers['customer_since'].isnull()
print(f"  Invalid customer_since: {invalid_cs.sum()}")

# Convert to standard format
customers['last_purchase_date'] = pd.to_datetime(customers['last_purchase_date']).dt.strftime('%Y-%m-%d')
customers['customer_since'] = pd.to_datetime(customers['customer_since']).dt.strftime('%Y-%m-%d')
print(f"  → Converted dates to YYYY-MM-DD format")

# 5.4 Clean loyalty_status
print_subsection("5.4 Cleaning loyalty status")
missing_loyalty = customers['loyalty_status'].isnull()
print(f"  Missing loyalty status: {missing_loyalty.sum()}")

# Fill missing with 'Bronze' (default tier)
customers['loyalty_status'] = customers['loyalty_status'].fillna('Bronze')
print(f"  → Filled missing with 'Bronze'")

# Standardize casing
customers['loyalty_status'] = customers['loyalty_status'].str.strip().str.title()

# 5.5 Handle missing loyalty points
print_subsection("5.5 Handling missing loyalty points")
missing_points = customers['total_loyalty_points'].isnull()
print(f"  Missing loyalty points: {missing_points.sum()}")

# Fill with 0 for missing
customers['total_loyalty_points'] = customers['total_loyalty_points'].fillna(0)
print(f"  → Filled missing with 0")

# 5.6 Clean segment_id
print_subsection("5.6 Cleaning segment_id")
missing_segment = customers['segment_id'].isnull()
print(f"  Missing segment_id: {missing_segment.sum()}")

# Fill with 'NR' (New/Not Rated)
customers['segment_id'] = customers['segment_id'].fillna('NR')
print(f"  → Filled missing with 'NR'")

# 5.7 Handle missing first_name
print_subsection("5.7 Handling missing first names")
missing_names = customers['first_name'].isnull()
print(f"  Missing first names: {missing_names.sum()}")
customers['first_name'] = customers['first_name'].fillna('Unknown')

print(f"\nFinal customers rows: {len(customers)}")


# ============================================================
# 6. CLEAN PROMOTIONS TABLE
# ============================================================
print_section("6. CLEANING PROMOTIONS TABLE")

promotions = promotions_raw.copy()
promotions_rejected = pd.DataFrame()

print(f"Initial rows: {len(promotions)}")

# 6.1 Standardize date formats
print_subsection("6.1 Standardizing date formats")
promotions['start_date'] = promotions['start_date'].apply(parse_date_flexible)
promotions['end_date'] = promotions['end_date'].apply(parse_date_flexible)

# Fill missing end_date with start_date + 30 days
missing_end = promotions['end_date'].isnull()
print(f"  Missing end dates: {missing_end.sum()}")
for idx in promotions[missing_end].index:
    start = promotions.loc[idx, 'start_date']
    if start:
        promotions.loc[idx, 'end_date'] = start + pd.Timedelta(days=30)

promotions['start_date'] = pd.to_datetime(promotions['start_date']).dt.strftime('%Y-%m-%d')
promotions['end_date'] = pd.to_datetime(promotions['end_date']).dt.strftime('%Y-%m-%d')
print(f"  → Dates standardized to YYYY-MM-DD")

# 6.2 Handle invalid discounts
print_subsection("6.2 Handling invalid discounts")

# Identify negative discounts
negative_disc = promotions['discount_percentage'] < 0
print(f"  Negative discounts: {negative_disc.sum()}")

# Identify discounts > 1 (>100%)
excessive_disc = promotions['discount_percentage'] > 1
print(f"  Excessive discounts (>100%): {excessive_disc.sum()}")

# Store invalid discount records
invalid_disc = negative_disc | excessive_disc
if invalid_disc.sum() > 0:
    rejected_promos = promotions[invalid_disc].copy()
    rejected_promos['rejection_reason'] = 'Invalid discount percentage'
    promotions_rejected = pd.concat([promotions_rejected, rejected_promos])
    
    # Fix: set negative to 0, cap excessive at 0.5 (50%)
    promotions.loc[negative_disc, 'discount_percentage'] = 0
    promotions.loc[excessive_disc, 'discount_percentage'] = 0.5
    print(f"  → Fixed invalid discounts (negative→0, >100%→50%)")

# Handle missing discounts
missing_disc = promotions['discount_percentage'].isnull()
print(f"  Missing discounts: {missing_disc.sum()}")
promotions['discount_percentage'] = promotions['discount_percentage'].fillna(0.1)  # Default 10%

# 6.3 Clean promotion names and categories
print_subsection("6.3 Cleaning names and categories")
promotions['promotion_name'] = promotions['promotion_name'].fillna('Unnamed Promotion')
promotions['applicable_category'] = promotions['applicable_category'].fillna('ALL')

print(f"\nFinal promotions rows: {len(promotions)}")
print(f"Rejected promotions (for review): {len(promotions_rejected)}")


# ============================================================
# 7. CLEAN LOYALTY RULES TABLE
# ============================================================
print_section("7. CLEANING LOYALTY RULES TABLE")

loyalty_rules = loyalty_rules_raw.copy()

print(f"Initial rows: {len(loyalty_rules)}")

# Fill missing rule names
missing_names = loyalty_rules['rule_name'].isnull()
print(f"Missing rule names: {missing_names.sum()}")
loyalty_rules.loc[missing_names, 'rule_name'] = 'Rule_' + loyalty_rules.loc[missing_names, 'rule_id'].astype(str)

# Fill missing bonus points with 0
loyalty_rules['bonus_points'] = loyalty_rules['bonus_points'].fillna(0)

print(f"Final loyalty_rules rows: {len(loyalty_rules)}")


# ============================================================
# 8. CLEAN SALES HEADER TABLE
# ============================================================
print_section("8. CLEANING STORE_SALES_HEADER TABLE")

sales_header = sales_header_raw.copy()
sales_header_rejected = pd.DataFrame()

print(f"Initial rows: {len(sales_header)}")

# 8.1 Handle missing customer_id (CRITICAL - these transactions can't be attributed)
print_subsection("8.1 Handling missing customer IDs")
missing_cust = sales_header['customer_id'].isnull()
print(f"  Missing customer_id: {missing_cust.sum()}")

# Store transactions without customer_id in rejected
if missing_cust.sum() > 0:
    rejected_sales = sales_header[missing_cust].copy()
    rejected_sales['rejection_reason'] = 'Missing customer_id - cannot attribute transaction'
    sales_header_rejected = pd.concat([sales_header_rejected, rejected_sales])
    
    # Remove from main dataset
    sales_header = sales_header[~missing_cust]
    print(f"  → Moved {missing_cust.sum()} transactions to rejected (no customer_id)")

# 8.2 Standardize transaction dates
print_subsection("8.2 Standardizing transaction dates")
sales_header['transaction_date'] = sales_header['transaction_date'].apply(parse_date_flexible)

# Check for invalid dates
invalid_dates = sales_header['transaction_date'].isnull()
print(f"  Invalid/missing dates: {invalid_dates.sum()}")

if invalid_dates.sum() > 0:
    rejected_dates = sales_header[invalid_dates].copy()
    rejected_dates['rejection_reason'] = 'Invalid transaction date'
    sales_header_rejected = pd.concat([sales_header_rejected, rejected_dates])
    sales_header = sales_header[~invalid_dates]
    print(f"  → Moved {invalid_dates.sum()} transactions to rejected (invalid date)")

# Convert to standard format
sales_header['transaction_date'] = pd.to_datetime(sales_header['transaction_date']).dt.strftime('%Y-%m-%d %H:%M:%S')

# 8.3 Handle invalid total_amount
print_subsection("8.3 Handling invalid total amounts")

# Check for negative amounts
negative_amounts = sales_header['total_amount'] < 0
print(f"  Negative amounts: {negative_amounts.sum()}")

# These could be returns - keep but flag, or move to rejected
if negative_amounts.sum() > 0:
    rejected_neg = sales_header[negative_amounts].copy()
    rejected_neg['rejection_reason'] = 'Negative total amount'
    sales_header_rejected = pd.concat([sales_header_rejected, rejected_neg])
    sales_header = sales_header[~negative_amounts]
    print(f"  → Moved {negative_amounts.sum()} transactions to rejected (negative amount)")

# Check for missing amounts
missing_amounts = sales_header['total_amount'].isnull()
print(f"  Missing amounts: {missing_amounts.sum()}")

if missing_amounts.sum() > 0:
    rejected_missing = sales_header[missing_amounts].copy()
    rejected_missing['rejection_reason'] = 'Missing total amount'
    sales_header_rejected = pd.concat([sales_header_rejected, rejected_missing])
    sales_header = sales_header[~missing_amounts]
    print(f"  → Moved {missing_amounts.sum()} transactions to rejected (missing amount)")

# 8.4 Handle missing store_id
print_subsection("8.4 Handling missing store IDs")
missing_store = sales_header['store_id'].isnull()
print(f"  Missing store_id: {missing_store.sum()}")

# Fill with 'UNKNOWN'
sales_header['store_id'] = sales_header['store_id'].fillna('UNKNOWN')

print(f"\nFinal sales_header rows: {len(sales_header)}")
print(f"Rejected transactions: {len(sales_header_rejected)}")


# ============================================================
# 9. CLEAN LINE ITEMS TABLE
# ============================================================
print_section("9. CLEANING STORE_SALES_LINE_ITEMS TABLE")

line_items = line_items_raw.copy()
line_items_rejected = pd.DataFrame()

print(f"Initial rows: {len(line_items)}")

# 9.1 Handle missing transaction_id (can't link to transaction)
print_subsection("9.1 Handling missing transaction IDs")
missing_trans = line_items['transaction_id'].isnull()
print(f"  Missing transaction_id: {missing_trans.sum()}")

if missing_trans.sum() > 0:
    rejected_trans = line_items[missing_trans].copy()
    rejected_trans['rejection_reason'] = 'Missing transaction_id'
    line_items_rejected = pd.concat([line_items_rejected, rejected_trans])
    line_items = line_items[~missing_trans]
    print(f"  → Moved {missing_trans.sum()} line items to rejected")

# 9.2 Remove line items for rejected transactions
print_subsection("9.2 Removing orphaned line items")
valid_trans_ids = set(sales_header['transaction_id'].unique())
orphaned = ~line_items['transaction_id'].isin(valid_trans_ids)
print(f"  Orphaned line items (transaction rejected): {orphaned.sum()}")

if orphaned.sum() > 0:
    rejected_orphan = line_items[orphaned].copy()
    rejected_orphan['rejection_reason'] = 'Transaction was rejected'
    line_items_rejected = pd.concat([line_items_rejected, rejected_orphan])
    line_items = line_items[~orphaned]
    print(f"  → Moved {orphaned.sum()} orphaned line items to rejected")

# 9.3 Handle invalid quantities
print_subsection("9.3 Handling invalid quantities")

# Zero quantities - not valid
zero_qty = line_items['quantity'] == 0
print(f"  Zero quantities: {zero_qty.sum()}")

if zero_qty.sum() > 0:
    rejected_zero = line_items[zero_qty].copy()
    rejected_zero['rejection_reason'] = 'Zero quantity'
    line_items_rejected = pd.concat([line_items_rejected, rejected_zero])
    line_items = line_items[~zero_qty]
    print(f"  → Moved {zero_qty.sum()} zero-quantity items to rejected")

# Negative quantities - could be returns
negative_qty = line_items['quantity'] < 0
print(f"  Negative quantities: {negative_qty.sum()}")

if negative_qty.sum() > 0:
    rejected_neg = line_items[negative_qty].copy()
    rejected_neg['rejection_reason'] = 'Negative quantity (possible return)'
    line_items_rejected = pd.concat([line_items_rejected, rejected_neg])
    line_items = line_items[~negative_qty]
    print(f"  → Moved {negative_qty.sum()} negative-quantity items to rejected")

# Missing quantities
missing_qty = line_items['quantity'].isnull()
print(f"  Missing quantities: {missing_qty.sum()}")

if missing_qty.sum() > 0:
    rejected_missing_qty = line_items[missing_qty].copy()
    rejected_missing_qty['rejection_reason'] = 'Missing quantity'
    line_items_rejected = pd.concat([line_items_rejected, rejected_missing_qty])
    line_items = line_items[~missing_qty]
    print(f"  → Moved {missing_qty.sum()} items with missing quantity to rejected")

# 9.4 Handle missing product_id
print_subsection("9.4 Handling missing product IDs")
missing_prod = line_items['product_id'].isnull()
print(f"  Missing product_id: {missing_prod.sum()}")

if missing_prod.sum() > 0:
    rejected_prod = line_items[missing_prod].copy()
    rejected_prod['rejection_reason'] = 'Missing product_id'
    line_items_rejected = pd.concat([line_items_rejected, rejected_prod])
    line_items = line_items[~missing_prod]
    print(f"  → Moved {missing_prod.sum()} items to rejected")

# 9.5 Handle missing/invalid line_item_amount
print_subsection("9.5 Handling line item amounts")
missing_amount = line_items['line_item_amount'].isnull()
print(f"  Missing line_item_amount: {missing_amount.sum()}")

# For missing amounts, recalculate from quantity * product price
if missing_amount.sum() > 0:
    for idx in line_items[missing_amount].index:
        prod_id = line_items.loc[idx, 'product_id']
        qty = line_items.loc[idx, 'quantity']
        prod_price = products[products['product_id'] == prod_id]['unit_price']
        if len(prod_price) > 0:
            line_items.loc[idx, 'line_item_amount'] = qty * prod_price.values[0]
        else:
            line_items.loc[idx, 'line_item_amount'] = qty * 20  # default
    print(f"  → Recalculated missing amounts from quantity × price")

# 9.6 Handle NULL promotion_id (this is valid - most items have no promo)
print_subsection("9.6 Handling promotion IDs")
null_promo = line_items['promotion_id'].isnull()
print(f"  NULL promotion_id (no promo applied): {null_promo.sum()}")
print(f"  → This is valid - keeping as NULL")

print(f"\nFinal line_items rows: {len(line_items)}")
print(f"Rejected line items: {len(line_items_rejected)}")


# ============================================================
# 10. DETECT AND HANDLE OUTLIERS IN TRANSACTION AMOUNTS
# ============================================================
print_section("10. DETECTING OUTLIERS IN TRANSACTION DATA")

print_subsection("10.1 Transaction amount outliers")
outliers_mask = detect_outliers_iqr(sales_header['total_amount'], multiplier=3)
n_outliers = outliers_mask.sum()
print(f"  Extreme outliers detected: {n_outliers}")

if n_outliers > 0:
    print(f"  Outlier range: {sales_header[outliers_mask]['total_amount'].min():.2f} to {sales_header[outliers_mask]['total_amount'].max():.2f}")
    # Keep outliers but flag them (they might be valid large orders)
    sales_header['is_outlier'] = outliers_mask.astype(int)
    print(f"  → Added 'is_outlier' flag column")
else:
    sales_header['is_outlier'] = 0


# ============================================================
# 11. FINAL DATA VALIDATION
# ============================================================
print_section("11. FINAL DATA VALIDATION")

print_subsection("Record counts after cleaning")
print(f"  stores:           {len(stores):>10,} rows")
print(f"  products:         {len(products):>10,} rows")
print(f"  customers:        {len(customers):>10,} rows")
print(f"  promotions:       {len(promotions):>10,} rows")
print(f"  loyalty_rules:    {len(loyalty_rules):>10,} rows")
print(f"  sales_header:     {len(sales_header):>10,} rows")
print(f"  line_items:       {len(line_items):>10,} rows")

print_subsection("Missing value check (should be minimal)")
for name, df in [('stores', stores), ('products', products), ('customers', customers),
                  ('sales_header', sales_header), ('line_items', line_items)]:
    missing = df.isnull().sum().sum()
    print(f"  {name}: {missing} total missing values")


# ============================================================
# 12. SAVE CLEANED DATA
# ============================================================
print_section("12. SAVING CLEANED DATA")

print("Saving cleaned tables...")
stores.to_csv(f"{CLEANED_DATA_DIR}/stores.csv", index=False)
print(f"  → stores.csv ({len(stores)} rows)")

products.to_csv(f"{CLEANED_DATA_DIR}/products.csv", index=False)
print(f"  → products.csv ({len(products)} rows)")

customers.to_csv(f"{CLEANED_DATA_DIR}/customer_details.csv", index=False)
print(f"  → customer_details.csv ({len(customers)} rows)")

promotions.to_csv(f"{CLEANED_DATA_DIR}/promotion_details.csv", index=False)
print(f"  → promotion_details.csv ({len(promotions)} rows)")

loyalty_rules.to_csv(f"{CLEANED_DATA_DIR}/loyalty_rules.csv", index=False)
print(f"  → loyalty_rules.csv ({len(loyalty_rules)} rows)")

sales_header.to_csv(f"{CLEANED_DATA_DIR}/store_sales_header.csv", index=False)
print(f"  → store_sales_header.csv ({len(sales_header)} rows)")

line_items.to_csv(f"{CLEANED_DATA_DIR}/store_sales_line_items.csv", index=False)
print(f"  → store_sales_line_items.csv ({len(line_items)} rows)")


# ============================================================
# 13. SAVE REJECTED DATA
# ============================================================
print_section("13. SAVING REJECTED/INVALID DATA")

print("Saving rejected records for audit...")

if len(products_rejected) > 0:
    products_rejected.to_csv(f"{REJECTED_DATA_DIR}/products_rejected.csv", index=False)
    print(f"  → products_rejected.csv ({len(products_rejected)} rows)")

if len(promotions_rejected) > 0:
    promotions_rejected.to_csv(f"{REJECTED_DATA_DIR}/promotions_rejected.csv", index=False)
    print(f"  → promotions_rejected.csv ({len(promotions_rejected)} rows)")

if len(sales_header_rejected) > 0:
    sales_header_rejected.to_csv(f"{REJECTED_DATA_DIR}/sales_header_rejected.csv", index=False)
    print(f"  → sales_header_rejected.csv ({len(sales_header_rejected)} rows)")

if len(line_items_rejected) > 0:
    line_items_rejected.to_csv(f"{REJECTED_DATA_DIR}/line_items_rejected.csv", index=False)
    print(f"  → line_items_rejected.csv ({len(line_items_rejected)} rows)")


# ============================================================
# 14. SUMMARY REPORT
# ============================================================
print_section("DATA CLEANING SUMMARY REPORT")

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA CLEANING COMPLETED                          │
├─────────────────────────────────────────────────────────────────────┤
│  ACTIONS PERFORMED:                                                 │
│                                                                     │
│  1. DATE STANDARDIZATION                                            │
│     - All dates converted to YYYY-MM-DD format                      │
│     - Handled multiple input formats (DD/MM/YYYY, MM-DD-YYYY, etc.) │
│                                                                     │
│  2. MISSING VALUE HANDLING                                          │
│     - Store names: Filled from city name                            │
│     - Product categories: Filled with 'Unknown'                     │
│     - Product prices: Filled with category median                   │
│     - Customer loyalty: Filled with 'Bronze' (default)              │
│     - Loyalty points: Filled with 0                                 │
│     - Segment IDs: Filled with 'NR'                                 │
│                                                                     │
│  3. INVALID DATA HANDLING                                           │
│     - Invalid emails: Set to NULL                                   │
│     - Invalid phones: Set to NULL                                   │
│     - Negative prices: Converted to positive                        │
│     - Negative stock: Set to 0                                      │
│     - Invalid discounts: Capped at 50%                              │
│                                                                     │
│  4. OUTLIER HANDLING                                                │
│     - Extreme prices (>1000): Divided by 100                        │
│     - Transaction outliers: Flagged with 'is_outlier' column        │
│                                                                     │
│  5. REJECTED RECORDS                                                │
│     - Transactions without customer_id                              │
│     - Transactions with invalid dates                               │
│     - Line items with zero/negative quantities                      │
│     - Orphaned line items (transaction rejected)                    │
│     - All saved to data/rejected/ for audit                         │
└─────────────────────────────────────────────────────────────────────┘
""")

print(f"\nFiles saved to:")
print(f"  Cleaned data: {CLEANED_DATA_DIR}")
print(f"  Rejected data: {REJECTED_DATA_DIR}")

print("\n" + "=" * 70)
print(" DATA CLEANING COMPLETE!")
print("=" * 70)
