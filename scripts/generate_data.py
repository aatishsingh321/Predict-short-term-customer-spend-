"""
Generate Raw Retail Dataset with Realistic Errors and Missing Values
Based on the provided schema for Customer Spend Prediction project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Output directory
OUTPUT_DIR = "/Users/apple/Customer Spend Predictor/data/raw"

# ============================================================
# 1. STORES TABLE
# ============================================================
def generate_stores(n_stores=15):
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin',
              'Seattle', 'Boston', 'Denver', 'Atlanta', 'Miami']
    regions = ['Northeast', 'West', 'Midwest', 'South', 'Southwest']
    
    stores = []
    for i in range(1, n_stores + 1):
        city = cities[i-1]
        # Assign region based on city
        if city in ['New York', 'Philadelphia', 'Boston']:
            region = 'Northeast'
        elif city in ['Los Angeles', 'San Diego', 'Seattle', 'Phoenix']:
            region = 'West'
        elif city in ['Chicago']:
            region = 'Midwest'
        elif city in ['Houston', 'San Antonio', 'Dallas', 'Austin']:
            region = 'Southwest'
        else:
            region = 'South'
        
        # Add some missing values and errors
        store_name = f"{city} Store" if random.random() > 0.1 else None  # 10% missing
        opening_date = (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 1000))).strftime('%Y-%m-%d')
        
        # Some dates with wrong format (error)
        if random.random() < 0.05:
            opening_date = opening_date.replace('-', '/')
        
        stores.append({
            'store_id': f'S{str(i).zfill(3)}',
            'store_name': store_name,
            'store_city': city if random.random() > 0.03 else None,  # 3% missing
            'store_region': region,
            'opening_date': opening_date
        })
    
    return pd.DataFrame(stores)

# ============================================================
# 2. PRODUCTS TABLE
# ============================================================
def generate_products(n_products=200):
    categories = ['Electronics', 'Apparel', 'Home & Garden', 'Sports', 'Beauty', 
                  'Toys', 'Books', 'Grocery', 'Automotive', 'Jewelry']
    
    product_names = {
        'Electronics': ['Wireless Headphones', 'Smart Watch', 'Bluetooth Speaker', 'Phone Case', 'USB Cable', 'Power Bank', 'Tablet Stand', 'Webcam', 'Mouse', 'Keyboard'],
        'Apparel': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hat', 'Scarf', 'Gloves', 'Socks', 'Belt', 'Dress'],
        'Home & Garden': ['Plant Pot', 'Candle', 'Pillow', 'Blanket', 'Vase', 'Picture Frame', 'Lamp', 'Rug', 'Curtains', 'Clock'],
        'Sports': ['Yoga Mat', 'Dumbbells', 'Water Bottle', 'Resistance Bands', 'Jump Rope', 'Tennis Balls', 'Basketball', 'Helmet', 'Gloves', 'Knee Pads'],
        'Beauty': ['Lipstick', 'Mascara', 'Face Cream', 'Shampoo', 'Perfume', 'Nail Polish', 'Sunscreen', 'Hair Dryer', 'Brush Set', 'Face Mask'],
        'Toys': ['Action Figure', 'Board Game', 'Puzzle', 'Lego Set', 'Doll', 'RC Car', 'Stuffed Animal', 'Building Blocks', 'Card Game', 'Toy Train'],
        'Books': ['Novel', 'Cookbook', 'Biography', 'Self-Help', 'Science Fiction', 'Mystery', 'History', 'Art Book', 'Travel Guide', 'Dictionary'],
        'Grocery': ['Coffee', 'Tea', 'Chocolate', 'Pasta', 'Olive Oil', 'Honey', 'Cereal', 'Snacks', 'Spices', 'Jam'],
        'Automotive': ['Car Freshener', 'Phone Mount', 'Seat Cover', 'Floor Mat', 'Sunshade', 'Tire Gauge', 'Jump Starter', 'Tool Kit', 'Wax', 'Cleaner'],
        'Jewelry': ['Necklace', 'Bracelet', 'Earrings', 'Ring', 'Watch', 'Anklet', 'Brooch', 'Cufflinks', 'Pendant', 'Chain']
    }
    
    products = []
    product_id = 1
    
    for category in categories:
        for name in product_names[category]:
            for variant in range(1, random.randint(2, 4)):
                # Price based on category
                base_price = {
                    'Electronics': random.uniform(15, 200),
                    'Apparel': random.uniform(10, 150),
                    'Home & Garden': random.uniform(5, 100),
                    'Sports': random.uniform(8, 80),
                    'Beauty': random.uniform(5, 60),
                    'Toys': random.uniform(10, 70),
                    'Books': random.uniform(8, 40),
                    'Grocery': random.uniform(3, 30),
                    'Automotive': random.uniform(10, 100),
                    'Jewelry': random.uniform(20, 300)
                }[category]
                
                price = round(base_price, 2)
                
                # Introduce errors
                # Some negative prices (error)
                if random.random() < 0.02:
                    price = -abs(price)
                # Some extremely high prices (outlier)
                if random.random() < 0.01:
                    price = price * 100
                # Some missing prices
                if random.random() < 0.03:
                    price = None
                
                # Missing product names
                prod_name = f"{name} V{variant}" if random.random() > 0.05 else None
                
                # Typos in category
                cat = category
                if random.random() < 0.03:
                    cat = category.lower()  # inconsistent casing
                if random.random() < 0.02:
                    cat = category + ' '  # trailing space
                
                products.append({
                    'product_id': f'P{str(product_id).zfill(4)}',
                    'product_name': prod_name,
                    'product_category': cat if random.random() > 0.02 else None,
                    'unit_price': price,
                    'current_stock_level': random.randint(-5, 500) if random.random() > 0.05 else None  # some negative stock (error)
                })
                product_id += 1
                
                if product_id > n_products:
                    break
            if product_id > n_products:
                break
        if product_id > n_products:
            break
    
    return pd.DataFrame(products)

# ============================================================
# 3. CUSTOMER_DETAILS TABLE
# ============================================================
def generate_customers(n_customers=5000):
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 
                   'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
                   'Thomas', 'Sarah', 'Charles', 'Karen', 'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia',
                   'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn', 'Liam', 'Noah', 'Oliver', 'Elijah']
    
    loyalty_statuses = ['Bronze', 'Silver', 'Gold', 'Platinum']
    segments = ['HS', 'AR', 'NR', 'LP', 'HC']  # High Spender, At Risk, New, Lapsed, High Contact
    
    customers = []
    
    for i in range(1, n_customers + 1):
        # Generate customer since date (1-4 years ago)
        customer_since = datetime(2022, 1, 1) - timedelta(days=random.randint(30, 1500))
        last_purchase = customer_since + timedelta(days=random.randint(0, (datetime(2025, 12, 1) - customer_since).days))
        
        # Loyalty status affects points
        loyalty = random.choices(loyalty_statuses, weights=[50, 30, 15, 5])[0]
        points_range = {
            'Bronze': (0, 500),
            'Silver': (500, 2000),
            'Gold': (2000, 5000),
            'Platinum': (5000, 20000)
        }[loyalty]
        
        # Generate email with some errors
        fname = random.choice(first_names)
        email = f"{fname.lower()}{i}@email.com"
        if random.random() < 0.03:
            email = email.replace('@', '')  # invalid email
        if random.random() < 0.02:
            email = 'N/A'  # placeholder
        if random.random() < 0.05:
            email = None
        
        # Phone with errors
        phone = f"{random.randint(100,999)}{random.randint(1000000,9999999)}"
        if random.random() < 0.03:
            phone = phone[:5]  # incomplete phone
        if random.random() < 0.02:
            phone = 'unknown'
        if random.random() < 0.08:
            phone = None
        
        # Date formatting inconsistencies
        last_purchase_str = last_purchase.strftime('%Y-%m-%d')
        customer_since_str = customer_since.strftime('%Y-%m-%d')
        if random.random() < 0.05:
            last_purchase_str = last_purchase.strftime('%d/%m/%Y')  # different format
        if random.random() < 0.05:
            customer_since_str = customer_since.strftime('%m-%d-%Y')  # different format
        
        customers.append({
            'customer_id': f'C{str(i).zfill(5)}',
            'first_name': fname if random.random() > 0.03 else None,
            'email': email,
            'loyalty_status': loyalty if random.random() > 0.02 else None,
            'total_loyalty_points': random.randint(*points_range) if random.random() > 0.04 else None,
            'last_purchase_date': last_purchase_str if random.random() > 0.03 else None,
            'segment_id': random.choice(segments) if random.random() > 0.05 else None,
            'customer_phone': phone,
            'customer_since': customer_since_str if random.random() > 0.02 else None
        })
    
    return pd.DataFrame(customers)

# ============================================================
# 4. PROMOTION_DETAILS TABLE
# ============================================================
def generate_promotions(n_promos=30):
    categories = ['Electronics', 'Apparel', 'Home & Garden', 'Sports', 'Beauty', 
                  'Toys', 'Books', 'Grocery', 'Automotive', 'Jewelry', 'ALL']
    
    promo_names = ['Summer Sale', 'Winter Clearance', 'Black Friday', 'Cyber Monday', 
                   'Holiday Special', 'Back to School', 'Flash Sale', 'Weekend Deal',
                   'Member Exclusive', 'New Year Sale', 'Spring Savings', 'Fall Festival']
    
    promotions = []
    
    for i in range(1, n_promos + 1):
        start = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 700))
        duration = random.randint(3, 30)
        end = start + timedelta(days=duration)
        
        discount = round(random.uniform(0.05, 0.50), 2)
        # Some invalid discounts
        if random.random() < 0.03:
            discount = random.uniform(1.1, 2.0)  # >100% discount (error)
        if random.random() < 0.02:
            discount = -0.1  # negative discount (error)
        
        promotions.append({
            'promotion_id': f'PR{str(i).zfill(3)}',
            'promotion_name': random.choice(promo_names) if random.random() > 0.05 else None,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d') if random.random() > 0.03 else None,
            'discount_percentage': discount if random.random() > 0.02 else None,
            'applicable_category': random.choice(categories) if random.random() > 0.03 else None
        })
    
    return pd.DataFrame(promotions)

# ============================================================
# 5. LOYALTY_RULES TABLE
# ============================================================
def generate_loyalty_rules():
    rules = [
        {'rule_id': 1, 'rule_name': 'Standard Earning', 'points_per_unit_spend': 1.0, 'min_spend_threshold': 0, 'bonus_points': 0},
        {'rule_id': 2, 'rule_name': 'Weekend Bonus', 'points_per_unit_spend': 1.5, 'min_spend_threshold': 0, 'bonus_points': 50},
        {'rule_id': 3, 'rule_name': 'Big Spender', 'points_per_unit_spend': 2.0, 'min_spend_threshold': 100, 'bonus_points': 100},
        {'rule_id': 4, 'rule_name': 'Holiday Double', 'points_per_unit_spend': 2.0, 'min_spend_threshold': 0, 'bonus_points': 0},
        {'rule_id': 5, 'rule_name': 'First Purchase', 'points_per_unit_spend': 3.0, 'min_spend_threshold': 0, 'bonus_points': 200},
        {'rule_id': 6, 'rule_name': None, 'points_per_unit_spend': 1.0, 'min_spend_threshold': 50, 'bonus_points': None},  # missing values
    ]
    return pd.DataFrame(rules)

# ============================================================
# 6. STORE_SALES_HEADER TABLE
# ============================================================
def generate_sales_header(customers_df, stores_df, n_transactions=25000):
    customer_ids = customers_df['customer_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    
    # Customer purchase frequency (some buy more than others)
    customer_weights = np.random.exponential(1, len(customer_ids))
    customer_weights = customer_weights / customer_weights.sum()
    
    transactions = []
    
    for i in range(1, n_transactions + 1):
        # Select customer (weighted - some customers buy more)
        cust_id = np.random.choice(customer_ids, p=customer_weights)
        
        # Transaction date between 2023-01-01 and 2025-12-31
        trans_date = datetime(2023, 1, 1) + timedelta(
            days=random.randint(0, 1000),
            hours=random.randint(8, 21),
            minutes=random.randint(0, 59)
        )
        
        # Format date with some inconsistencies
        date_str = trans_date.strftime('%Y-%m-%d %H:%M:%S')
        if random.random() < 0.03:
            date_str = trans_date.strftime('%d-%m-%Y %H:%M')  # different format
        if random.random() < 0.02:
            date_str = trans_date.strftime('%Y/%m/%d')  # different format
        
        # Total amount (will be sum of line items, but add some discrepancies)
        total = round(random.uniform(10, 500), 2)
        if random.random() < 0.01:
            total = -abs(total)  # negative total (return/error)
        if random.random() < 0.02:
            total = None
        
        # Some missing customer IDs
        if random.random() < 0.05:
            cust_id = None
        
        # Phone from customer (with some mismatches)
        phone = None
        if cust_id and random.random() > 0.1:
            cust_row = customers_df[customers_df['customer_id'] == cust_id]
            if len(cust_row) > 0:
                phone = cust_row.iloc[0]['customer_phone']
        
        transactions.append({
            'transaction_id': f'T{str(i).zfill(6)}',
            'customer_id': cust_id,
            'store_id': random.choice(store_ids) if random.random() > 0.02 else None,
            'transaction_date': date_str if random.random() > 0.01 else None,
            'total_amount': total,
            'customer_phone': phone
        })
    
    return pd.DataFrame(transactions)

# ============================================================
# 7. STORE_SALES_LINE_ITEMS TABLE
# ============================================================
def generate_line_items(transactions_df, products_df, promotions_df):
    transaction_ids = transactions_df['transaction_id'].tolist()
    product_ids = products_df['product_id'].tolist()
    promotion_ids = promotions_df['promotion_id'].tolist() + [None] * 10  # Most items have no promo
    
    line_items = []
    line_id = 1
    
    for trans_id in transaction_ids:
        # Each transaction has 1-8 items
        n_items = random.randint(1, 8)
        
        for _ in range(n_items):
            prod_id = random.choice(product_ids)
            prod_row = products_df[products_df['product_id'] == prod_id]
            
            unit_price = 20.0  # default
            if len(prod_row) > 0 and prod_row.iloc[0]['unit_price'] is not None:
                unit_price = prod_row.iloc[0]['unit_price']
                if unit_price < 0:
                    unit_price = abs(unit_price)
            
            quantity = random.randint(1, 5)
            # Some negative quantities (returns)
            if random.random() < 0.03:
                quantity = -quantity
            # Some zero quantities (error)
            if random.random() < 0.01:
                quantity = 0
            
            line_amount = round(quantity * unit_price, 2)
            # Add some discrepancies
            if random.random() < 0.02:
                line_amount = round(line_amount * random.uniform(0.8, 1.2), 2)  # doesn't match qty * price
            if random.random() < 0.02:
                line_amount = None
            
            promo_id = random.choice(promotion_ids)
            
            line_items.append({
                'line_item_id': line_id,
                'transaction_id': trans_id if random.random() > 0.01 else None,
                'product_id': prod_id if random.random() > 0.02 else None,
                'promotion_id': promo_id,
                'quantity': quantity if random.random() > 0.02 else None,
                'line_item_amount': line_amount
            })
            line_id += 1
    
    return pd.DataFrame(line_items)

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("Generating raw retail dataset with realistic errors...")
    print("=" * 60)
    
    # Generate all tables
    print("1. Generating stores...")
    stores_df = generate_stores(15)
    
    print("2. Generating products...")
    products_df = generate_products(200)
    
    print("3. Generating customers...")
    customers_df = generate_customers(5000)
    
    print("4. Generating promotions...")
    promotions_df = generate_promotions(30)
    
    print("5. Generating loyalty rules...")
    loyalty_rules_df = generate_loyalty_rules()
    
    print("6. Generating sales transactions...")
    sales_header_df = generate_sales_header(customers_df, stores_df, 25000)
    
    print("7. Generating line items...")
    line_items_df = generate_line_items(sales_header_df, products_df, promotions_df)
    
    # Save to CSV
    print("\nSaving to CSV files...")
    stores_df.to_csv(f"{OUTPUT_DIR}/stores.csv", index=False)
    products_df.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
    customers_df.to_csv(f"{OUTPUT_DIR}/customer_details.csv", index=False)
    promotions_df.to_csv(f"{OUTPUT_DIR}/promotion_details.csv", index=False)
    loyalty_rules_df.to_csv(f"{OUTPUT_DIR}/loyalty_rules.csv", index=False)
    sales_header_df.to_csv(f"{OUTPUT_DIR}/store_sales_header.csv", index=False)
    line_items_df.to_csv(f"{OUTPUT_DIR}/store_sales_line_items.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"\nTable Summary:")
    print(f"  - stores.csv:                 {len(stores_df):,} rows")
    print(f"  - products.csv:               {len(products_df):,} rows")
    print(f"  - customer_details.csv:       {len(customers_df):,} rows")
    print(f"  - promotion_details.csv:      {len(promotions_df):,} rows")
    print(f"  - loyalty_rules.csv:          {len(loyalty_rules_df):,} rows")
    print(f"  - store_sales_header.csv:     {len(sales_header_df):,} rows")
    print(f"  - store_sales_line_items.csv: {len(line_items_df):,} rows")
    
    # Print data quality issues introduced
    print("\n" + "=" * 60)
    print("DATA QUALITY ISSUES INTRODUCED (for cleaning exercise):")
    print("=" * 60)
    print("""
    1. MISSING VALUES:
       - Null customer IDs in ~5% of transactions
       - Null emails, phones, names in customer data
       - Null prices in some products
       - Null dates in some records
    
    2. INCONSISTENT FORMATS:
       - Mixed date formats (YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY)
       - Inconsistent category casing (Electronics vs electronics)
       - Trailing spaces in some categories
    
    3. INVALID DATA:
       - Negative prices (errors)
       - Negative quantities (could be returns or errors)
       - Negative stock levels
       - Invalid emails (missing @)
       - Discount > 100% or negative discounts
       - Zero quantities
    
    4. OUTLIERS:
       - Some extremely high prices (100x normal)
       - Incomplete phone numbers
    
    5. REFERENTIAL ISSUES:
       - Some line items with null transaction_id
       - Some transactions with null customer_id
    """)
