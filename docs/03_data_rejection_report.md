# Data Rejection Report

## Overview

During the data cleaning process, certain records were identified as invalid, incomplete, or unusable for the Customer Spend Prediction model. These records have been moved to the `data/rejected/` directory for audit and review purposes.

**Total Records Rejected:** 21,048 records across 4 tables

---

## 1. Sales Header Rejections

**File:** `sales_header_rejected.csv`  
**Total Rejected:** 2,163 transactions

| Rejection Reason | Count | % of Rejected | Description |
|------------------|-------|---------------|-------------|
| Missing customer_id | 1,240 | 57.3% | No customer identifier - cannot attribute transaction to any customer |
| Missing total amount | 442 | 20.4% | Transaction value is NULL - cannot use for spend prediction |
| Negative total amount | 254 | 11.7% | Transaction has negative value (possible return or data error) |
| Invalid transaction date | 227 | 10.5% | Date could not be parsed or is NULL |

### Why These Were Rejected

#### 1.1 Missing Customer ID (1,240 records)
```
Reason: Cannot attribute transaction to any customer
Impact: These transactions cannot be used to calculate customer-level features
        or target variable (future spend)
Action: Removed from dataset - no way to identify the customer
```

**Example:**
| transaction_id | customer_id | total_amount | rejection_reason |
|----------------|-------------|--------------|------------------|
| T000035 | NULL | NULL | Missing customer_id |
| T000060 | NULL | 294.93 | Missing customer_id |

#### 1.2 Missing Total Amount (442 records)
```
Reason: Transaction value is unknown
Impact: Cannot calculate actual spend for this transaction
Action: Removed - amount is critical for spend prediction model
```

#### 1.3 Negative Total Amount (254 records)
```
Reason: Negative monetary values indicate returns or refunds
Impact: Returns complicate spend prediction and may skew results
Action: Removed from training data to avoid noise
Note:   Could be analyzed separately for return behavior patterns
```

#### 1.4 Invalid Transaction Date (227 records)
```
Reason: Date field was NULL or in an unrecognizable format
Impact: Cannot determine when transaction occurred
        Cannot apply cutoff date logic for train/test split
Action: Removed - temporal information is critical
```

---

## 2. Line Items Rejections

**File:** `line_items_rejected.csv`  
**Total Rejected:** 18,879 line items

| Rejection Reason | Count | % of Rejected | Description |
|------------------|-------|---------------|-------------|
| Transaction was rejected | 9,645 | 51.1% | Parent transaction was rejected |
| Negative quantity | 3,121 | 16.5% | Item quantity is negative (possible return) |
| Missing quantity | 2,040 | 10.8% | Quantity field is NULL |
| Missing product_id | 1,935 | 10.3% | Cannot identify which product was purchased |
| Missing transaction_id | 1,151 | 6.1% | Cannot link to parent transaction |
| Zero quantity | 987 | 5.2% | Quantity is zero (invalid record) |

### Why These Were Rejected

#### 2.1 Transaction Was Rejected (9,645 records)
```
Reason: Parent transaction in sales_header was rejected
Impact: Orphaned line items with no valid transaction context
Action: Automatically rejected to maintain referential integrity
```

#### 2.2 Negative Quantity (3,121 records)
```
Reason: Negative quantities typically indicate returns
Impact: Returns should not be counted as purchases for spend prediction
Action: Removed - could be used separately for return analysis
```

**Example:**
| line_item_id | product_id | quantity | rejection_reason |
|--------------|------------|----------|------------------|
| 1523 | P0042 | -2 | Negative quantity (possible return) |
| 2891 | P0108 | -1 | Negative quantity (possible return) |

#### 2.3 Missing Quantity (2,040 records)
```
Reason: Quantity field is NULL
Impact: Cannot calculate line item value (quantity × price)
Action: Removed - quantity is essential for spend calculation
```

#### 2.4 Missing Product ID (1,935 records)
```
Reason: Cannot identify which product was purchased
Impact: Cannot join with products table for category analysis
        Cannot calculate proper line item amount
Action: Removed - product identification is required
```

#### 2.5 Missing Transaction ID (1,151 records)
```
Reason: Cannot link line item to any transaction
Impact: Orphaned record with no transaction context
Action: Removed - referential integrity violated
```

#### 2.6 Zero Quantity (987 records)
```
Reason: Purchasing zero items is not a valid transaction
Impact: Adds no value to spend analysis
Action: Removed - invalid business logic
```

---

## 3. Products Rejections

**File:** `products_rejected.csv`  
**Total Rejected:** 1 product

| Rejection Reason | Count | Description |
|------------------|-------|-------------|
| Extreme price outlier (>1000) | 1 | Price was abnormally high (likely 100× error) |

### Why This Was Rejected

#### 3.1 Extreme Price Outlier (1 record)
```
Reason: Unit price exceeded $1,000 (detected as 100× data entry error)
Impact: Would severely skew average calculations and model training
Action: Flagged for review - price was corrected in cleaned data
        (divided by 100), original stored in rejected file
```

**Example:**
| product_id | product_name | unit_price | rejection_reason |
|------------|--------------|------------|------------------|
| P0089 | Bracelet V2 | 11,523.45 | Extreme price outlier (>1000) |

**Note:** The product was NOT removed from the cleaned dataset. The price was corrected to $115.23 and the original erroneous record was saved in rejected for audit.

---

## 4. Promotions Rejections

**File:** `promotions_rejected.csv`  
**Total Rejected:** 5 promotions

| Rejection Reason | Count | Description |
|------------------|-------|-------------|
| Invalid discount percentage | 5 | Discount > 100% or negative value |

### Why These Were Rejected

#### 4.1 Invalid Discount Percentage (5 records)
```
Reason: Discount percentage was greater than 1.0 (>100%) or negative
Impact: Invalid discount rates would cause incorrect price calculations
Action: Flagged for review - discounts were capped at 50% in cleaned data
        Original erroneous records saved in rejected file
```

**Example:**
| promotion_id | promotion_name | discount_percentage | rejection_reason |
|--------------|----------------|---------------------|------------------|
| PR007 | Flash Sale | 1.45 (145%) | Invalid discount percentage |
| PR019 | Summer Sale | -0.10 (-10%) | Invalid discount percentage |

**Note:** These promotions were NOT removed from the cleaned dataset. The discount values were corrected (capped at 0.50) and original records saved for audit.

---

## Summary Statistics

### Rejection Rate by Table

| Table | Original Rows | Rejected | Rejection Rate |
|-------|---------------|----------|----------------|
| store_sales_header | 25,000 | 2,163 | **8.65%** |
| store_sales_line_items | 112,656 | 18,879 | **16.76%** |
| products | 196 | 1 | **0.51%** |
| promotion_details | 30 | 5 | **16.67%** |

### Records Available for Analysis

| Table | Cleaned Rows |
|-------|--------------|
| stores | 15 |
| products | 196 |
| customer_details | 5,000 |
| promotion_details | 30 |
| loyalty_rules | 6 |
| store_sales_header | **22,837** |
| store_sales_line_items | **93,777** |

---

## Recommendations for Data Quality Improvement

Based on the rejection analysis, here are recommendations for improving data quality at the source:

### 1. Customer ID Validation
- **Issue:** 5% of transactions missing customer_id
- **Recommendation:** Make customer_id a required field at point of sale
- **Business Impact:** Losing ~$300K+ in attributable transaction data

### 2. Data Entry Validation
- **Issue:** Negative quantities, zero quantities, extreme prices
- **Recommendation:** Add validation rules at data entry:
  - Quantity must be > 0
  - Price must be within category-specific ranges
  - Discount must be between 0% and 100%

### 3. Date Format Standardization
- **Issue:** Multiple date formats causing parsing failures
- **Recommendation:** Enforce ISO 8601 format (YYYY-MM-DD) at source

### 4. Referential Integrity
- **Issue:** Line items without valid transaction_id or product_id
- **Recommendation:** Enforce foreign key constraints in database

---

## File Locations

| File | Path | Purpose |
|------|------|---------|
| Rejected Sales | `data/rejected/sales_header_rejected.csv` | Invalid transactions |
| Rejected Line Items | `data/rejected/line_items_rejected.csv` | Invalid line items |
| Rejected Products | `data/rejected/products_rejected.csv` | Price outliers |
| Rejected Promotions | `data/rejected/promotions_rejected.csv` | Invalid discounts |

---

*Report Generated: 2026-02-04*  
*Data Cleaning Script: scripts/data_cleaning.py*
