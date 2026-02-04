# Dataset Scouting & Choice

## 1. Candidate Datasets Identified

### Dataset 1: UCI Online Retail II Dataset
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

| Criteria | Status | Details |
|----------|--------|---------|
| Retail Context | ✅ | UK-based online retail (gifts/homewares) |
| Customer ID | ✅ | `Customer ID` field present |
| Timestamps | ✅ | `InvoiceDate` with date and time |
| Transaction Values | ✅ | `Quantity` × `Price` = transaction value |
| Multiple Purchases/Customer | ✅ | ~5,000+ customers with repeat purchases |
| Size | ✅ | ~1M transactions, 5,942 customers |
| Publicly Available | ✅ | Free, no restrictions |
| Format | ✅ | CSV/Excel |

**Fields Available:**
- `Invoice` - Invoice number (transaction ID)
- `StockCode` - Product code
- `Description` - Product name
- `Quantity` - Units purchased
- `InvoiceDate` - Date and time of transaction
- `Price` - Unit price
- `Customer ID` - Unique customer identifier
- `Country` - Customer's country

**Date Range:** 01/12/2009 to 09/12/2011 (~2 years)

---

### Dataset 2: Brazilian E-Commerce (Olist) Dataset
**Source:** [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| Criteria | Status | Details |
|----------|--------|---------|
| Retail Context | ✅ | Brazilian e-commerce marketplace |
| Customer ID | ✅ | `customer_unique_id` field |
| Timestamps | ✅ | `order_purchase_timestamp` |
| Transaction Values | ✅ | `price` + `freight_value` |
| Multiple Purchases/Customer | ⚠️ | Limited repeat customers (~3%) |
| Size | ✅ | ~100K orders, 99K customers |
| Publicly Available | ✅ | Free on Kaggle |
| Format | ✅ | CSV |

**Limitation:** Most customers have only 1 order, making it harder to compute meaningful historical features.

---

### Dataset 3: Instacart Market Basket Dataset
**Source:** [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis)

| Criteria | Status | Details |
|----------|--------|---------|
| Retail Context | ✅ | Grocery delivery |
| Customer ID | ✅ | `user_id` field |
| Timestamps | ⚠️ | Only order sequence, no actual dates |
| Transaction Values | ❌ | No price information |
| Multiple Purchases/Customer | ✅ | Multiple orders per user |

**Limitation:** No price/monetary values - cannot compute spend predictions.

---

## 2. Dataset Comparison Summary

| Criteria | UCI Online Retail II | Olist Brazil | Instacart |
|----------|---------------------|--------------|-----------|
| Customer ID | ✅ | ✅ | ✅ |
| Timestamps | ✅ Full datetime | ✅ Full datetime | ⚠️ Sequence only |
| Monetary Values | ✅ Price × Qty | ✅ Price | ❌ None |
| Repeat Customers | ✅ High (~60%) | ⚠️ Low (~3%) | ✅ High |
| Data Size | ~1M rows | ~100K rows | ~3M rows |
| Date Range | 2 years | 2 years | N/A |
| **Overall Fit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 3. Final Choice: UCI Online Retail II Dataset

### Justification

#### ✅ Perfect Fit for Problem Requirements
1. **Customer Identification**: Clear `Customer ID` allows tracking individual customers
2. **Timestamps**: Full `InvoiceDate` enables cutoff-based train/test split and temporal features
3. **Monetary Values**: `Quantity × Price` gives transaction value for target calculation
4. **Repeat Purchases**: High percentage of returning customers enables meaningful historical features

#### ✅ Meets Data Requirements from Problem Statement
| Requirement | UCI Dataset |
|-------------|-------------|
| 4.1 Retail context | ✅ Online retail store |
| 4.2 Customer identification | ✅ Customer ID field |
| 4.3 Timestamps | ✅ InvoiceDate |
| 4.4 Monetary measure | ✅ Quantity × Price |
| 4.5 Size (5K+ customers, 20K+ transactions) | ✅ ~5,942 customers, ~1M transactions |
| 4.6 Publicly available | ✅ UCI ML Repository |

#### ✅ Additional Benefits
- **Product Categories**: Can derive from `Description` or `StockCode`
- **Geographic Data**: `Country` field for regional analysis
- **Clean Structure**: Single table, easy to work with
- **Well-Documented**: Extensively used in academic research

#### ⚠️ Limitations to Address
1. **Missing Customer IDs**: ~25% of transactions have null Customer ID → Will filter these out
2. **Cancellations**: Negative quantities indicate returns → Will handle in data cleaning
3. **No Customer Demographics**: No age, gender, loyalty tier → Will engineer features from behavior

---

## 4. Dataset Download Instructions

```bash
# Option 1: Direct download from UCI
wget https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip

# Option 2: Using Python
from ucimlrepo import fetch_ucirepo
online_retail_ii = fetch_ucirepo(id=502)
df = online_retail_ii.data.features
```

**Alternative Kaggle Mirror:**
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

---

## 5. Expected Data Schema After Loading

| Column | Type | Description | Use |
|--------|------|-------------|-----|
| Invoice | string | Transaction ID | Group items in same order |
| StockCode | string | Product ID | Product features |
| Description | string | Product name | Category extraction |
| Quantity | int | Units purchased | Calculate transaction value |
| InvoiceDate | datetime | Transaction timestamp | Cutoff logic, recency |
| Price | float | Unit price | Calculate transaction value |
| Customer ID | float | Customer identifier | Group by customer |
| Country | string | Customer country | Geographic feature |

**Derived Fields:**
```python
transaction_value = Quantity × Price
```

---

## 6. Preliminary Data Quality Notes

Based on dataset documentation:
- **Total Records**: ~1,067,371 transactions
- **Unique Customers**: ~5,942 (with ID)
- **Date Range**: Dec 2009 - Dec 2011
- **Countries**: 43 countries (mainly UK ~91%)

### Data Cleaning Required
1. Remove rows where `Customer ID` is null
2. Remove cancelled orders (Invoice starting with 'C')
3. Remove rows with negative `Quantity` or `Price`
4. Handle outliers in `Price` and `Quantity`

---

## Summary

> **Selected Dataset**: UCI Online Retail II
> 
> **Reason**: Best fit for predicting 30-day customer spend with clear customer IDs, full timestamps, monetary values, and high repeat purchase rate.

Next Step: Download dataset and begin data loading & cleaning.
