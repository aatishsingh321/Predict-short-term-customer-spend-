# Exploratory Data Analysis (EDA) Documentation

## Overview

This document explains the EDA process performed on the Customer Spend Prediction dataset, including what analyses were conducted, why each analysis was chosen, and the key findings.

**Script:** `scripts/eda_analysis.py`  
**Output Plots:** `outputs/eda_plots/`

---

## 1. Descriptive Statistics

### What We Did
Calculated summary statistics (mean, median, std, min, max, percentiles) for key numerical variables.

### Why We Did It
Descriptive statistics provide a quick understanding of:
- **Central tendency**: What is the typical value?
- **Spread**: How variable is the data?
- **Extremes**: Are there outliers?

### Key Findings

| Metric | Value | Insight |
|--------|-------|---------|
| Avg Transaction Amount | $256.24 | Moderate basket size |
| Median Transaction | $257.37 | Symmetric distribution (mean ≈ median) |
| Avg Transactions/Customer | 5.6 | Customers make ~6 purchases on average |
| Avg Total Spend/Customer | $1,435.32 | Significant variation in customer value |
| Max Spend/Customer | $10,346.28 | High-value customers exist |

---

## 2. Distribution Plots

### 2.1 Transaction Amount Distribution
**File:** `01_transaction_amount_distribution.png`

**What:** Histogram and box plot of individual transaction amounts.

**Why:** Understanding transaction distribution helps:
- Identify if data is normally distributed or skewed
- Detect outliers that may affect model training
- Determine if transformations (log, sqrt) are needed

**Finding:** Nearly uniform distribution between $10-$500, no extreme outliers after cleaning.

---

### 2.2 Customer Total Spend Distribution
**File:** `02_customer_spend_distribution.png`

**What:** Histogram of total spend per customer, plus log-transformed view.

**Why:** 
- Customer-level spend is our target's basis
- Right-skewed distributions are common in spend data
- Log transformation often helps with modeling

**Finding:** Right-skewed distribution - many low spenders, few high spenders. Log transformation makes it more normal.

---

### 2.3 Transactions per Customer
**File:** `03_transactions_per_customer.png`

**What:** Distribution of how many times each customer purchased.

**Why:**
- Frequency is a key RFM metric
- Helps identify heavy vs. light users
- Impacts feature engineering decisions

**Finding:** Most customers have 1-10 transactions, with a long tail up to 40. This suggests varying engagement levels.

---

### 2.4 Loyalty Status Analysis
**File:** `04_loyalty_status_analysis.png`

**What:** Pie chart of customer distribution + bar chart of avg transaction by loyalty tier.

**Why:**
- Loyalty status is a key customer attribute
- Need to verify if loyalty tiers correlate with spending
- Important for segmentation and feature encoding

**Finding:** 
- Bronze: 49.6%, Silver: 31.1%, Gold: 14.5%, Platinum: 4.9%
- Higher tiers show higher average transaction amounts

---

### 2.5 Product Category Analysis
**File:** `05_product_category_analysis.png`

**What:** Revenue and item count by product category.

**Why:**
- Understand which categories drive revenue
- Identify popular vs. niche categories
- Guide category-based feature engineering

**Finding:** Revenue is relatively balanced across categories, with Jewelry and Electronics leading.

---

### 2.6 Time Series Analysis
**File:** `06_time_series_analysis.png`

**What:** Daily transaction count and revenue over time.

**Why:**
- Detect seasonality or trends
- Identify any anomalies or data gaps
- Critical for understanding temporal patterns

**Finding:** Relatively stable transaction volume over the 1000-day period with some variance.

---

### 2.7 Day of Week Analysis
**File:** `07_day_of_week_analysis.png`

**What:** Transaction count and average amount by day of week.

**Why:**
- Retail often has day-of-week patterns (e.g., weekend spikes)
- May indicate customer types (weekday vs. weekend shoppers)
- Useful for temporal features

**Finding:** Fairly even distribution across weekdays with slight variations.

---

### 2.8 Hour of Day Analysis
**File:** `08_hour_of_day_analysis.png`

**What:** Transaction distribution across hours (8am-9pm).

**Why:**
- Identify peak shopping hours
- Distinguish morning vs. evening shoppers
- Potential feature for customer profiling

**Finding:** Peak activity between 10am-6pm, as expected for retail.

---

### 2.9 Store Region Analysis
**File:** `09_store_region_analysis.png`

**What:** Revenue and average transaction by store region.

**Why:**
- Geographic patterns may affect spending
- Regional features could improve predictions
- Identify high-performing regions

**Finding:** All regions perform similarly, with Southwest having slightly higher average transactions.

---

### 2.10 Customer Tenure Analysis
**File:** `10_customer_tenure_analysis.png`

**What:** Distribution of customer tenure + scatter plot of tenure vs. total spend.

**Why:**
- Tenure indicates customer maturity
- Longer tenure may correlate with higher lifetime value
- Important for customer segmentation

**Finding:** Wide range of tenure (30-1500 days). Weak but positive correlation with total spend.

---

### 2.11 Loyalty Points vs Spend
**File:** `11_loyalty_points_vs_spend.png`

**What:** Scatter plot of loyalty points vs. total customer spend.

**Why:**
- Verify if loyalty program engagement correlates with spending
- Identify if points are a useful feature

**Finding:** Weak correlation (-0.024), suggesting loyalty points alone don't predict spend well.

---

### 2.12 Correlation Heatmap
**File:** `12_correlation_heatmap.png`

**What:** Correlation matrix of customer-level features.

**Why:**
- Identify highly correlated features (multicollinearity)
- Find features strongly related to target (total_spend)
- Guide feature selection

**Correlation Matrix:**
| Feature | total_spend | Interpretation |
|---------|-------------|----------------|
| num_transactions | 0.969 | **Very strong** - More purchases = more spend |
| num_stores | 0.913 | **Very strong** - Multi-store shoppers spend more |
| avg_transaction | 0.172 | Moderate - Higher basket size helps |
| loyalty_points | -0.024 | Weak negative - Not predictive |
| tenure_days | -0.021 | Weak - Tenure alone not predictive |

---

### 2.13 RFM Distributions
**File:** `13_rfm_distributions.png`

**What:** Histograms of Recency, Frequency, and Monetary metrics.

**Why:**
- RFM is the foundation of customer value analysis
- These three metrics are proven predictors of future behavior
- Forms basis for customer segmentation

**Finding:**
| Metric | Mean | Median | Insight |
|--------|------|--------|---------|
| Recency | 242 days | 157 days | Many inactive customers |
| Frequency | 5.6 | 4 | Right-skewed, few heavy buyers |
| Monetary | $1,435 | $1,030 | Right-skewed, few high spenders |

---

## 3. Initial Hypotheses

Based on EDA findings, we formulated 8 hypotheses to guide feature engineering:

### Hypothesis 1: Past Spend Predicts Future Spend
**Evidence:** Strong correlation (0.969) between transaction count and total spend.  
**Features to Create:** `total_spend_30d`, `total_spend_60d`, `total_spend_90d`

### Hypothesis 2: Purchase Frequency Indicates Engagement
**Evidence:** High variance in transactions per customer (1 to 40).  
**Features to Create:** `num_transactions_30d`, `num_transactions_60d`, `avg_days_between_purchases`

### Hypothesis 3: Recency Affects Future Purchases
**Evidence:** 45.4% of customers are churned (>180 days since last purchase).  
**Features to Create:** `days_since_last_purchase`, `is_active_30d`, `is_active_60d`

### Hypothesis 4: Loyalty Status Correlates with Spend
**Evidence:** Higher loyalty tiers have higher average transaction amounts.  
**Features to Create:** `loyalty_status_encoded`, `total_loyalty_points`

### Hypothesis 5: Customer Tenure Affects Behavior
**Evidence:** Weak positive correlation between tenure and spend.  
**Features to Create:** `customer_tenure_days`, `tenure_bucket`

### Hypothesis 6: Category Preferences Indicate Future Purchases
**Evidence:** Clear category preferences vary by customer.  
**Features to Create:** `top_category`, `num_categories`, `category_diversity_score`

### Hypothesis 7: Store Behavior Reflects Shopping Patterns
**Evidence:** Correlation (0.913) between stores visited and total spend.  
**Features to Create:** `num_stores_visited`, `preferred_store`, `preferred_region`

### Hypothesis 8: Temporal Patterns Exist
**Evidence:** Variations in transactions by day of week and hour.  
**Features to Create:** `preferred_day_of_week`, `preferred_hour`, `weekend_shopper_flag`

---

## 4. Key Insights Summary

### Customer Behavior
- **Active Customers (30 days):** 601 (14.7%)
- **Churned Customers (>180 days):** 1,849 (45.4%)
- **High-Value Customers (top 10%):** 408

### Data Characteristics
- Transaction amounts are uniformly distributed ($10-$500)
- Customer spend is right-skewed (many low, few high spenders)
- Strong correlation between frequency and monetary value

### Recommended Features for Modeling
1. **RFM Features** - Recency, Frequency, Monetary
2. **Historical Spend Windows** - 30d, 60d, 90d aggregations
3. **Loyalty Attributes** - Status (encoded), Points
4. **Customer Tenure** - Days since first purchase
5. **Category Preferences** - Top category, diversity
6. **Store Behavior** - Stores visited, preferred region
7. **Temporal Patterns** - Day of week, hour preferences
8. **Transaction Metrics** - Avg order value, basket size

---

## 5. Files Generated

| File | Location | Description |
|------|----------|-------------|
| 13 PNG plots | `outputs/eda_plots/` | All visualizations |
| customer_rfm.csv | `data/cleaned/` | RFM metrics per customer |
| customer_features_basic.csv | `data/cleaned/` | Basic customer features |

---

## 6. Why These EDA Methods Were Chosen

| Method | Purpose | When to Use |
|--------|---------|-------------|
| **Descriptive Statistics** | Understand data distribution | Always - first step in any analysis |
| **Histograms** | Visualize distribution shape | Continuous variables |
| **Box Plots** | Identify outliers and spread | Comparing groups or detecting anomalies |
| **Bar Charts** | Compare categories | Categorical variables |
| **Pie Charts** | Show proportions | Few categories (<6) |
| **Time Series Plots** | Detect trends/seasonality | Temporal data |
| **Scatter Plots** | Find relationships | Two continuous variables |
| **Correlation Heatmap** | Identify feature relationships | Multiple numeric features |
| **RFM Analysis** | Customer segmentation | Retail/transaction data |

---

## 7. Next Steps

Based on EDA findings, the next phase will:

1. **Feature Engineering** - Create all hypothesized features
2. **Define Target Variable** - Calculate 30-day future spend
3. **Train/Test Split** - Use cutoff date approach
4. **Model Training** - Start with baseline, then advanced models

---

*EDA Completed: 2026-02-04*  
*Script: scripts/eda_analysis.py*
