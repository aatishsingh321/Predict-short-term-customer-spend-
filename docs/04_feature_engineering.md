# Feature Engineering Documentation

## Overview

This document explains the feature engineering process for the **30-Day Customer Spend Prediction** project. The goal is to transform raw transactional data into a customer-level dataset suitable for training a regression model.

---

## Purpose of `customer_features.csv`

The `customer_features.csv` file is a **customer-level dataset** where:
- **Each row** = One customer
- **Columns** = Features (X) describing the customer's historical behavior + Target (y) representing future spend

### What It Will Be Used For:

| Stage | Usage |
|-------|-------|
| **EDA** | Analyze distributions, correlations, identify patterns |
| **Model Training** | Features (X) are inputs to the regression model |
| **Model Evaluation** | Target (y) = `future_spend_30d` is what the model predicts |
| **Inference** | Same features will be computed for new customers to make predictions |

---

## Data Split Logic

```
Timeline:
├────────────────────────────────────┼──────────────────────────┤
│         HISTORICAL DATA            │      FUTURE DATA         │
│      (Used for Features X)         │    (Used for Target y)   │
│                                    │                          │
│    2023-01-01 to 2024-12-31        │  2025-01-01 to 2025-01-31│
├────────────────────────────────────┼──────────────────────────┤
                                   CUTOFF
                                 2025-01-01
```

**Critical Rule**: Features are computed ONLY from data before the cutoff date to prevent data leakage.

---

## Feature Categories

### 1. RFM Features (Recency, Frequency, Monetary)

RFM analysis is a proven marketing technique for customer segmentation. These features capture core purchasing behavior.

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `recency_days` | Days since last purchase before cutoff | Recent buyers are more likely to buy again |
| `frequency_30d` | # transactions in last 30 days | Short-term activity indicates engagement |
| `frequency_60d` | # transactions in last 60 days | Medium-term purchase patterns |
| `frequency_90d` | # transactions in last 90 days | Quarterly buying behavior |
| `frequency_180d` | # transactions in last 180 days | Semi-annual patterns |
| `frequency_365d` | # transactions in last 365 days | Annual purchasing cadence |
| `frequency_total` | Total # transactions ever | Overall engagement level |
| `monetary_30d` | Total spend in last 30 days | Recent spending power |
| `monetary_60d` | Total spend in last 60 days | Medium-term spend |
| `monetary_90d` | Total spend in last 90 days | Quarterly spend |
| `monetary_180d` | Total spend in last 180 days | Semi-annual spend |
| `monetary_365d` | Total spend in last 365 days | Annual spend |
| `monetary_total` | Lifetime total spend | Customer lifetime value indicator |
| `avg_order_value` | Average transaction amount | Typical basket size |
| `avg_order_value_30d/60d/90d` | AOV for time windows | Recent vs historical basket size |
| `max_order_value` | Largest single transaction | Big purchase potential |
| `min_order_value` | Smallest single transaction | Minimum engagement |
| `std_order_value` | Variability in order amounts | Consistency of spending |

**Why RFM?**
- Customers who bought recently → more likely to buy again
- Customers who buy frequently → higher future spend
- Customers who spend more → likely to continue spending

---

### 2. Product/Category Preference Features

These features capture WHAT customers buy, not just HOW MUCH.

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `unique_categories` | # distinct product categories purchased | Breadth of interest |
| `unique_products` | # distinct products purchased | Variety seeking behavior |
| `total_items_purchased` | Sum of quantities bought | Volume buyer indicator |
| `top_category` | Most frequently purchased category | Primary interest area |
| `pct_spend_electronics` | % of spend on Electronics | Category affinity |
| `pct_spend_apparel` | % of spend on Apparel | Category affinity |
| `pct_spend_home_garden` | % of spend on Home & Garden | Category affinity |
| ... | (one column per category) | Category-specific behavior |

**Why Category Features?**
- Category preferences indicate what promotions might work
- Customers buying across categories may have higher lifetime value
- Seasonal categories affect future spending patterns

---

### 3. Store/Channel Usage Features

These features capture WHERE customers shop.

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `unique_stores` | # distinct stores visited | Multi-channel behavior |
| `unique_regions` | # distinct regions shopped in | Geographic mobility |
| `primary_store_id` | Most frequently visited store | Home store preference |
| `primary_region` | Most frequent shopping region | Regional patterns |

**Why Channel Features?**
- Customers shopping at multiple stores may be more engaged
- Regional preferences can indicate lifestyle/demographics
- Store-specific promotions can be targeted

---

### 4. Customer Attribute Features

Static or slowly-changing customer characteristics from the customer master table.

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `loyalty_status` | Tier: Bronze/Silver/Gold/Platinum | Higher tiers = more valuable customers |
| `total_loyalty_points` | Accumulated points | Engagement with loyalty program |
| `segment_id` | Customer segment (LP, AR, NR, HC, HS) | Marketing segment classification |
| `tenure_days` | Days since customer registration | Longer tenure = more stable behavior |

**Why Customer Attributes?**
- Loyalty tier strongly correlates with future spend
- Tenure indicates customer maturity and predictability
- Segments may have different spending patterns

---

### 5. Behavioral Features

These capture HOW customers shop (timing patterns).

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `avg_days_between_purchases` | Average gap between transactions | Purchase rhythm |
| `std_days_between_purchases` | Variability in purchase timing | Consistency of behavior |
| `avg_purchase_hour` | Average hour of day for purchases | Time preference |
| `preferred_time_of_day` | Morning/Afternoon/Evening/Night | Shopping time pattern |
| `preferred_day` | Most common day of week | Weekday preference |
| `weekend_transaction_pct` | % of purchases on weekends | Weekend vs weekday shopper |

**Why Behavioral Features?**
- Regular purchase cadence predicts future buying
- Time patterns can indicate lifestyle (e.g., working professionals shop evenings)
- Weekend shoppers may have different basket sizes

---

### 6. Target Variables

| Feature | Description | Usage |
|---------|-------------|-------|
| `future_spend_30d` | Total spend in 30 days after cutoff | **PRIMARY TARGET** for regression |
| `future_transactions_30d` | # transactions in 30 days after cutoff | Optional secondary target |
| `will_purchase_30d` | Binary: 1 if any purchase, 0 otherwise | For classification models |

---

## Dataset Statistics

```
Total Customers: 3,818
Total Features: 52 (excluding customer_id)
Target: future_spend_30d

Target Distribution:
├── Customers who will purchase: 591 (15.5%)
├── Customers who won't purchase: 3,227 (84.5%)
├── Mean future spend: $44.69
├── Median future spend: $0.00
└── Max future spend: ~$2,000+
```

---

## How Features Connect to Model Training

```
┌─────────────────────────────────────────────────────────────────┐
│                    customer_features.csv                        │
│                                                                 │
│  ┌───────────┬─────────────────────────┬──────────────────┐    │
│  │customer_id│    FEATURES (X)         │   TARGET (y)     │    │
│  ├───────────┼─────────────────────────┼──────────────────┤    │
│  │  C00001   │ recency=15, freq=4, ... │ future_spend=125 │    │
│  │  C00002   │ recency=200, freq=1,... │ future_spend=0   │    │
│  │  C00003   │ recency=5, freq=10, ... │ future_spend=350 │    │
│  │    ...    │         ...             │       ...        │    │
│  └───────────┴─────────────────────────┴──────────────────┘    │
│                         │                        │              │
│                         ▼                        ▼              │
│              ┌─────────────────┐    ┌─────────────────┐        │
│              │ Train Features  │    │  Train Target   │        │
│              │       X         │    │       y         │        │
│              └────────┬────────┘    └────────┬────────┘        │
│                       │                      │                  │
│                       └──────────┬───────────┘                  │
│                                  ▼                              │
│                    ┌─────────────────────────┐                  │
│                    │   REGRESSION MODEL      │                  │
│                    │  (Random Forest/XGBoost)│                  │
│                    └─────────────────────────┘                  │
│                                  │                              │
│                                  ▼                              │
│                    ┌─────────────────────────┐                  │
│                    │  Predicted 30-Day Spend │                  │
│                    └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Engineering Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Multiple time windows (30/60/90/180/365 days) | Captures both recent and historical behavior |
| Category spend percentages | Normalizes across different spending levels |
| Cutoff date = 2025-01-01 | Leaves sufficient future data for target calculation |
| Fill missing numerics with 0 | Appropriate for count/sum features (no activity = 0) |
| Fill missing categoricals with "Unknown" | Preserves information about missing data |
| Include behavioral timing features | Adds signals beyond just monetary value |

---

## Next Steps

1. **EDA**: Analyze feature distributions and correlations
2. **Feature Selection**: Remove highly correlated or low-importance features
3. **Preprocessing**: Scale numeric features, encode categorical features
4. **Model Training**: Train regression models (Linear, Ridge, Random Forest, XGBoost)
5. **Evaluation**: Compare models using MAE, RMSE, R²

---

## Files Generated

| File | Description |
|------|-------------|
| `data/processed/customer_features.csv` | Main dataset for modeling |
| `data/processed/feature_list.txt` | List of all features with groupings |
| `scripts/feature_engineering.py` | Source code for reproducibility |
