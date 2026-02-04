# Problem Understanding

## 1. Problem Statement (In Our Own Words)

### Business Context
Retailers want to know not just what customers spent in the past, but **how much they will spend in the near future**. This helps with:
- Targeted marketing campaigns
- Inventory planning
- Customer retention strategies
- Resource allocation

### The Challenge
**Given a customer's historical behavior and attributes, predict the total amount they will spend in the next 30 days.**

This is essentially predicting **Short-Term Customer Lifetime Value (CLV)** over a fixed 30-day horizon.

---

## 2. Machine Learning Problem Framing

| Aspect | Definition |
|--------|------------|
| **Problem Type** | Supervised Learning - Regression |
| **Input (X)** | Customer features computed from data up to a cutoff date |
| **Output (y)** | Continuous numeric value (predicted spend amount) |

---

## 3. Target Variable Definition: "Short-Term Spend"

### Definition
```
short_term_spend = Total monetary amount a customer spends in the 30 days 
                   immediately following the cutoff date
```

### Calculation Logic
```python
# For each customer:
# 1. Set a cutoff_date (e.g., 2024-01-01)
# 2. Sum all transaction amounts from cutoff_date to cutoff_date + 30 days

future_spend_30d = sum(transaction_amount) 
                   WHERE transaction_date > cutoff_date 
                   AND transaction_date <= cutoff_date + 30 days
                   AND customer_id = current_customer
```

### Example
| Customer | Cutoff Date | Transactions in Next 30 Days | Target (future_spend_30d) |
|----------|-------------|------------------------------|---------------------------|
| C001 | 2024-01-01 | $50 + $30 + $20 | **$100** |
| C002 | 2024-01-01 | $200 | **$200** |
| C003 | 2024-01-01 | No purchases | **$0** |

---

## 4. Input Features (X) - Computed Before Cutoff Date

Features describe each customer's behavior **up to the cutoff date**:

### 4.1 RFM Features (Recency, Frequency, Monetary)
- **Recency**: Days since last purchase before cutoff
- **Frequency**: Number of transactions in past 30/60/90 days
- **Monetary**: Total/average spend in past 30/60/90 days

### 4.2 Transaction Behavior
- Average order value
- Number of orders
- Total items purchased
- Days between purchases (purchase cadence)

### 4.3 Product/Category Preferences
- Most purchased category
- Number of unique categories
- Category diversity score

### 4.4 Customer Attributes
- Loyalty status (Bronze/Silver/Gold)
- Total loyalty points
- Customer tenure (days since first purchase)
- Segment ID
- Region

### 4.5 Channel Usage
- Preferred store
- Number of unique stores visited

---

## 5. Data Leakage Prevention

**Critical Rule**: Features must only use data **before** the cutoff date.

```
Timeline:
|-------- Historical Data --------|-- Cutoff --|------ Future 30 Days ------|
        (Use for Features X)                      (Use for Target y)
```

- ✅ Use: Transactions before cutoff date
- ❌ Never use: Any information from after cutoff date

---

## 6. Success Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean Absolute Error | Average prediction error in dollars |
| **RMSE** | Root Mean Square Error | Penalizes large errors more |
| **R²** | Coefficient of Determination | % of variance explained (higher = better) |

### Business Interpretation
- MAE of $20 means: "On average, our predictions are off by $20"
- This helps stakeholders understand model accuracy in business terms

---

## 7. Key Assumptions

1. Past behavior is indicative of future spending
2. 30 days is a meaningful prediction horizon for retail
3. Customers with no future purchases have target = $0
4. All transaction amounts are in the same currency

---

## Summary

> **Goal**: Build a regression model that takes customer features (computed from historical data) and predicts how much that customer will spend in the next 30 days.

This prediction enables retailers to:
- Identify high-value customers for retention campaigns
- Allocate marketing budget efficiently
- Forecast short-term revenue
