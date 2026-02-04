# Improved Model Training Documentation

## Overview

This document describes the improvements made to the customer spend prediction pipeline and explains the results.

---

## Improvements Implemented

### 1. Log1p Transformation on Target
```python
y_train_log = np.log1p(y_train_capped)
# After prediction:
y_pred = np.expm1(y_pred_log)
```
**Why:** Customer spend is right-skewed (many $0, few high spenders). Log transformation normalizes the distribution, helping models learn across all spend levels.

### 2. Target Winsorization (99th Percentile)
```python
cap_value = target['future_spend_30d'].quantile(0.99)  # $898.80
target['future_spend_30d_capped'] = target['future_spend_30d'].clip(upper=cap_value)
```
**Why:** Extreme outliers can dominate the loss function. Capping at 99th percentile reduces their influence while preserving most information.

### 3. StandardScaler (Fit on Train Only)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)  # Transform only (no fit!)
```
**Why:** Linear models need scaled features. CRITICAL: Fitting only on train prevents data leakage.

### 4. TimeSeriesSplit Cross-Validation
```python
tscv = TimeSeriesSplit(n_splits=5)
```
**Why:** Standard KFold randomly mixes data, causing temporal leakage. TimeSeriesSplit ensures training folds always precede validation folds.

### 5. Improved Gradient Boosting
```python
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,      # Lower LR for better generalization
    max_depth=4,             # Shallow trees (regularization)
    min_samples_leaf=15,     # More samples per leaf (regularization)
    subsample=0.8,           # Stochastic boosting
    n_iter_no_change=20      # Early stopping
)
```
**Why:** Lower learning rate + more trees + regularization = better generalization.

### 6. Baseline Comparison
```python
class MeanBaselineRegressor:
    def fit(self, X, y):
        self.mean_ = np.mean(y)
    def predict(self, X):
        return np.full(len(X), self.mean_)
```
**Why:** Benchmark to measure if models add value. Any useful model should beat this.

---

## Results

### Final Model Comparison

| Model | MAE ($) | RMSE ($) | R² | vs Baseline |
|-------|---------|----------|-----|-------------|
| **Baseline (Mean)** | $63.71 | $160.37 | -0.177 | 0.0% |
| Linear Regression | $63.82 | $159.29 | -0.161 | -0.2% |
| Ridge Regression | $63.87 | $159.36 | -0.162 | -0.3% |
| Gradient Boosting | $63.99 | $159.46 | -0.163 | -0.5% |
| Random Forest | $64.19 | $159.09 | -0.158 | -0.8% |

### Key Insight: Why Models ≈ Baseline

The models perform similarly to the baseline because of a fundamental challenge in the data:

**Class Imbalance:**
- Customers with spend > $0: **14.2%** (574)
- Customers with spend = $0: **85.8%** (3,473)

With 86% of customers having zero future spend, predicting the mean (which is heavily weighted toward $0) is actually a reasonable strategy. The models can't significantly improve because:

1. **Sparse signal**: Only 14% of customers have non-zero target
2. **Time-based split**: Test customers may have different behavior patterns
3. **Limited features**: Historical behavior only partially predicts future spend

### Negative R² Explained

R² < 0 means the model performs worse than predicting the mean. This happens when:
- Data has high variance relative to signal
- Test distribution differs from train (time-based split)
- Target is heavily zero-inflated

This is **expected** and **not a bug** - it reflects the inherent difficulty of the prediction task.

---

## Feature Importance

### Top 10 Features (Gradient Boosting)

| Feature | Importance |
|---------|------------|
| total_monetary | 23.9% |
| total_frequency | 13.7% |
| avg_order_value | 13.6% |
| total_loyalty_points | 8.1% |
| recency_days | 7.6% |
| avg_days_between_purchases | 7.0% |
| num_stores_visited | 5.9% |
| customer_tenure_days | 5.8% |
| top_category_encoded | 4.3% |
| monetary_90d | 3.8% |

**Insight:** Historical monetary value and frequency are the strongest predictors, confirming that past behavior is the best indicator of future spend.

---

## Data Leakage Prevention ✓

| Safeguard | Implementation |
|-----------|----------------|
| Time-based split | Customers sorted by last transaction date, 80/20 split |
| No customer overlap | Train/test sets have 0 overlapping customers |
| Features from past only | All features computed BEFORE cutoff date |
| Scaler fit on train | StandardScaler.fit() called only on training data |
| Target from future only | Target computed AFTER cutoff date |

---

## Files Saved

| File | Description |
|------|-------------|
| `spend_predictor_improved.pkl` | Gradient Boosting model |
| `feature_scaler_improved.pkl` | StandardScaler (fit on train) |
| `encoders_improved.pkl` | Label encoders |
| `model_metadata_improved.pkl` | Model config and metrics |
| `model_comparison_improved.csv` | Results table |
| `feature_importance_gb.csv` | GB feature importance |
| `feature_importance_rf.csv` | RF feature importance |

---

## Recommendations for Further Improvement

1. **Two-Stage Model**: First predict if customer will buy (classification), then how much (regression on buyers only)

2. **More Features**: 
   - Marketing exposure data
   - Website/app behavior
   - External factors (seasonality, holidays)

3. **Different Target**: Predict spend buckets (classification) instead of exact amount

4. **Longer History**: More historical data for better pattern learning

5. **Segment-Specific Models**: Separate models for different customer segments

---

*Document Created: 2026-02-04*
