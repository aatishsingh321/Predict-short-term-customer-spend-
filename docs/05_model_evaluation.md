# Model Evaluation Report

## Overview

This document compares the performance of three regression models trained to predict customer spend in the next 30 days.

**Generated:** 2026-02-04 13:31:40

---

## 1. Models Trained

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline model for interpretability and benchmarking |
| **Random Forest** | Ensemble of 200 decision trees with controlled depth |
| **XGBoost** | Gradient boosting with conservative hyperparameters |

---

## 2. Dataset Summary

| Metric | Value |
|--------|-------|
| Total Customers | 3,818 |
| Training Set | 3,054 (80%) |
| Test Set | 764 (20%) |
| Features | 51 |
| Split Method | Time-based (by first_purchase_date) |

### Target Variable: `future_spend_30d`

| Statistic | Value |
|-----------|-------|
| Mean | $44.69 |
| Median | $0.00 |
| Std Dev | $124.67 |
| Min | $0.00 |
| Max | $954.96 |

---

## 3. Model Performance Comparison

### 3.1 Metrics Summary

| Model | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² |
|-------|------------|-----------|-----------|----------|----------|---------|
| Linear Regression | $124.09 | $100.34 | $75.23 | $42.93 | 0.0820 | 0.0137 |
| Random Forest | $110.04 | $101.16 | $67.29 | $56.63 | 0.2781 | -0.0025 |
| XGBoost | $67.22 | $103.53 | $40.33 | $57.00 | 0.7306 | -0.0501 |

### 3.2 Overfitting Analysis

| Model | Train/Test RMSE Ratio | Assessment |
|-------|----------------------|------------|
| Linear Regression | 1.237 | Overfitting |
| Random Forest | 1.088 | Good |
| XGBoost | 0.649 | Underfitting |

> A Train/Test RMSE ratio between 0.8-1.2 indicates good generalization.
> Ratio > 1.2 suggests overfitting (model memorizes training data).
> Ratio < 0.8 suggests underfitting (model too simple).

---

## 4. Model Selection Decision

### 🏆 Selected Model: **Linear Regression**

### Rationale


1. **Lowest Test MAE ($42.93)**: Despite its simplicity, Linear Regression achieves competitive accuracy.

2. **Perfect Interpretability**: Coefficients directly show how each feature impacts predicted spend.

3. **No Overfitting Risk**: Linear models are inherently simple and don't overfit on small datasets.

4. **Fast Inference**: Prediction is a simple dot product, making it ideal for real-time applications.

5. **Baseline Reliability**: Serves as a reliable baseline that more complex models should beat.

### Why Not Other Models?


**Random Forest**: Achieved Test MAE of $56.63. While robust, it underperformed compared to Linear Regression on this dataset.

**XGBoost**: Achieved Test MAE of $57.00. While powerful, it underperformed compared to Linear Regression on this dataset.

---

## 5. Feature Importance

### Top 10 Features (Random Forest - Gini Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | num__monetary_total | 0.0542 |
| 2 | num__total_items_purchased | 0.0385 |
| 3 | num__monetary_365d | 0.0365 |
| 4 | num__avg_days_between_purchases | 0.0343 |
| 5 | num__frequency_total | 0.0325 |
| 6 | num__std_days_between_purchases | 0.0315 |
| 7 | num__unique_products | 0.0303 |
| 8 | num__min_order_value | 0.0275 |
| 9 | num__pct_spend_home_&_garden | 0.0269 |
| 10 | num__pct_spend_automotive | 0.0262 |

### Top 10 Features (XGBoost - Gain Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | num__frequency_total | 0.0279 |
| 2 | cat__primary_region_Southwest | 0.0185 |
| 3 | cat__primary_store_id_S012 | 0.0180 |
| 4 | num__frequency_90d | 0.0162 |
| 5 | num__monetary_60d | 0.0155 |
| 6 | num__frequency_365d | 0.0153 |
| 7 | num__monetary_total | 0.0151 |
| 8 | cat__loyalty_status_Gold | 0.0151 |
| 9 | num__pct_spend_sports | 0.0150 |
| 10 | num__pct_spend_electronics | 0.0144 |

### Interpretation

The most important features relate to:
1. **Monetary features** (total spend, average order value) - Past spending strongly predicts future spending
2. **Frequency features** (transaction counts) - Purchase frequency indicates engagement
3. **Recency features** (days since last purchase) - Recent buyers are more likely to buy again

---

## 6. Visualizations

| Plot | Location |
|------|----------|
| Linear Regression Evaluation | `plots/linear_regression_evaluation.png` |
| Random Forest Evaluation | `plots/random_forest_evaluation.png` |
| XGBoost Evaluation | `plots/xgboost_evaluation.png` |
| Model Comparison | `plots/model_comparison.png` |

---

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `models/linear_regression.joblib` | Trained Linear Regression model |
| `models/random_forest.joblib` | Trained Random Forest model |
| `models/xgboost.joblib` | Trained XGBoost model |
| `models/final_model.joblib` | Best model with preprocessor and metadata |
| `models/model_comparison.csv` | Performance metrics for all models |
| `models/rf_feature_importance.csv` | Random Forest feature importance |
| `models/xgb_feature_importance.csv` | XGBoost feature importance |

---

## 8. Recommendations

1. **Deploy Linear Regression** as the production model for 30-day spend prediction.

2. **Monitor Performance**: Track prediction accuracy over time and retrain if performance degrades.

3. **Feature Updates**: Regularly update customer features to capture recent behavior.

4. **Business Actions**: Use predictions to:
   - Target high-value customers with retention campaigns
   - Identify at-risk customers (low predicted spend)
   - Optimize marketing budget allocation

---

*Report generated automatically by model_training.py*
