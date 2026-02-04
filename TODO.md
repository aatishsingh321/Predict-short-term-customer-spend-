# Predicting Short-Term Customer Spend (30-Day CLV) - Project TODO

## Project Overview
Build a regression model to predict how much a customer will spend in the next 30 days based on past behavior and characteristics.

**Type:** Supervised Learning - Regression  
**Target (y):** Total monetary amount the customer spends in the next 30 days after cutoff date  
**Total Time:** 6 hours (3 sprints)

---

## Sprint 1: System Design (1 hour)

### 5.1.1 Problem Understanding
- [x] Restate the problem in your own words
- [x] Define "short-term spend" (target variable = total spend in next 30 days)

### 5.1.2 Dataset Scouting & Choice
- [x] Identify 1-2 candidate retail/e-commerce datasets
- [x] Verify dataset has: customer_id, timestamps, transaction values, multiple purchases per customer
- [x] Choose final dataset with justification

### 5.1.3 High-Level Architecture
- [x] Define tech stack (Python, libraries: pandas, scikit-learn, etc.)
- [x] Design how UI will call the model (local API/function call)
- [x] Create high-level system diagram
- [x] Define cutoff date logic for train/test split

### Sprint 1 Deliverables
- [x] Written problem summary (business + ML)
- [x] Chosen dataset + justification
- [x] Defined horizon (30 days) and cutoff date logic
- [x] High-level system diagram

---

## Sprint 2: Data Selection, EDA and Model Training (4 hours)

### 5.2.1 Data Loading & Cleaning
- [x] Load transactions data
- [x] Handle missing values
- [x] Clean data types (dates, amounts)

### 5.2.2 Feature Engineering (Customer-Level)
- [x] Create RFM features (Recency, Frequency, Monetary)
- [x] Product/category preferences
- [x] Channel usage features
- [x] Customer attributes (region, segment, loyalty tier)

### 5.2.3 Define Target Variable
- [x] Calculate `future_spend_30d` for each customer
- [x] Create customer-level dataset (one row per customer)

### 5.2.4 EDA
- [x] Descriptive statistics
- [x] Distribution plots
- [x] Initial hypotheses

### Model Selection, Training, Evaluation

### 5.2.5 Data Splitting
- [ ] Train/test split based on cutoff date
- [ ] Validate no data leakage

### 5.2.6 Baseline Model
- [ ] Build simple baseline (e.g., mean/median predictor)

### 5.2.7 Model Selection & Training
- [ ] Try multiple regression models (Linear, Ridge, Random Forest, XGBoost)
- [ ] Hyperparameter tuning

### 5.2.8 Model Evaluation
- [ ] Calculate MAE, RMSE, R²
- [ ] Compare models

### 5.2.9 Interpretability (Optional)
- [ ] Feature importance analysis
- [ ] SHAP values if time permits

### 5.2.10 Model Packaging
- [ ] Save final trained model (pickle/joblib)
- [ ] Create inference function

### Sprint 2 Deliverables
- [ ] Customer-level dataset with features
- [ ] Trained and saved final model
- [ ] Comparison of baseline vs final model
- [ ] Short write-up of key insights
- [ ] Callable inference interface

---

## Sprint 3: Front-End UI and Endpoint Integration (1 hour)

### 5.3.1 Design UI Inputs
- [ ] Define input fields (customer features)
- [ ] Design UI layout

### 5.3.2 Implement UI
- [ ] Build simple UI (Streamlit/Gradio/Flask)
- [ ] Add input validation

### 5.3.3 Integrate UI with Model
- [ ] Connect UI to model inference function
- [ ] Display prediction results

### 5.3.4 Error Handling & UX
- [ ] Add error messages
- [ ] Loading states
- [ ] Clear result display

### Sprint 3 Deliverables
- [ ] Working UI that accepts customer features
- [ ] Returns model prediction for short-term spend

---

## Final Presentation

### 5.3.5 Problem Overview
- [ ] Prepare slides/presentation on problem statement

### 5.3.6 Data & Target
- [ ] Explain data source and target definition

### 5.3.7 Features & Modeling
- [ ] Document features used and modeling approach

### 5.3.8 System & Demo
- [ ] Prepare live demo of working system

### 5.3.9 Reflection
- [ ] Document learnings and trade-offs

### Presentation Deliverables
- [ ] Slide deck or structured presentation
- [ ] Live or recorded demo of running system

---

## Database Schema Reference

| Table | Key Fields |
|-------|------------|
| stores | store_id, store_name, store_city, store_region, opening_date |
| products | product_id, product_name, product_category, unit_price, current_stock_level |
| customer_details | customer_id, first_name, email, loyalty_status, total_loyalty_points, last_purchase_date, segment_id, customer_phone, customer_since |
| promotion_details | promotion_id, promotion_name, start_date, end_date, discount_percentage, applicable_category |
| loyalty_rules | rule_id, rule_name, points_per_unit_spend, min_spend_threshold, bonus_points |
| store_sales_header | transaction_id, customer_id, store_id, transaction_date, total_amount, customer_phone |
| store_sales_line_items | line_item_id, transaction_id, product_id, promotion_id, quantity, line_item_amount |

---

## Input Features (X)
- Historical transaction behavior (past spend, number of orders, recency)
- Product/category preferences
- Channel usage (online vs in-store, web vs mobile)
- Customer attributes (region, segment, loyalty tier)

## Success Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)

---

## Evaluation Criteria
1. **Problem Understanding** - Clear business value and ML framing
2. **Data & Modeling** - Sensible dataset, thoughtful features, proper train/test split
3. **Engineering & Integration** - Clean code, model packaging, functional UI
4. **Communication & Presentation** - Clear storyline: problem → data → model → results → demo
