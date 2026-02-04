# High-Level Architecture

## 1. Tech Stack Definition

### Core Language
- **Python 3.10+** - Primary development language

### Data Processing & Analysis
| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0 | Data manipulation, feature engineering |
| `numpy` | ≥1.24 | Numerical computations |
| `matplotlib` | ≥3.7 | Data visualization |
| `seaborn` | ≥0.12 | Statistical visualizations |

### Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.3 | Model training, evaluation, preprocessing |
| `xgboost` | ≥2.0 | Gradient boosting models |
| `shap` | ≥0.42 | Model interpretability (optional) |

### Model Persistence & Deployment
| Library | Version | Purpose |
|---------|---------|---------|
| `joblib` | ≥1.3 | Model serialization |
| `streamlit` | ≥1.28 | Web UI framework |

### Development Tools
| Tool | Purpose |
|------|---------|
| `jupyter` | Interactive development & EDA |
| `pytest` | Unit testing |
| `black` | Code formatting |

### Requirements File
```
# requirements.txt
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=2.0
shap>=0.42
joblib>=1.3
streamlit>=1.28
jupyter>=1.0
pytest>=7.0
```

---

## 2. UI-to-Model Integration Design

### Approach: Local Function Call via Streamlit

The model will be integrated using a **local function call** pattern, where Streamlit directly imports and calls the inference function. This is suitable for our project scope (demo/prototype).

### Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  User Input Form                                     │   │
│  │  - Customer ID (dropdown/text)                       │   │
│  │  - OR Manual feature entry                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  inference.py (Local Import)                         │   │
│  │  - predict_spend(customer_features) → float          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Prediction Display                                  │   │
│  │  - Predicted 30-day spend: $XXX.XX                   │   │
│  │  - Confidence indicator (optional)                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Inference Function Interface

```python
# src/inference.py

def load_model(model_path: str = "models/final_model.joblib"):
    """Load the trained model from disk."""
    import joblib
    return joblib.load(model_path)

def predict_spend(features: dict, model=None) -> float:
    """
    Predict 30-day customer spend.
    
    Parameters:
    -----------
    features : dict
        Customer features including:
        - recency_days: int (days since last purchase)
        - frequency_30d: int (purchases in last 30 days)
        - frequency_90d: int (purchases in last 90 days)
        - monetary_30d: float (total spend in last 30 days)
        - monetary_90d: float (total spend in last 90 days)
        - avg_order_value: float
        - loyalty_status: str ('Bronze', 'Silver', 'Gold', 'Platinum')
        - customer_tenure_days: int
        - unique_categories: int
        - unique_stores: int
    
    model : sklearn model, optional
        Pre-loaded model. If None, loads from default path.
    
    Returns:
    --------
    float : Predicted spend amount for next 30 days
    """
    import pandas as pd
    
    if model is None:
        model = load_model()
    
    # Convert features to DataFrame
    df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    return max(0, prediction)  # Ensure non-negative spend
```

### Why Local Function Call (vs REST API)?

| Aspect | Local Function Call | REST API |
|--------|---------------------|----------|
| Complexity | ✅ Simple | More setup required |
| Latency | ✅ Minimal | Network overhead |
| Deployment | Single process | Multiple services |
| Scalability | Limited | Better for production |
| **Best For** | ✅ Demos/Prototypes | Production systems |

For this 6-hour project, local function call is the pragmatic choice. The architecture can be extended to REST API later if needed.

---

## 3. High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   stores     │  │   products   │  │  customers   │  │ promotions   │    │
│  │    .csv      │  │    .csv      │  │    .csv      │  │    .csv      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│  ┌──────┴─────────────────┴─────────────────┴─────────────────┴──────┐     │
│  │                    store_sales_header.csv                          │     │
│  │                    store_sales_line_items.csv                      │     │
│  └────────────────────────────┬───────────────────────────────────────┘     │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                                    │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌────────────────┐  │
│  │   Data Cleaning     │───▶│ Feature Engineering │───▶│  Train/Test    │  │
│  │   - Missing values  │    │   - RFM features    │    │    Split       │  │
│  │   - Type casting    │    │   - Aggregations    │    │  (by cutoff)   │  │
│  │   - Outliers        │    │   - Encoding        │    │                │  │
│  └─────────────────────┘    └─────────────────────┘    └───────┬────────┘  │
│                                                                 │          │
│                                                                 ▼          │
│                           ┌─────────────────────────────────────────────┐  │
│                           │           Customer-Level Dataset            │  │
│                           │  ┌───────────────────────────────────────┐  │  │
│                           │  │ customer_id | features (X) | target(y)│  │  │
│                           │  │ C00001      | [...]        | $150.00  │  │  │
│                           │  │ C00002      | [...]        | $0.00    │  │  │
│                           │  │ ...         | ...          | ...      │  │  │
│                           │  └───────────────────────────────────────┘  │  │
│                           └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODELING LAYER                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Model Training Pipeline                       │   │
│  │                                                                      │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐    │   │
│  │   │ Baseline │   │  Linear  │   │  Random  │   │   XGBoost    │    │   │
│  │   │  (Mean)  │   │  Ridge   │   │  Forest  │   │              │    │   │
│  │   └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘    │   │
│  │        │              │              │                │            │   │
│  │        └──────────────┴──────────────┴────────────────┘            │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │                   ┌──────────────────────┐                         │   │
│  │                   │   Model Evaluation   │                         │   │
│  │                   │   MAE, RMSE, R²      │                         │   │
│  │                   └──────────┬───────────┘                         │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │                   ┌──────────────────────┐                         │   │
│  │                   │    Best Model        │                         │   │
│  │                   │  final_model.joblib  │                         │   │
│  │                   └──────────────────────┘                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Streamlit UI                                 │   │
│  │  ┌───────────────────┐         ┌─────────────────────────────────┐  │   │
│  │  │    Input Form     │         │       Prediction Output         │  │   │
│  │  │  - Customer ID    │────────▶│  - Predicted 30-day Spend       │  │   │
│  │  │  - OR Features    │         │  - Feature Importance (opt)     │  │   │
│  │  └───────────────────┘         └─────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                       │
│                  │  inference.py   │                                       │
│                  │ predict_spend() │                                       │
│                  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Layer | Components | Responsibility |
|-------|------------|----------------|
| **Data** | CSV files | Raw data storage |
| **Processing** | Scripts/Notebooks | Clean, transform, engineer features |
| **Modeling** | Training pipeline | Train, evaluate, select best model |
| **Application** | Streamlit + Inference | User interaction, predictions |

---

## 4. Cutoff Date Logic for Train/Test Split

### Overview

Unlike traditional random train/test splits, time-series customer data requires a **temporal split** to prevent data leakage and simulate real-world prediction scenarios.

### Cutoff Date Strategy

```
Data Timeline (based on our dataset: 2023-01-01 to 2025-12-31)
═══════════════════════════════════════════════════════════════════════════════

│◄─────────────── Historical Data ───────────────►│◄── Future Window ──►│
│                                                  │                      │
│                  TRAINING SET                    │      TEST SET        │
│         (compute features & targets)             │  (evaluate model)    │
│                                                  │                      │
├──────────────────────────────────────────────────┼──────────────────────┤
2023-01-01                                    CUTOFF DATE           +30 days

```

### Recommended Cutoff Date: 2025-01-01

Based on the transaction data spanning 2023-01-01 to 2025-12-31:

| Split | Date Range | Purpose |
|-------|------------|---------|
| **Training Period** | 2023-01-01 to 2024-12-31 | Build features from historical behavior |
| **Cutoff Date** | 2025-01-01 | Point of prediction |
| **Test Period** | 2025-01-01 to 2025-01-31 | 30-day window for target calculation |

### Implementation Logic

```python
# config.py
CUTOFF_DATE = "2025-01-01"
PREDICTION_HORIZON_DAYS = 30

# Feature computation uses data BEFORE cutoff
# Target computation uses data AFTER cutoff (within 30 days)
```

```python
# feature_engineering.py
import pandas as pd
from datetime import timedelta

def create_customer_dataset(transactions_df, cutoff_date="2025-01-01"):
    """
    Create customer-level dataset with features and target.
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Transaction data with columns: customer_id, transaction_date, total_amount
    cutoff_date : str
        Date string for train/test split (format: YYYY-MM-DD)
    
    Returns:
    --------
    DataFrame : Customer-level dataset with features (X) and target (y)
    """
    cutoff = pd.to_datetime(cutoff_date)
    future_end = cutoff + timedelta(days=30)
    
    # Split data temporally
    historical = transactions_df[transactions_df['transaction_date'] < cutoff]
    future = transactions_df[
        (transactions_df['transaction_date'] >= cutoff) & 
        (transactions_df['transaction_date'] < future_end)
    ]
    
    # ═══════════════════════════════════════════════════════════
    # FEATURES (X) - Computed from HISTORICAL data only
    # ═══════════════════════════════════════════════════════════
    
    features = historical.groupby('customer_id').agg(
        # Recency: days since last purchase before cutoff
        recency_days=('transaction_date', lambda x: (cutoff - x.max()).days),
        
        # Frequency: number of transactions
        frequency_total=('transaction_id', 'count'),
        
        # Monetary: total and average spend
        monetary_total=('total_amount', 'sum'),
        monetary_avg=('total_amount', 'mean'),
        
        # Additional features
        first_purchase=('transaction_date', 'min'),
        unique_stores=('store_id', 'nunique')
    ).reset_index()
    
    # Customer tenure
    features['tenure_days'] = (cutoff - features['first_purchase']).dt.days
    features = features.drop('first_purchase', axis=1)
    
    # ═══════════════════════════════════════════════════════════
    # TARGET (y) - Computed from FUTURE data only
    # ═══════════════════════════════════════════════════════════
    
    target = future.groupby('customer_id').agg(
        future_spend_30d=('total_amount', 'sum')
    ).reset_index()
    
    # Merge features and target
    customer_df = features.merge(target, on='customer_id', how='left')
    
    # Customers with no future purchases have $0 spend
    customer_df['future_spend_30d'] = customer_df['future_spend_30d'].fillna(0)
    
    return customer_df
```

### Data Leakage Prevention Checklist

| ✅ Check | Description |
|----------|-------------|
| ✅ | Features computed only from data **before** cutoff date |
| ✅ | Target computed only from data **after** cutoff date |
| ✅ | No customer attributes that could leak future info |
| ✅ | Time-based features use relative dates (days since X) |
| ✅ | Model never sees test period data during training |

### Visual Timeline

```
Customer C00001 Example:
═══════════════════════════════════════════════════════════════════════════════

Past Transactions (for Features):          Future Transactions (for Target):
─────────────────────────────────          ──────────────────────────────────

 $50      $75      $100     $30                    $80      $45
  │        │        │        │                      │        │
  ▼        ▼        ▼        ▼                      ▼        ▼
──┴────────┴────────┴────────┴──────────────────────┴────────┴─────────────▶
  │                          │           │                   │            time
2024-06   2024-08   2024-10  2024-12    2025-01-01         2025-01-31
                                          │                   │
                                       CUTOFF              END OF
                                        DATE              PREDICTION
                                                           WINDOW

Features (X):                           Target (y):
├─ recency_days = 15                    └─ future_spend_30d = $125
├─ frequency_total = 4                      ($80 + $45)
├─ monetary_total = $255
├─ monetary_avg = $63.75
└─ ...
```

---

## 5. Project Directory Structure

```
Predict-short-term-customer-spend/
│
├── data/
│   ├── raw/                    # Original CSV files
│   │   ├── customer_details.csv
│   │   ├── products.csv
│   │   ├── stores.csv
│   │   ├── store_sales_header.csv
│   │   ├── store_sales_line_items.csv
│   │   ├── promotion_details.csv
│   │   └── loyalty_rules.csv
│   │
│   └── processed/              # Cleaned & feature-engineered data
│       └── customer_features.csv
│
├── docs/
│   ├── 01_problem_understanding.md
│   ├── 02_dataset_choice.md
│   └── 03_high_level_architecture.md   # This document
│
├── models/
│   └── final_model.joblib      # Trained model artifact
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_modeling.ipynb       # Model training & evaluation
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants (cutoff date, paths)
│   ├── data_processing.py      # Data cleaning functions
│   ├── feature_engineering.py  # Feature creation functions
│   ├── train.py                # Model training script
│   └── inference.py            # Prediction function
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
├── tests/
│   └── test_inference.py       # Unit tests
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
└── TODO.md                     # Task tracking
```

---

## 6. Summary

| Component | Decision |
|-----------|----------|
| **Language** | Python 3.10+ |
| **ML Libraries** | scikit-learn, XGBoost |
| **UI Framework** | Streamlit |
| **Model Integration** | Local function call (via `inference.py`) |
| **Cutoff Date** | 2025-01-01 |
| **Prediction Horizon** | 30 days |
| **Model Persistence** | joblib |

This architecture balances simplicity for a 6-hour project while maintaining proper ML practices (temporal split, no data leakage, modular code structure).
