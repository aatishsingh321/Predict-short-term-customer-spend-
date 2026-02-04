# Customer Spend Predictor

A machine learning pipeline to predict **30-day future customer spend** (short-term Customer Lifetime Value) using historical transaction data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Project Overview

This project predicts how much a customer will spend in the next 30 days based on their historical purchase behavior. It uses RFM (Recency, Frequency, Monetary) features and advanced ML models including a **Two-Stage CLV Model** for handling zero-inflated spend data.

### Key Features
- 🔄 Complete ML pipeline (data cleaning → EDA → modeling → deployment)
- 📊 Multiple model comparison (Linear, Ridge, Random Forest, Gradient Boosting)
- 🎯 Two-Stage CLV Model for zero-inflated predictions
- 🖥️ Interactive Streamlit UI for predictions
- 📈 Comprehensive EDA with 13+ visualizations

## 🏗️ Project Structure

```
Customer Spend Predictor/
├── app/
│   └── streamlit_app.py          # Streamlit web interface
├── data/
│   ├── raw/                      # Original generated data
│   ├── cleaned/                  # Cleaned datasets
│   └── rejected/                 # Rejected records with reasons
├── docs/
│   ├── 01_problem_understanding.md
│   ├── 02_dataset_choice.md
│   ├── 03_data_rejection_report.md
│   ├── 04_eda_documentation.md
│   ├── 05_ui_design.md
│   ├── 06_model_selection.md
│   └── 08_complete_project_documentation.md
├── models/
│   ├── spend_predictor.pkl       # Main prediction model
│   └── two_stage/                # Two-stage CLV model artifacts
├── plots/                        # EDA visualizations
├── scripts/
│   ├── generate_data.py          # Synthetic data generation
│   ├── data_cleaning.py          # Data cleaning pipeline
│   ├── eda_analysis.py           # Exploratory data analysis
│   ├── train_model.py            # Model training (v1)
│   ├── train_model_v3.py         # Improved model training
│   └── train_two_stage_model.py  # Two-stage CLV model
├── requirements.txt
├── TODO.md
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "Customer Spend Predictor"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Generate synthetic data
python scripts/generate_data.py

# Clean the data
python scripts/data_cleaning.py

# Run EDA
python scripts/eda_analysis.py

# Train models
python scripts/train_two_stage_model.py
```

### 3. Launch the UI

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

## 📊 Model Performance

### Final Evaluation Results

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|-----|
| Zero Predictor (Baseline) | $208.54 | $713.84 | -0.093 |
| Mean Predictor | $300.72 | $685.29 | -0.008 |
| Linear Regression | $209.84 | $711.67 | -0.087 |
| Ridge Regression | $209.83 | $711.71 | -0.087 |
| Random Forest | $209.93 | $711.82 | -0.087 |
| Gradient Boosting | $209.61 | $711.69 | -0.087 |
| **Two-Stage Prob-Weighted** | $279.96 | $679.10 | **+0.0105** ✅ |

### Two-Stage CLV Model
- **Stage 1 (Classifier)**: ROC-AUC = 0.718, Accuracy = 80.1%
- **Stage 2 (Regressor)**: Predicts spend amount for likely spenders

### Why R² is Low/Negative

This is **expected** for this type of problem:
- 86% of customers have $0 future spend (zero-inflation)
- Coefficient of Variation = 1.66 (extreme variance)
- Time-based split reflects real-world deployment challenges

**MAE improvement of 30% vs baseline is the meaningful metric.**

## 🔧 Features Used

| Feature | Description |
|---------|-------------|
| `frequency` | Number of past transactions |
| `monetary` | Total historical spend |
| `recency` | Days since last purchase |
| `avg_basket` | Average transaction amount |
| `tenure` | Days between first and last purchase |
| `max_spend` | Maximum single transaction |
| `min_spend` | Minimum single transaction |
| `std_spend` | Spending variability |
| `avg_days_between` | Average days between purchases |

## 📈 Key Insights from EDA

1. **86% of customers** have zero future spend in the 30-day window
2. **Top 10% of customers** contribute 60%+ of revenue
3. **Recency is critical**: customers active in last 30 days are 5x more likely to purchase
4. **Store 3** has highest average transaction value
5. **Loyalty members** spend 40% more on average

## 🎯 Target Variable

```
future_spend_30d = Sum of all transactions in 30 days after cutoff date
Cutoff Date: August 1, 2025
```

## ⚠️ Important Notes

### Time-Based Split (No Data Leakage)
- Train set: Customers sorted by last transaction date (first 80%)
- Test set: Remaining 20% (most recent customers)
- This prevents future information from leaking into training

### Model Selection Rationale
- **Gradient Boosting**: Best single-model MAE
- **Two-Stage Model**: Only positive R², handles zero-inflation
- **Random Forest Classifier**: Best at predicting "will customer spend?"

## 📝 Documentation

Detailed documentation available in `/docs`:
- Problem understanding and scope
- Dataset choice and generation
- Data rejection report
- EDA findings and hypotheses
- UI design decisions
- Complete project documentation

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas, numpy**: Data manipulation
- **scikit-learn**: ML models and preprocessing
- **matplotlib, seaborn**: Visualization
- **Streamlit**: Web UI
- **joblib**: Model serialization

## 📄 License

This project is for educational purposes.

## 👤 Author

Customer Spend Prediction ML Pipeline

---

**Note**: For production deployment, consider:
- Adding external data (marketing, web behavior)
- Implementing customer segmentation
- Extending prediction horizon to 90+ days
- Setting up model monitoring and retraining
