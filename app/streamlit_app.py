"""
================================================================================
CUSTOMER SPEND PREDICTOR - STREAMLIT UI
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Web interface for predicting customer 30-day spend

Run with: streamlit run app/streamlit_app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# CONFIGURATION
# ============================================================
MODELS_DIR = "/Users/apple/Customer Spend Predictor/models"
DATA_DIR = "/Users/apple/Customer Spend Predictor/data/cleaned"

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Spend Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================
@st.cache_resource
def load_model_artifacts():
    """Load trained model and all required artifacts"""
    try:
        with open(f"{MODELS_DIR}/spend_predictor.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open(f"{MODELS_DIR}/feature_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        with open(f"{MODELS_DIR}/encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
        
        with open(f"{MODELS_DIR}/feature_columns.pkl", 'rb') as f:
            feature_columns = pickle.load(f)
        
        with open(f"{MODELS_DIR}/model_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, encoders, feature_columns, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

@st.cache_data
def load_customer_data():
    """Load customer data for lookup"""
    try:
        customers = pd.read_csv(f"{DATA_DIR}/customer_features_full.csv")
        return customers
    except:
        return None

# Load artifacts
model, scaler, encoders, feature_columns, metadata = load_model_artifacts()
customer_data = load_customer_data()

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================
def validate_inputs(inputs):
    """Validate all input fields"""
    errors = []
    
    if inputs['recency_days'] < 0:
        errors.append("Recency cannot be negative")
    if inputs['recency_days'] > 1000:
        errors.append("Recency seems too high (>1000 days)")
    
    for window in [30, 60, 90]:
        if inputs[f'frequency_{window}d'] < 0:
            errors.append(f"Frequency ({window}d) cannot be negative")
        if inputs[f'monetary_{window}d'] < 0:
            errors.append(f"Monetary ({window}d) cannot be negative")
    
    if inputs['avg_order_value'] < 0:
        errors.append("Average order value cannot be negative")
    if inputs['avg_order_value'] > 10000:
        errors.append("Average order value seems too high (>$10,000)")
    
    if inputs['total_loyalty_points'] < 0:
        errors.append("Loyalty points cannot be negative")
    
    if inputs['num_stores_visited'] < 1:
        errors.append("Number of stores must be at least 1")
    if inputs['num_stores_visited'] > 15:
        errors.append("Number of stores cannot exceed 15")
    
    return errors

def prepare_features(inputs, encoders):
    """Prepare feature vector from inputs"""
    # Encode categorical variables
    loyalty_encoded = encoders['loyalty_encoder'].transform([inputs['loyalty_status']])[0]
    segment_encoded = encoders['segment_encoder'].transform([inputs['segment_id']])[0]
    
    # Handle unknown category
    try:
        category_encoded = encoders['category_encoder'].transform([inputs['top_category']])[0]
    except:
        category_encoded = 0  # Default to first category
    
    # Create feature vector in correct order
    features = [
        inputs['recency_days'],
        inputs['frequency_30d'],
        inputs['frequency_60d'],
        inputs['frequency_90d'],
        inputs['monetary_30d'],
        inputs['monetary_60d'],
        inputs['monetary_90d'],
        inputs['total_frequency'],
        inputs['total_monetary'],
        inputs['avg_order_value'],
        inputs['num_stores_visited'],
        inputs['avg_days_between_purchases'],
        inputs['total_loyalty_points'],
        inputs['customer_tenure_days'],
        inputs['num_categories'],
        inputs['is_weekend_shopper'],
        loyalty_encoded,
        segment_encoded,
        category_encoded
    ]
    
    return np.array(features).reshape(1, -1)

def get_spend_segment(prediction):
    """Classify customer based on predicted spend"""
    if prediction == 0:
        return "Inactive", "🔴", "#FF5252"
    elif prediction < 100:
        return "Low Value", "🟠", "#FF9800"
    elif prediction < 300:
        return "Medium Value", "🟡", "#FFEB3B"
    elif prediction < 500:
        return "Medium-High Value", "🟢", "#8BC34A"
    else:
        return "High Value", "⭐", "#4CAF50"

# ============================================================
# MAIN UI
# ============================================================

# Header
st.markdown('<p class="main-header">🛒 Customer Spend Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict 30-Day Customer Spend Using Machine Learning</p>', unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.error("⚠️ Model not loaded. Please run train_model.py first.")
    st.stop()

# Sidebar - Model Info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"""
    **Model Type:** {metadata.get('model_type', 'N/A')}  
    **Training Samples:** {metadata.get('training_samples', 'N/A'):,}  
    **Prediction Window:** {metadata.get('prediction_window', 30)} days
    
    **Performance Metrics:**
    - MAE: ${metadata.get('mae', 0):.2f}
    - RMSE: ${metadata.get('rmse', 0):.2f}
    - R²: {metadata.get('r2', 0):.4f}
    """)
    
    st.divider()
    st.header("📊 Quick Stats")
    if customer_data is not None:
        st.metric("Total Customers", f"{len(customer_data):,}")
        st.metric("Avg Historical Spend", f"${customer_data['total_monetary'].mean():,.2f}")

# Main content
col1, col2 = st.columns([1, 1])

# ============================================================
# LEFT COLUMN - INPUT FIELDS
# ============================================================
with col1:
    st.header("📝 Customer Features")
    
    # Customer Lookup (Optional)
    with st.expander("🔍 Quick Lookup (Optional)", expanded=False):
        if customer_data is not None:
            customer_ids = ['-- Select Customer --'] + customer_data['customer_id'].tolist()
            selected_customer = st.selectbox("Select Existing Customer", customer_ids)
            
            if selected_customer != '-- Select Customer --':
                cust_row = customer_data[customer_data['customer_id'] == selected_customer].iloc[0]
                st.success(f"✓ Loaded data for {selected_customer}")
        else:
            selected_customer = None
            st.info("Customer data not available for lookup")
    
    st.divider()
    
    # Get default values (from selected customer or defaults)
    if 'selected_customer' in dir() and selected_customer and selected_customer != '-- Select Customer --':
        defaults = customer_data[customer_data['customer_id'] == selected_customer].iloc[0].to_dict()
    else:
        defaults = {
            'recency_days': 30,
            'frequency_30d': 2,
            'frequency_60d': 4,
            'frequency_90d': 6,
            'monetary_30d': 200.0,
            'monetary_60d': 400.0,
            'monetary_90d': 600.0,
            'total_frequency': 10,
            'total_monetary': 1000.0,
            'avg_order_value': 100.0,
            'num_stores_visited': 2,
            'avg_days_between_purchases': 15.0,
            'loyalty_status': 'Bronze',
            'total_loyalty_points': 100.0,
            'segment_id': 'NR',
            'customer_tenure_days': 365.0,
            'num_categories': 3,
            'top_category': 'Electronics',
            'is_weekend_shopper': 0
        }
    
    # RFM Features
    st.subheader("📊 RFM Features")
    
    col_rfm1, col_rfm2 = st.columns(2)
    
    with col_rfm1:
        recency_days = st.slider(
            "Recency (Days Since Last Purchase)",
            min_value=0, max_value=365, 
            value=int(defaults.get('recency_days', 30)),
            help="Number of days since the customer's last purchase"
        )
        
        frequency_30d = st.number_input(
            "Transactions (Last 30 Days)",
            min_value=0, max_value=50,
            value=int(defaults.get('frequency_30d', 2)),
            help="Number of transactions in the last 30 days"
        )
        
        frequency_60d = st.number_input(
            "Transactions (Last 60 Days)",
            min_value=0, max_value=100,
            value=int(defaults.get('frequency_60d', 4))
        )
        
        frequency_90d = st.number_input(
            "Transactions (Last 90 Days)",
            min_value=0, max_value=150,
            value=int(defaults.get('frequency_90d', 6))
        )
    
    with col_rfm2:
        monetary_30d = st.number_input(
            "Spend Last 30 Days ($)",
            min_value=0.0, max_value=10000.0,
            value=float(defaults.get('monetary_30d', 200.0)),
            step=10.0
        )
        
        monetary_60d = st.number_input(
            "Spend Last 60 Days ($)",
            min_value=0.0, max_value=20000.0,
            value=float(defaults.get('monetary_60d', 400.0)),
            step=10.0
        )
        
        monetary_90d = st.number_input(
            "Spend Last 90 Days ($)",
            min_value=0.0, max_value=30000.0,
            value=float(defaults.get('monetary_90d', 600.0)),
            step=10.0
        )
        
        total_frequency = st.number_input(
            "Total Transactions (All Time)",
            min_value=1, max_value=500,
            value=int(defaults.get('total_frequency', 10))
        )
    
    total_monetary = st.number_input(
        "Total Historical Spend ($)",
        min_value=0.0, max_value=50000.0,
        value=float(defaults.get('total_monetary', 1000.0)),
        step=50.0
    )
    
    st.divider()
    
    # Customer Attributes
    st.subheader("👤 Customer Attributes")
    
    col_attr1, col_attr2 = st.columns(2)
    
    with col_attr1:
        loyalty_status = st.selectbox(
            "Loyalty Status",
            options=metadata.get('loyalty_classes', ['Bronze', 'Silver', 'Gold', 'Platinum']),
            index=0 if defaults.get('loyalty_status', 'Bronze') not in metadata.get('loyalty_classes', []) 
                  else metadata.get('loyalty_classes', []).index(defaults.get('loyalty_status', 'Bronze'))
        )
        
        total_loyalty_points = st.number_input(
            "Total Loyalty Points",
            min_value=0, max_value=50000,
            value=int(defaults.get('total_loyalty_points', 100))
        )
        
        segment_options = metadata.get('segment_classes', ['AR', 'HC', 'HS', 'LP', 'NR'])
        segment_id = st.selectbox(
            "Customer Segment",
            options=segment_options,
            index=segment_options.index(defaults.get('segment_id', 'NR')) if defaults.get('segment_id', 'NR') in segment_options else 0,
            help="AR=At Risk, HC=High Contact, HS=High Spender, LP=Lapsed, NR=New/Regular"
        )
    
    with col_attr2:
        customer_tenure_days = st.slider(
            "Customer Tenure (Days)",
            min_value=0, max_value=2000,
            value=int(defaults.get('customer_tenure_days', 365)),
            help="Days since customer's first purchase"
        )
        
        avg_order_value = st.number_input(
            "Avg Order Value ($)",
            min_value=0.0, max_value=1000.0,
            value=float(defaults.get('avg_order_value', 100.0)),
            step=5.0
        )
        
        num_stores_visited = st.slider(
            "Stores Visited",
            min_value=1, max_value=15,
            value=int(defaults.get('num_stores_visited', 2))
        )
    
    st.divider()
    
    # Shopping Behavior
    st.subheader("🛍️ Shopping Behavior")
    
    col_shop1, col_shop2 = st.columns(2)
    
    with col_shop1:
        avg_days_between = st.number_input(
            "Avg Days Between Purchases",
            min_value=1.0, max_value=365.0,
            value=float(defaults.get('avg_days_between_purchases', 15.0)),
            step=1.0
        )
        
        num_categories = st.slider(
            "Categories Purchased",
            min_value=1, max_value=11,
            value=int(defaults.get('num_categories', 3))
        )
    
    with col_shop2:
        category_options = metadata.get('category_classes', ['Electronics', 'Apparel', 'Home & Garden'])
        default_cat = defaults.get('top_category', 'Electronics')
        top_category = st.selectbox(
            "Top Category",
            options=category_options,
            index=category_options.index(default_cat) if default_cat in category_options else 0
        )
        
        is_weekend_shopper = st.checkbox(
            "Weekend Shopper",
            value=bool(defaults.get('is_weekend_shopper', 0))
        )

# ============================================================
# RIGHT COLUMN - PREDICTION RESULTS
# ============================================================
with col2:
    st.header("🔮 Prediction")
    
    # Collect all inputs
    inputs = {
        'recency_days': recency_days,
        'frequency_30d': frequency_30d,
        'frequency_60d': frequency_60d,
        'frequency_90d': frequency_90d,
        'monetary_30d': monetary_30d,
        'monetary_60d': monetary_60d,
        'monetary_90d': monetary_90d,
        'total_frequency': total_frequency,
        'total_monetary': total_monetary,
        'avg_order_value': avg_order_value,
        'num_stores_visited': num_stores_visited,
        'avg_days_between_purchases': avg_days_between,
        'loyalty_status': loyalty_status,
        'total_loyalty_points': total_loyalty_points,
        'segment_id': segment_id,
        'customer_tenure_days': customer_tenure_days,
        'num_categories': num_categories,
        'top_category': top_category,
        'is_weekend_shopper': int(is_weekend_shopper)
    }
    
    # Predict button
    if st.button("🔮 PREDICT 30-DAY SPEND", use_container_width=True):
        # Validate inputs
        errors = validate_inputs(inputs)
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            with st.spinner("Calculating prediction..."):
                try:
                    # Prepare features
                    features = prepare_features(inputs, encoders)
                    
                    # Make prediction
                    prediction = model.predict(features)[0]
                    
                    # Ensure non-negative
                    prediction = max(0, prediction)
                    
                    # Get segment
                    segment, icon, color = get_spend_segment(prediction)
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="font-size: 1.2rem; color: #666;">Predicted 30-Day Spend</p>
                        <p class="prediction-value">${prediction:,.2f}</p>
                        <p style="font-size: 1.5rem;">{icon} {segment}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.divider()
                    st.subheader("📊 Prediction Insights")
                    
                    col_ins1, col_ins2, col_ins3 = st.columns(3)
                    
                    with col_ins1:
                        st.metric(
                            "Daily Average",
                            f"${prediction/30:.2f}",
                            help="Average daily spend over 30 days"
                        )
                    
                    with col_ins2:
                        weekly = prediction / 4.3
                        st.metric(
                            "Weekly Average",
                            f"${weekly:.2f}",
                            help="Average weekly spend"
                        )
                    
                    with col_ins3:
                        # Compare to average
                        if customer_data is not None:
                            avg_future = customer_data['future_spend_30d'].mean()
                            diff = ((prediction - avg_future) / avg_future * 100) if avg_future > 0 else 0
                            st.metric(
                                "vs. Average",
                                f"${prediction:.2f}",
                                f"{diff:+.1f}%",
                                help="Compared to average customer"
                            )
                    
                    # Recommendation
                    st.divider()
                    st.subheader("💡 Recommendation")
                    
                    if segment == "High Value":
                        st.success("🌟 **VIP Customer** - Consider exclusive offers and priority support")
                    elif segment == "Medium-High Value":
                        st.success("📈 **Growing Customer** - Encourage with loyalty rewards")
                    elif segment == "Medium Value":
                        st.info("🎯 **Potential Customer** - Target with personalized campaigns")
                    elif segment == "Low Value":
                        st.warning("⚠️ **At-Risk Customer** - Consider re-engagement campaign")
                    else:
                        st.error("🔴 **Inactive Customer** - Needs win-back strategy")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    else:
        # Show placeholder
        st.info("👆 Enter customer features and click **PREDICT** to see the 30-day spend prediction")
        
        # Show sample prediction
        st.divider()
        st.subheader("📋 Input Summary")
        
        summary_df = pd.DataFrame({
            'Feature': ['Recency', 'Frequency (30d)', 'Monetary (30d)', 'Loyalty', 'Tenure'],
            'Value': [
                f"{recency_days} days",
                f"{frequency_30d} transactions",
                f"${monetary_30d:,.2f}",
                loyalty_status,
                f"{customer_tenure_days} days"
            ]
        })
        st.table(summary_df)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>Customer Spend Predictor v1.0 | Built with Streamlit & scikit-learn</p>
    <p>Model: Gradient Boosting Regressor | Prediction Window: 30 Days</p>
</div>
""", unsafe_allow_html=True)
