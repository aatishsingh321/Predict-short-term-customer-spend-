# UI Design Document

## Customer Spend Prediction - Front-End Interface

### Overview
This document defines the input fields and layout for the Customer Spend Prediction UI. The interface will accept customer features and return a predicted 30-day spend amount.

---

## 1. Input Fields Definition

### 1.1 Primary Input Mode: Customer ID Lookup
For existing customers, users can simply enter a Customer ID to auto-populate features.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `customer_id` | Text (Dropdown/Search) | Optional | Select existing customer to auto-fill features |

---

### 1.2 Manual Feature Input Fields

For new customers or manual predictions, the following fields are required:

#### A. RFM Features (Core Predictors)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `recency_days` | Number (Slider) | 0-365 | 30 | Days since last purchase |
| `frequency_30d` | Number (Input) | 0-20 | 2 | Number of transactions in last 30 days |
| `frequency_60d` | Number (Input) | 0-40 | 4 | Number of transactions in last 60 days |
| `frequency_90d` | Number (Input) | 0-60 | 6 | Number of transactions in last 90 days |
| `monetary_30d` | Number (Currency) | 0-5000 | 200 | Total spend in last 30 days ($) |
| `monetary_60d` | Number (Currency) | 0-10000 | 400 | Total spend in last 60 days ($) |
| `monetary_90d` | Number (Currency) | 0-15000 | 600 | Total spend in last 90 days ($) |

#### B. Customer Attributes

| Field | Type | Options | Default | Description |
|-------|------|---------|---------|-------------|
| `loyalty_status` | Dropdown | Bronze, Silver, Gold, Platinum | Bronze | Customer loyalty tier |
| `total_loyalty_points` | Number (Input) | 0-20000 | 100 | Accumulated loyalty points |
| `customer_tenure_days` | Number (Slider) | 0-2000 | 365 | Days since first purchase |
| `segment_id` | Dropdown | HS, AR, NR, LP, HC | NR | Customer segment |

#### C. Transaction Behavior

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `avg_order_value` | Number (Currency) | 10-500 | 100 | Average transaction amount ($) |
| `avg_items_per_order` | Number (Input) | 1-20 | 3 | Average items per transaction |
| `num_stores_visited` | Number (Slider) | 1-15 | 2 | Unique stores shopped at |

#### D. Category Preferences

| Field | Type | Options | Default | Description |
|-------|------|---------|---------|-------------|
| `top_category` | Dropdown | Electronics, Apparel, Home & Garden, Sports, Beauty, Toys, Books, Grocery, Automotive, Jewelry | Electronics | Most purchased category |
| `num_categories` | Number (Slider) | 1-10 | 3 | Number of unique categories purchased |

#### E. Temporal Patterns (Optional/Advanced)

| Field | Type | Options | Default | Description |
|-------|------|---------|---------|-------------|
| `preferred_day` | Dropdown | Monday-Sunday | Saturday | Most common shopping day |
| `is_weekend_shopper` | Checkbox | True/False | False | Primarily shops on weekends |

---

## 2. UI Layout Design

### 2.1 Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🛒 CUSTOMER SPEND PREDICTOR                              │
│                    Predict 30-Day Customer Spend                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  📋 QUICK LOOKUP (Optional)                                          │   │
│  │  ┌──────────────────────────────────────────┐  ┌─────────────────┐  │   │
│  │  │ 🔍 Search Customer ID...                 │  │  Auto-Fill      │  │   │
│  │  └──────────────────────────────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ─────────────────────── OR ENTER MANUALLY ───────────────────────         │
│                                                                             │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐          │
│  │  📊 RFM FEATURES            │  │  👤 CUSTOMER ATTRIBUTES     │          │
│  │                             │  │                             │          │
│  │  Recency (days)             │  │  Loyalty Status             │          │
│  │  ├──────────●───────┤ 30    │  │  [Bronze     ▼]             │          │
│  │                             │  │                             │          │
│  │  Transactions (30d)         │  │  Loyalty Points             │          │
│  │  [    2    ]                │  │  [    100    ]              │          │
│  │                             │  │                             │          │
│  │  Transactions (60d)         │  │  Tenure (days)              │          │
│  │  [    4    ]                │  │  ├────●──────────┤ 365      │          │
│  │                             │  │                             │          │
│  │  Transactions (90d)         │  │  Segment                    │          │
│  │  [    6    ]                │  │  [NR          ▼]            │          │
│  │                             │  │                             │          │
│  │  Spend Last 30d ($)         │  └─────────────────────────────┘          │
│  │  [   200   ]                │                                           │
│  │                             │  ┌─────────────────────────────┐          │
│  │  Spend Last 60d ($)         │  │  🛍️ SHOPPING BEHAVIOR       │          │
│  │  [   400   ]                │  │                             │          │
│  │                             │  │  Avg Order Value ($)        │          │
│  │  Spend Last 90d ($)         │  │  [   100   ]                │          │
│  │  [   600   ]                │  │                             │          │
│  │                             │  │  Avg Items/Order            │          │
│  └─────────────────────────────┘  │  [    3    ]                │          │
│                                   │                             │          │
│  ┌─────────────────────────────┐  │  Stores Visited             │          │
│  │  📦 CATEGORY PREFERENCES    │  │  ├──●────────────┤ 2        │          │
│  │                             │  │                             │          │
│  │  Top Category               │  └─────────────────────────────┘          │
│  │  [Electronics  ▼]           │                                           │
│  │                             │                                           │
│  │  Categories Purchased       │                                           │
│  │  ├────●───────────┤ 3       │                                           │
│  │                             │                                           │
│  └─────────────────────────────┘                                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │                    [ 🔮 PREDICT SPEND ]                             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      📈 PREDICTION RESULT                           │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                                                               │  │   │
│  │  │     Predicted 30-Day Spend:  $XXX.XX                         │  │   │
│  │  │                                                               │  │   │
│  │  │     ████████████████░░░░░░░░  Confidence: 85%                │  │   │
│  │  │                                                               │  │   │
│  │  │     Prediction Range: $XXX - $XXX                            │  │   │
│  │  │                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  Customer Segment: [High Value] / [Medium Value] / [Low Value]     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Component Specifications

#### Header Section
- **Title:** "🛒 Customer Spend Predictor"
- **Subtitle:** "Predict 30-Day Customer Spend"
- **Style:** Centered, large font, branded colors

#### Quick Lookup Section
- **Purpose:** Allow lookup of existing customers
- **Components:**
  - Search/dropdown for Customer ID
  - "Auto-Fill" button to populate fields
- **Behavior:** When customer selected, all fields auto-populate

#### Input Sections (4 Cards)

| Card | Title | Fields | Layout |
|------|-------|--------|--------|
| 1 | 📊 RFM Features | Recency, Frequency (3), Monetary (3) | Left column |
| 2 | 👤 Customer Attributes | Loyalty, Points, Tenure, Segment | Right column top |
| 3 | 🛍️ Shopping Behavior | AOV, Items/Order, Stores | Right column bottom |
| 4 | 📦 Category Preferences | Top Category, Num Categories | Left column bottom |

#### Predict Button
- **Text:** "🔮 PREDICT SPEND"
- **Style:** Large, prominent, centered
- **Color:** Primary brand color (e.g., blue/green)

#### Results Section
- **Predicted Amount:** Large, bold number with currency
- **Confidence Meter:** Progress bar showing model confidence
- **Prediction Range:** Min-Max range for the prediction
- **Customer Segment:** Classification based on predicted spend

---

### 2.3 Responsive Design

| Screen Size | Layout |
|-------------|--------|
| Desktop (>1024px) | 2-column layout as shown |
| Tablet (768-1024px) | 2-column, reduced margins |
| Mobile (<768px) | Single column, stacked sections |

---

### 2.4 Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Primary | Blue | #1E88E5 |
| Secondary | Green | #43A047 |
| Background | Light Gray | #F5F5F5 |
| Card Background | White | #FFFFFF |
| Text | Dark Gray | #212121 |
| Accent | Orange | #FF9800 |

---

## 3. Input Validation Rules

| Field | Validation | Error Message |
|-------|------------|---------------|
| `recency_days` | 0 ≤ value ≤ 365 | "Recency must be between 0 and 365 days" |
| `frequency_*` | value ≥ 0, integer | "Frequency must be a positive number" |
| `monetary_*` | value ≥ 0 | "Spend amount cannot be negative" |
| `loyalty_points` | 0 ≤ value ≤ 50000 | "Points must be between 0 and 50,000" |
| `tenure_days` | value ≥ 0 | "Tenure cannot be negative" |
| `avg_order_value` | 10 ≤ value ≤ 1000 | "Average order value must be $10-$1000" |
| `num_stores` | 1 ≤ value ≤ 15 | "Stores visited must be 1-15" |

---

## 4. User Flow

```
┌──────────────────┐
│   Start          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│ Existing         │ Yes │ Enter Customer   │
│ Customer?        │────▶│ ID               │
└────────┬─────────┘     └────────┬─────────┘
         │ No                     │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│ Enter Features   │     │ Auto-Fill        │
│ Manually         │     │ Features         │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │ Review/Adjust    │
         │ Features         │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Click "Predict"  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ View Prediction  │
         │ Results          │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ New Prediction?  │──────┐
         └──────────────────┘      │
                  │ Yes            │ No
                  ▼                ▼
         ┌──────────────────┐   ┌──────────────────┐
         │ Reset Form       │   │ End              │
         └──────────────────┘   └──────────────────┘
```

---

## 5. Output Display

### 5.1 Prediction Result Components

| Component | Description | Example |
|-----------|-------------|---------|
| **Predicted Amount** | Main prediction value | "$342.50" |
| **Confidence Score** | Model confidence (if available) | "85%" |
| **Prediction Range** | Min-Max range | "$280 - $405" |
| **Customer Segment** | Value-based classification | "Medium-High Value" |
| **Recommendation** | Business action suggestion | "Consider retention offer" |

### 5.2 Segment Classification

| Predicted Spend | Segment | Color | Icon |
|-----------------|---------|-------|------|
| $0 - $100 | Low Value | Red | 🔴 |
| $100 - $300 | Medium Value | Yellow | 🟡 |
| $300 - $500 | Medium-High Value | Light Green | 🟢 |
| $500+ | High Value | Dark Green | ⭐ |

---

## 6. Technology Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Frontend Framework | **Streamlit** | Rapid prototyping, Python-native |
| Styling | Streamlit components + custom CSS | Easy customization |
| Charts | Plotly / Streamlit charts | Interactive visualizations |
| Backend | Python + scikit-learn | Model integration |

---

## 7. Accessibility Considerations

- All form fields have clear labels
- Color is not the only indicator (icons + text used)
- Keyboard navigation supported
- Screen reader compatible labels
- Sufficient color contrast (WCAG AA)

---

## 8. Future Enhancements

1. **Batch Prediction:** Upload CSV for multiple customers
2. **Historical Comparison:** Show past predictions vs. actual
3. **What-If Analysis:** Adjust features to see impact
4. **Export Results:** Download predictions as PDF/CSV
5. **API Access:** REST API for integration with other systems

---

*Document Created: 2026-02-04*  
*Version: 1.0*
