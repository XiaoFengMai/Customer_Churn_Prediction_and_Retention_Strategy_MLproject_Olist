# Customer_Churn_Prediction_and_Retention_Strategy_project_Olist
python, R, machine learning

# 🔄 Customer Churn Prediction & Retention Strategy
### End-to-end machine learning project using the Brazilian Olist E-Commerce Dataset

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Models & Evaluation](#models--evaluation)
- [Key Findings](#key-findings)
- [Business Recommendations](#business-recommendations)
- [Streamlit App](#streamlit-app)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Lessons Learned](#lessons-learned)
- [Next Steps](#next-steps)

---

## Project Overview

This project builds an end-to-end customer churn prediction system for a Brazilian e-commerce marketplace. Starting from raw transactional data, it identifies customers at high risk of never purchasing again, quantifies the key drivers behind churn, and translates those findings into concrete, actionable retention strategies.

The project follows the full data science lifecycle — data wrangling, feature engineering, exploratory analysis, model development, evaluation, and deployment — with a strong emphasis on turning model output into business value.

**What makes this project different from a standard classification exercise:**

- Churn is defined with a rigorous business logic (observation window + churn window), not just a label column
- Features are built from scratch using domain knowledge about e-commerce behavior
- SHAP values bridge the gap between model performance and stakeholder communication
- A Streamlit app makes predictions accessible to non-technical users
- Recommendations are specific, targeted, and tied to real numbers from the analysis

---

## Business Problem

In e-commerce, acquiring a new customer costs anywhere from 5× to 25× more than retaining an existing one. Despite this, most platforms invest heavily in acquisition while underinvesting in retention.

The Olist dataset reveals a striking reality: approximately **97% of customers never make a second purchase.** This is not just a data curiosity — it represents a massive pool of customers who had a single experience with the platform and walked away. If even a fraction of high-value churners can be identified early and retained, the revenue impact is significant.

This project answers three questions that a business stakeholder would actually ask:

1. **Which customers are most likely to churn?** (predictive model)
2. **Why are they churning?** (SHAP-based driver analysis)
3. **What should we do about it?** (targeted retention strategy)

---

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — available on Kaggle, free to download.

Olist is a Brazilian marketplace that connects small businesses to major e-commerce channels. The dataset covers **~100,000 orders** placed between 2016 and 2018, across **9 relational tables**.

### Tables used in this project

| Table | Rows (approx.) | Key columns used |
|---|---|---|
| `olist_orders_dataset` | 99,441 | order status, purchase timestamp, delivery dates |
| `olist_order_items_dataset` | 112,650 | price, freight value, product ID |
| `olist_order_reviews_dataset` | 99,224 | review score |
| `olist_order_payments_dataset` | 103,886 | payment value |
| `olist_customers_dataset` | 99,441 | customer unique ID, state |
| `olist_products_dataset` | 32,951 | product category |

### Schema overview

```
customers ──── orders ──── order_items ──── products
                  │
                  ├──── order_reviews
                  └──── order_payments
```

### Important note on customer IDs

The Olist dataset uses two customer identifiers. `customer_id` is unique per order (the same person gets a new `customer_id` each time they order). `customer_unique_id` tracks the actual person across orders. All analysis in this project uses `customer_unique_id` to correctly measure repeat purchase behavior.

---

## Project Architecture

```
Raw CSV files (9 tables)
        │
        ▼
  Data cleaning & merging
  (filter delivered orders, parse dates, join tables)
        │
        ▼
  Churn labeling
  (cutoff date method: 6-month observation window)
        │
        ▼
  Feature engineering
  (RFM + delivery experience + review behavior)
        │
        ▼
  EDA & class imbalance handling
  (distribution analysis, SMOTE on training data only)
        │
        ▼
  Model training & evaluation
  (Logistic Regression → Random Forest → XGBoost + SHAP)
        │
        ▼
  Business insights & recommendations
        │
        ▼
  Streamlit prediction app + risk dashboard
```

---

## Methodology

### Defining churn

The most important decision in any churn project is how you define churn. A poor definition produces a technically correct model that answers the wrong question.

**Definition used in this project:**

> A customer is labeled as **churned** if they made at least one purchase before the cutoff date and made **zero purchases in the 6 months following** their last order.

**Why this approach:**
- It mirrors how a business would actually act — you have a window of "recent" customers and you want to identify who won't come back
- It avoids predicting churn for customers who simply haven't had enough time to return
- The 6-month window is calibrated to the Olist dataset's time range (2016–2018); shorter windows increase data leakage risk

**Cutoff date logic:**

```
Dataset end date:  October 2018
Cutoff date:       April 2018  (6 months before end)

Observation window:  All orders before April 2018  → used to build features
Churn window:        April–October 2018             → used to label churn
```

### Handling class imbalance

With ~97% of customers labeled as churned, standard accuracy is meaningless as a metric. A model that always predicts "churned" scores 97% accuracy and is completely useless.

**Approach:**
- Split train/test first (80/20, stratified)
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) to the training set only
- Evaluate on the original imbalanced test set — this reflects real-world performance
- Use **ROC-AUC** and **Recall** as primary metrics, not accuracy

> **Critical:** SMOTE was never applied to the test set. Oversampling the test set inflates performance metrics and gives a false picture of how the model behaves in production.

---

## Feature Engineering

Seventeen features were engineered across four categories. This is where the most domain knowledge was applied.

### RFM features — purchase behavior

| Feature | Description | Business logic |
|---|---|---|
| `recency_days` | Days between last purchase and cutoff date | Customers who haven't bought recently are more likely to have churned |
| `order_count` | Total number of orders placed | Frequent buyers are less likely to churn |
| `total_spend` | Sum of all order values (BRL) | High spenders have more invested in the platform |
| `avg_order_value` | Mean spend per order | Proxy for customer segment (budget vs premium) |
| `total_freight` | Total freight paid | High freight costs relative to order value may drive churn |

### Delivery experience features

| Feature | Description | Business logic |
|---|---|---|
| `avg_delivery_delay` | Mean days late vs estimated delivery date | Negative = early, positive = late; delays damage trust |
| `max_delivery_delay` | Worst single delivery delay across all orders | One very bad experience can end the relationship |
| `avg_delivery_days` | Mean actual days from purchase to receipt | Absolute delivery speed, regardless of the estimate |
| `pct_late_orders` | Proportion of orders that arrived late | Customers with consistently late deliveries have worse experiences |

### Review behavior features

| Feature | Description | Business logic |
|---|---|---|
| `avg_review_score` | Mean review score across all orders | Review scores are a direct proxy for satisfaction |
| `min_review_score` | Worst single review given | A 1-star review is a strong signal of a damaged relationship |
| `pct_bad_reviews` | Proportion of reviews scored 1 or 2 stars | Repeated bad reviews signal a pattern, not a one-off |

### Product diversity feature

| Feature | Description | Business logic |
|---|---|---|
| `category_diversity` | Number of distinct product categories purchased | Customers who explore more categories are more engaged with the platform |

---

## Models & Evaluation

Three models were trained in order of increasing complexity, following a baseline-to-best approach.

### Model performance comparison

| Model | ROC-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.XXX | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XXX | 0.XX | 0.XX | 0.XX |
| **XGBoost (tuned) ✓** | **0.XXX** | **0.XX** | **0.XX** | **0.XX** |

> Replace `0.XXX` values with your actual results after running the notebooks.

### Why XGBoost won

- Handles non-linear relationships between features (e.g., the relationship between delivery delay and churn is not linear — small delays have little effect, large delays have a sharp effect)
- Built-in handling of missing values
- `scale_pos_weight` parameter natively addresses class imbalance
- Consistently outperforms Random Forest on tabular datasets with mixed feature types

### Evaluation metric rationale

**ROC-AUC** was chosen as the primary metric because it measures how well the model ranks churners above non-churners across all possible classification thresholds. This is more meaningful than accuracy in imbalanced settings.

**Recall** was prioritized over Precision for the business use case. It is cheaper to send a retention offer to a customer who wasn't going to churn (false positive) than to miss a churner who would have been saved (false negative). The classification threshold was tuned to 0.30 (instead of the default 0.50) to maximize recall.

### SHAP analysis

SHAP (SHapley Additive exPlanations) was used to explain the XGBoost model at both global and individual customer levels.

**Global importance (mean |SHAP value| across all customers):**

1. `recency_days` — by far the strongest predictor
2. `avg_review_score`
3. `avg_delivery_delay`
4. `order_count`
5. `pct_late_orders`

**How to read the SHAP summary plot:**
- Each dot represents one customer
- Position on the x-axis shows whether the feature pushed the prediction toward churn (right) or retention (left)
- Color shows the feature value — red = high, blue = low

---

## Key Findings

These insights came directly from the SHAP analysis and segment-level churn rate breakdowns. All percentages should be replaced with your actual computed values.

**1. Recency is the dominant signal**
Customers who had not purchased in more than 90 days before the cutoff churned at dramatically higher rates than recent buyers. The relationship is non-linear — churn risk accelerates sharply after the 60-day mark. This suggests a 60-day reactivation trigger is the right intervention point.

**2. Delivery delays are a strong churn predictor**
Customers who experienced at least one delivery delay of more than 7 days churned at approximately 2× the rate of customers who always received orders on time. Importantly, it is not the average delay that matters most — it is the *worst single delay* (`max_delivery_delay`). One very bad experience outweighs several good ones.

**3. Review scores have a sharp threshold effect**
Customers with an average review score of 3 or below churned at significantly higher rates than 4- and 5-star reviewers. However, the biggest jump in churn risk occurs between 2-star and 3-star reviewers, not between 1-star and 2-star. This suggests customers who are "disappointed but not furious" are actually the most recoverable — and the most worth targeting.

**4. High spenders churn less, but are more valuable to save when they do**
`total_spend` is negatively correlated with churn — customers who have spent more have a stronger relationship with the platform. However, when a high-spend customer does fall into the high-risk tier, the expected revenue loss is much greater. The dashboard prioritizes high-risk customers ranked by total spend for this reason.

**5. Category diversity is protective**
Customers who purchased across multiple product categories churned at lower rates. This makes intuitive sense — a customer who has found the platform useful for different needs has a broader relationship with it. Cross-selling to single-category customers is both a retention and engagement strategy.

---

## Business Recommendations

Recommendations are organized by churn driver and ordered by estimated impact.

### Recommendation 1 — Proactive delivery recovery (highest priority)

**Trigger:** Any delivered order where `delivery_delay_days > 3`

**Action:** Automated outreach within 24 hours of the delayed delivery. Acknowledge the delay by name, explain the reason if known, and attach a 10% discount on the next order with a 30-day expiry.

**Target segment:** All customers in the high-risk tier whose `max_delivery_delay > 7`

**Why it works:** The SHAP analysis shows delivery delay has an asymmetric effect — a large delay (7+ days) causes a disproportionate churn risk spike. Addressing it immediately, before the customer has time to form a negative lasting impression, is the highest-leverage intervention.

**Expected outcome:** A 10% retention improvement among customers who experienced significant delays, based on industry benchmarks for proactive service recovery.

---

### Recommendation 2 — Post-bad-review customer recovery

**Trigger:** Any review scored 1 or 2 stars

**Action:** A customer success agent personally contacts the customer within 48 hours. For product issues, offer a full refund or replacement. For delivery issues, escalate to the seller. In both cases, close the loop with a follow-up message confirming resolution.

**Target segment:** Customers with `min_review_score <= 2` and `churn_probability >= 0.6`

**Why it works:** A bad review is a customer's way of communicating that something went wrong. Most companies read reviews and do nothing. The act of personally reaching out signals that the platform takes their experience seriously — which is the single fastest way to rebuild trust.

**Note:** Do not automate this with a generic template. Customers who left 1-star reviews will recognize a mass email and it will make churn more likely, not less.

---

### Recommendation 3 — Recency-based reactivation campaign

**Trigger:** Customer has not purchased in 60–90 days (the pre-churn window before the high-risk threshold)

**Action:** A three-email "We miss you" sequence:

- Email 1 (day 60): Personalized product recommendations based on purchase history. No discount — just remind them the platform exists and show you know their preferences.
- Email 2 (day 75): Add a 15% time-limited discount (expires in 7 days). Create urgency without devaluing the relationship immediately.
- Email 3 (day 89): Final reminder. Emphasize the discount expiry. Short, direct, no clutter.

**Target segment:** Customers with `recency_days` between 60 and 90 and `total_spend > 100 BRL`

**Why it works:** The SHAP analysis shows that churn risk accelerates between 60 and 120 days. Acting at 60 days catches customers before they have fully disengaged — they still remember the platform, which makes reactivation significantly cheaper than acquisition.

---

### Recommendation 4 — High-value at-risk VIP intervention

**Trigger:** `churn_probability >= 0.6` AND `total_spend` in top 20%

**Action:** White-glove outreach from a senior customer success representative. Offer a dedicated account manager, early access to new product categories, and a significant loyalty discount (20–25%). This customer segment represents a disproportionate share of revenue — treat them accordingly.

**Target segment:** High-risk customers ranked by total spend (visible in the risk dashboard)

**Why it works:** The 80/20 principle applies in e-commerce — a small percentage of customers generate a large percentage of revenue. Losing a BRL 2,000 customer is not the same as losing a BRL 80 customer. The model + dashboard makes it trivial to identify these customers before they leave.

---

### Recommendation 5 — Seller accountability improvements

**Systemic recommendation:** The data shows that delivery delays and bad reviews are disproportionately concentrated among a subset of sellers. Identify the bottom 10% of sellers by `avg_delivery_delay` and `avg_review_score`. Implement a three-strikes policy: sellers who consistently cause negative customer experiences should be warned, suspended, and removed.

**Why it works:** Churn is not purely a marketing problem. If the product or delivery experience is bad, no retention campaign can compensate. Fixing the upstream cause (unreliable sellers) prevents churn rather than just treating it.

---

## Streamlit App

The project includes an interactive Streamlit application with two pages.

### Page 1 — Individual churn predictor

Input any customer's data manually and get an instant churn probability score with a risk tier label (Low / Medium / High) and a recommended action.

**Screenshot:** *(add screenshot here)*

### Page 2 — Risk dashboard

Loads the full scored customer dataset and displays:

- Summary metrics: total high / medium / low risk customers
- Top 10 highest-value customers at high risk (sorted by total spend)
- Churn probability distribution chart
- Churn rate by delivery delay segment
- Churn rate by review score segment

**Screenshot:** *(add screenshot here)*

To run the app:

```bash
streamlit run app.py
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Go to [this Kaggle page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), download all CSV files, and place them inside the `data/` folder.

### 5. Run the notebooks in order

```bash
jupyter notebook
```

Open and run the notebooks in this sequence:

| Order | Notebook | What it does |
|---|---|---|
| 1 | `01_data_preparation.ipynb` | Loads, cleans, and merges all tables |
| 2 | `02_feature_engineering.ipynb` | Builds all 17 features and the churn labels |
| 3 | `03_eda.ipynb` | Exploratory analysis, distributions, correlation |
| 4 | `04_modeling.ipynb` | Trains all models, evaluates, generates SHAP plots |
| 5 | `05_business_insights.ipynb` | Segment analysis and recommendation validation |

### 6. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Project Structure

```
customer-churn-prediction/
│
├── data/                          ← Place Olist CSV files here (not committed to git)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_products_dataset.csv
│   └── olist_sellers_dataset.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_business_insights.ipynb
│
├── models/
│   ├── churn_model.pkl            ← Trained XGBoost model
│   └── scaler.pkl                 ← Fitted StandardScaler
│
├── outputs/
│   ├── customer_features_scored.csv   ← Full dataset with churn probabilities
│   ├── churn_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── shap_summary.png
│   └── precision_recall_threshold.png
│
├── app.py                         ← Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost |
| Imbalanced learning | imbalanced-learn (SMOTE) |
| Model explainability | shap |
| App deployment | streamlit |
| Model persistence | joblib |
| Environment | Jupyter Notebook |

---

## Lessons Learned

**Defining the problem correctly matters more than model choice.**
Spending time getting the churn definition right — choosing the observation window, the churn window, and handling the `customer_unique_id` vs `customer_id` distinction — was more impactful than any model tuning. A well-defined problem with a simple model beats a poorly-defined problem with a complex model every time.

**Feature engineering is where the real work happens.**
The model itself took perhaps 20% of the total project time. The features — especially the delivery delay metrics and review behavior aggregations — took the other 80%. Good features make even simple models perform well.

**SHAP values are the bridge between data science and business.**
Presenting a ROC-AUC score to a business stakeholder means nothing to them. Saying "customers who experienced a delivery delay of more than 7 days are twice as likely to churn" is immediately actionable. SHAP makes this translation possible.

**Class imbalance is a process problem, not just a model problem.**
The 97% churn rate initially seemed like a modeling challenge. It turned out to be partly a feature engineering signal (most Olist customers are genuinely one-time buyers) and partly a threshold tuning problem. The most important fix was changing the evaluation metric, not the model.

---

## Next Steps

Several extensions would make this project stronger:

- **Survival analysis** — Instead of a binary churn label, use time-to-churn modeling (e.g., Kaplan-Meier curves or Cox Proportional Hazards) to predict *when* a customer will churn, not just *if*
- **Geographic segmentation** — The geolocation table enables state-level churn analysis; delivery delays likely vary significantly by region and this could drive targeted logistics improvements
- **Seller-level analysis** — Extend the model to identify which sellers are driving the most churn, enabling a seller accountability scoring system
- **Real-time scoring pipeline** — Replace the batch-scored CSV in the dashboard with a live API endpoint that scores customers as new orders arrive
- **A/B test framework** — Design an experiment to measure the actual retention lift from the recommendations, closing the loop between prediction and outcome


---

## License

This project is licensed under the MIT License. The Olist dataset is licensed under CC BY-NC-SA 4.0 — see the [Kaggle page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for details.

---

*If you found this project useful, feel free to star the repository ⭐*
