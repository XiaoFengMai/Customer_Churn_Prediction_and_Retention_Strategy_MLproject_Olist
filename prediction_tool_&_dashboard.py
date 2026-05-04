# Phase 6 Prediction tool & dashboard

# 6.1 save model and scaler
import joblib

joblib.dump(best_xgb, 'churn_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
print("Model and scaler saved.")

# 6.2
import streamlit as st
import pandas as pd
import joblib

model  = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = [
    'recency_days', 'order_count', 'total_spend', 'avg_order_value',
    'total_freight', 'avg_delivery_delay', 'max_delivery_delay',
    'avg_delivery_days', 'pct_late_orders', 'avg_review_score',
    'min_review_score', 'pct_bad_reviews', 'category_diversity'
]

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.markdown("Enter customer data to get their churn probability.")

col1, col2 = st.columns(2)

with col1:
    recency_days       = st.slider("Days since last purchase", 0, 365, 90)
    order_count        = st.number_input("Number of orders", min_value=1, value=2)
    total_spend        = st.number_input("Total spend (BRL)", min_value=0.0, value=150.0)
    avg_order_value    = st.number_input("Avg order value (BRL)", min_value=0.0, value=75.0)
    total_freight      = st.number_input("Total freight paid", min_value=0.0, value=20.0)
    pct_late_orders    = st.slider("% late orders", 0.0, 1.0, 0.2)
    category_diversity = st.slider("Category diversity", 1, 10, 2)

with col2:
    avg_delivery_delay = st.slider("Avg delivery delay (days)", -10, 30, 0)
    max_delivery_delay = st.slider("Max delivery delay (days)", -10, 60, 0)
    avg_delivery_days  = st.slider("Avg actual delivery days", 1, 60, 12)
    avg_review_score   = st.slider("Avg review score", 1.0, 5.0, 4.0)
    min_review_score   = st.slider("Min review score", 1, 5, 3)
    pct_bad_reviews    = st.slider("% bad reviews (1-2 stars)", 0.0, 1.0, 0.0)

if st.button("Predict churn probability"):
    input_data = pd.DataFrame([[
        recency_days, order_count, total_spend, avg_order_value, total_freight,
        avg_delivery_delay, max_delivery_delay, avg_delivery_days, pct_late_orders,
        avg_review_score, min_review_score, pct_bad_reviews, category_diversity
    ]], columns=feature_names)

    prob = model.predict_proba(input_data)[0][1]

    st.metric("Churn probability", f"{prob:.1%}")

    if prob >= 0.6:
        st.error("HIGH RISK — immediate retention action recommended")
    elif prob >= 0.3:
        st.warning("MEDIUM RISK — include in next reactivation campaign")
    else:
        st.success("LOW RISK — no immediate action needed")



# 6.3 build a risk dashboard tab, extend app.py with second page that loads full customer data and shows:
# Total high/medium/low risk customers, Total high/medium/low risk customers, Top 10 highest-risk customers by total spend (most valuable to save), and Churn trend over time (compute monthly predicted churn rate)

# Dashboard tab (add to app.py)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["Individual prediction", "Risk dashboard"])

if page == "Risk dashboard":
    st.title("Customer Risk Dashboard")

    df = pd.read_csv('customer_features_with_scores.csv')  # pre-scored file

    c1, c2, c3 = st.columns(3)
    c1.metric("High risk customers",   (df['risk_tier'] == 'High risk').sum())
    c2.metric("Medium risk customers", (df['risk_tier'] == 'Medium risk').sum())
    c3.metric("Low risk customers",    (df['risk_tier'] == 'Low risk').sum())

    st.subheader("Top 10 highest-value customers at high risk")
    high_risk_top = (
        df[df['risk_tier'] == 'High risk']
        .nlargest(10, 'total_spend')
        [['customer_unique_id', 'total_spend', 'recency_days', 'churn_probability']]
    )
    st.dataframe(high_risk_top.style.format({'churn_probability': '{:.1%}', 'total_spend': 'BRL {:.0f}'}))