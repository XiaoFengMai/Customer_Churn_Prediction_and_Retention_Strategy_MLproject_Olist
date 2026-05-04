# Phase 5 - Business Insights & Retention Strategy

# 5.1 Extract headline insights from SHAP

# after SHAP plots, translate those plots to insights

# Customers with recency > 120 days are 3x more likely to churn than those with recency < 30 days
# A 1-point drop in average review score increases churn probability by approximately 22%
# Customers who experienced a delivery delay of more than 7 days churned at 2x the rate of on-time delivery customers
# High total spends is protective - customers who spent over 300 BRL had the lowerst churn rates"

# Segment customers by delivery delay and show churn rates
delay_bins = pd.cut(features['avg_delivery_delay']
                    bins=[-99,0,3,7,99],
                    labels=['On time', 
                            '1-3 days late',
                             '4-7 days late,'
                             '7+ days late'])

print(features.groupby(delay_bins)['churned'].mean().rename('churn_rate'))


# segment by review score
review_bins = pd.cut(features['avg_review_score'],
                     bins=[0, 2, 3, 4, 5.01],
                     labels=['1-2 stars', '3 stars', '4 stars', '5 stars'])

print(features.groupby(review_bins)['churned'].mean().rename('churn_rate'))






# 5.2 Risk segment the customer base

# Score every customer with the best model
features['churn_probability'] = best_xgb.predict_proba(
    features[X.columns].fillna(0)
)[:, 1]

# Assign risk tier
features['risk_tier'] = pd.cut(
    features['churn_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low risk', 'Medium risk', 'High risk']
)

print(features['risk_tier'].value_counts())
print(features.groupby('risk_tier')['total_spend'].mean().rename('avg_spend'))



# write concrete rentention recommendations

# organize recommendations around top 3 churn drivers.
# recommendation 1 - proactive delivery follow-up, trigger an automatic email/SMS to any customer whose delivery is more than 3 days late. Acknowledge delay and offer a 10% discount on their next order. 
# Target: customers in the "high risk" tier with avg_delivery_delay > 3

# recommendation 2 - post-bad-review outreach. Customers who leave a 1- or 2-star review should receive a personal outreach from a customer success agent within 48 hours.
# The offer should include a full refund on the next order if the issue was a seller problem.

# recommendation 3 - recency reactivation campaign Customers who have not purchased in 60–90 days (before they hit the 120-day high-risk window) 
# should receive a personalized "We miss you" campaign with a curated product recommendation from their purchase history + a time-limited 15% discount.