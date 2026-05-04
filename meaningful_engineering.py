from datasetup_churn_definition import *

# Phase 2: build meaniningful features


# 2.1 build base feature table RFM (Recency, Frequency, Monetary), a proven framework for customer behavior

# snapshot date is equal to cutoff
snapshot_date = cutoff_date

# Aggregate per customer from pre_cutoff orders
rfm_base = pre_cutoff.merge(
    order_items[['order_id', 'price', 'freight_value']],
    on='order_id'
).groupby('customer_unique_id').agg(
    last_purchase   =   ('order_purchase_timestamp',    'max'),
    order_count     =   ('order_id',                'nunique'),
    total_spend     =   ('price',           'sum'),
    avg_order_value =   ('price',           'mean'),
    total_fright    =   ('freight_value',   'sum')  
).reset_index()

# Recency: days since last purchase
rfm_base['recency_days'] = (snapshot_date - rfm_base['last_purchase']).dt.days

# Drop the raw date column
rfm_base = rfm_base.drop(columns=['last_purchase'])





# 2.2 add delivery experience features
# delivery problems are a major churn driver - this is the differentiator

delivery_features = orders_delivered[
    orders_delivered['order_purchase_timestamp'] <= cutoff_date]

delivery_features = delivery_features.merge(
    customers[['customer_id', 'customer_unique_id']], on='customer_id'
)

# Actual vs estimated delivery
delivery_features['delivery_delay_days'] = (            # create a new column called delivery_delay_days in the dataframe
    delivery_features['order_delivered_customer_date'] - delivery_features ['order_estimated_delivery_date']            # the result of this will be stored in the new column
).dt.days       # extract just the number of days from the time delta object (3 days 00:00:00, -2 days 00:00:00, etc)
# a positive value means the delivery was late and results in a higher churn, a negative value means it was early, and 0 means it was on time



# Days from purchase to actual delivery
delivery_features['actual_delivery_days'] = (
    delivery_features['order_delivered_customer_date'] - delivery_features ['order_purchase_timestamp']
).dt.days   # extract just the number of days from the time delta object (3 days 00:00:00, -2 days 00:00:00, etc)
# the shorter the delivery days, the lower the churn


# grouping delivery data
delivery_agg = delivery_features.groupby('customer_unique_id').agg(     # group all of customer orders by buckets, one bucket per customer unique id; then apply the following calculations to each bucket
    avg_delivery_delay  =   ('delivery_delay_days', 'mean'),
    max_delivery_delay  =   ('delivery_delay_days', 'max'),
    avg_delivery_days   =   ('actual_delivery_days', 'mean'),
    pct_late_orders     =   ('delivery_delay_days', lambda x: (x > 0).mean())     # uses a custom lambda function, lamba x, x is all the delay values for one customer, x > 0 turns each value into True/False (delay > 0 = true it's late), mean() gives the proportion of Trues out of total results
).reset_index()     # promotes the customer_unique_id back to the normal column, especially inmportant for merging with other tables later


# 2.3 add review score features
review_features = order_reviews[['order_id', 'review_score']].copy()

review_with_customer = (
    pre_cutoff[['order_id', 'customer_unique_id' ]]
    .merge(review_features, on='order_id')
)

review_agg = review_with_customer.groupby('customer_unique_id').agg(
    avg_review_score = ('review_score', 'mean'),
    min_review_score = ('review_score', 'min'),
    pct_bad_reviews = ('review_score', lambda x: (x <= 2).mean())
).reset_index()



# 2.4 merge all the features into a single table
features = (
    rfm_base
    .merge(delivery_agg, on='customer_unique_id', how='left')
    .merge(review_agg, on='customer_unique_id', how='left')
    .merge(churn_labels, on='customer_unique_id', how='inner')
)

# fill nulls for customers who never left a review or had no delivery data
features.fillna({
    'avg_delivery_delay':0,
    'max_delivery_delay': 0,
    'avg_delivery_days': 3.0,
    'min_review_score':3.0,
    'pct_bad_reviews': 0
}, inplace=True)

print(features.shape)
print(features.isnull().sum())


# 2.5 Add product category diversity
items_with_category = (
    order_items[['order_id', 'product_id']]
    .merge(products[['product_id', 'product_category_name']], on='product_id')
)

category_diversity = (
    pre_cutoff[['order_id', 'customer_unique_id']]
    .merge(items_with_category, on='order_id')
    .groupby('customer_unique_id')['product_category_name']
    .nunique()
    .reset_index()
    .rename(columns={'product_category_name': 'category_diversity'})
)

features = features.merge(category_diversity, on='customer_unique_id', how='left')
features['category_diversity'].fillna(1, inplace=True)



