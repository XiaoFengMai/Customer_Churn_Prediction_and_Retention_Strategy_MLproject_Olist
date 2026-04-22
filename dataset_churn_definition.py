import pandas as pd
import numpy as np

# Phase 1 - Data Setup and Churn Definition

# 1.1 Load olist datasets using panda
orders          = pd.read_csv('olist_orders_dataset.csv') 
order_items     = pd.read_csv('olist_order_items_dataset.csv')
order_reviews   = pd.read_csv('olist_order_reviews_dataset.csv')
order_payments  = pd.read_csv ('olist_order_payments_dataset.csv')
customers       = pd.read_csv('olist_customers_dataset.csv')
products        = pd.read_csv('olist_products_dataset.csv')
sellers         = pd.read_csv('olist_sellers_dataset.csv')
geolocation     = pd.read_csv('olist_geolocation_dataset.csv')


# perform a quick check on each table
for name, df in [('orders', orders), ('items', order_items), ('reviews', order_reviews)]:
    print(f"\n---{name} ---")
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())


# 1.2 parse all date columns
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])   # for all the columns listed within date column, convert those columns within orders table to proper datetime format



# 1.3 filter to delivered orders only
orders_delivered = orders[orders['order_status'] == 'delivered'].copy()  # filter the orders table to only include rows where order status is delivered and create a new table called orders_delivered
print(f"Delivered orders: {len(orders_delivered)} / {len(orders)} total") # print the number of delivered orders compared to total orders  


# define churn clearly, in e-commerce, churn means a customer a customer who was once active but never returned.
# define the observation window and the churn window
# observation window is teh period used to build features
# churn window is the period after cutoff, used to check if the customer returned or not. if they had not purchased within 6 months after their last order, they are labeled as churned.

# set a cutoff date - use 6 months before the the dataset's last date
# that way there's enough data to label churn
max_date = orders_delivered['order_purchase_timestamp'].max()
cutoff_date = max_date - pd.DateOffset(months=6)

print(f"Data runs until: {max_date.date()}")  # print the last date in the dataset
print(f"Churn cutoff date: {cutoff_date.date()}") # prints cutoff date for churn


# customers who made at least one purchase before cuttoff
pre_cutoff = orders_delivered[orders_delivered['order_purchase_timestamp'] <= cutoff_date]

# customers who made at least one purchase AFTER the cutoff
post_cutoff = orders_delivered[orders_delivered['order_purchase_timestamp'] > cutoff_date]

# Label: 1 = churned (no purchase after cutoff), 0 = active (retained)
pre_customers = set(pre_cutoff['customer_unique_id'].unique()) # get unique customer ids from pre cutoff table
post_customers = set(post_cutoff['customer_unique_id'].unique()) # get unique customer ids from post cutoff table


# Note: Olist uses customer_id per order; join to customers table for unique ID
# merge to get customer_unique_id
pre_cutoff = pre_cutoff.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
post_cutoff = post_cutoff.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')

pre_customers = set(pre_cutoff['customer_unique_id'].unique())
post_customers = set(post_cutoff['customer_unique_id'].unique)

churn_labels = pd.DataFrame({'customer_unique_id': list(pre_customers)})
churn_labels['churned'] = (~churn_labels['customer_unique_id'].isin(post_customers)).astype(int) # if the customer unique id is not in post customers, label as churned (1), else active (0)

print(f"\nChurn rate:: {churn_labels['churned'].mean():.1%}") # print the churn rate as a percentage 


# the Olist dataset hsa a high churn rate (~97%) because most Brazilian e-commerce customers are one-time buyers.
# this is quite realistic
