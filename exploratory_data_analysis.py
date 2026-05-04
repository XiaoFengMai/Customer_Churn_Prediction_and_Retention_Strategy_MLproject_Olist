# Phase 3 exploratory data analysis (EDA)
# 3.1 understand churn rate and class imbalance

import matplotlib.pyplot as plt     # used for plotting charts
import seaborn as sns       # is a python data visualization library based on matplotlib. used to generate statistical graphs, makes certain charts like heatmaps easier to generate
import pandas as pd

from datasetup_churn_definition import *
from meaningful_engineering import *

print("Churn distribution:")    
print(features['churned'].value_counts(normalize=True))     # prints hwo many customers are churned vs retained as percentages. normalize=True converts raw counts into proportions (rates)


# Bar chart
fig, ax = plt.subplots(figsize=(6, 4))      # create a blank canvas fig as the picture frame, ax as the actual drawing area inside it. sets to 6 inches wide and 4 inches tall
features['churned'].value_counts().plot(kind='bar'), ax=ax, color=['#3B8BD4', '#E24B4A']    # counts how many customers are retained in each class (0 = retained, 1 = churned), first color blue goes to larger bar (churned) and second color red to the smaller bar (retained)
ax.set_xticklabels(['Retained (0)', 'Churned (1)'], rotation=0)     # set xticklabels replaces the default "0" and "1" with readable names. rotation 0 keeps the bars horizontal
ax.set_title('Class distribution')
plt.tight_layout()      # automatically adjusts spacing so nothing gets cut off
plt.savefig('churn_distribution.png', dpi=150)      # saves the chart to disk at 150 DPI (high enough quality)




# 3.2 compare features by churn group
# storing the 8 most important feature column names in a variable so no need to retype them every time they are referenced.
feature_cols = [
    'recency_days', 'order_count', 'total_spend', 'avg_order_value', 
    'avg_delivery_delay', 'pct_late_orders', 'avg_review_score', 
    'pct_bad_reviews'
]

# group by ('churned) splits customers into two groups: churned and not churned
# [feature_cols] only look at those 8 columns
# .mean() calculates the average of each feature within each group
# .T transpose flips rows and columns so features are rows and churn groups are columns
# .round(2) rounds to 2 decimal places
print(features.groupby('churned')[feature_cols].mean().T.round(2))       # summary statistics split by churn



# 3.3 distribution plots for top features
fig, axes = plt.subplots(2,4, figsize=(16,8))       # create axes, a 2D array(matrix) of 8 subplot charts arranged in a grid of 2 rows x 4 columns, overall figure size is 16in wide and 8in tall
axes = axes.flatten()       # convert the 2D grid array into a flat 1D list of 8 axes which allows you to loop through them with simple index (axes[0], axes[1])

for i, col in enumerate(feature_cols):
    ax = axes[i]        # loop through each of the 8 plots, enumerate gives the position number(i) and column name(col), ax = axes[i] points to the correct chart slot for this feature
    for churn_val, color, label in [(0, '#3B8BD4', 'Retained'), (1, '#E24B4A', 'Churned')]:     # this is a second loop that runs twice, once for trained customers and another for churned customers. each iteration gets group #, color, and display label
        subset = features[features['churned'] == churn_val][col].dropna()               # filter to retained OR churned customers. [col] grabs the current feature column, dropna removes any missing values
        ax.hist(subset, bins=40, alpha=0.5, color=color, label=label, density=True)         # , ax.hist draws a histogram with 40 bars, alpha=0.5 adds 50% transparency so two overlapping histogarms are both visible, density=true normalizes the y-axis to proportions
    ax.set_title(col)
    ax.legend(fontsize=8)           # lael each chart with the feature name and add a legend showing which color is retained vs churned

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150)           # fix spacing between charts and save to disk


# 3.4 correlation heatmap
corr = features [feature_cols +['churned']].corr()      # adds the churn label to the list so it appears on the heatmap. .corr computes the Pearson correlation between every pair of columns.

fig, ax = plt.subplots(figsize=(10,8))      
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',       # annot=True prints the actual correlation number inside each cell, fmt='.2f' formats those numbers to 2 decimal places. cmap='coolwarm' red for positive correlation, blue for negative
            center=0, ax=ax, linewidths=0.5)                # center=0 makes 0 appear as white/neutral, so positives and negatives are visually distinct, linewidths=0.5 thin grideline betweeen cells to improve readability
ax.set_title('Feature correlation matrix')          
plt.tight_layout()          # automatically adjusts subplot so that they fit in the figure area
plt.savefig('correlation_heatmap.png', dpi=150)         # saves current figure, save as a .png, and dpi controls resolution/quality of image, higher DPI equals sharper image and larger file size; dpi 150 usually enough for presentations, dpi 300 is typically used for high print quality.



# 3.5 handle class imbalance with SMOTE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = features[feature_cols + ['category_diversity']].fillna(0)           # split data into features (X=inputs, engineered columns and y=your prediction, a single churn label column) ), fillna(0) replaces any null values with 0 so the model does not crash
y = features['churned']

X_train, X_test, y_train, y_test = train_test_split(            # splits the data into training and testing sets, test_size=0.2 means 20% of the data goes to the test set
    X, y, test_size=0.2, random_state=42, stratify=y            # random_state=42 ensures the split is reproducible (get the same split everytime), stratify=y ensures thE 97%/3% churn ratio is the same in both training and test set. without this, may end up with no retained customers in test set
)

# SMOTE (syntehtic minority oversampling technique) creates synthetic new examples of the minority class (retained customers).  
# it does not duplicate existing rows, instead it generates new, plausable customer profiles. the result is X_train_bal and y_train_bal have a balanced 50/50 split between churned and retained.
# only SMOTE training date, not testing. test set must stay untouched to simulate real world conditions
smote = SMOTE(random_state=42)      
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Training set before SMOTE: {y_train.value_counts().to_dict()}")        
print(f"Training set after SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")           # print the before and after class counts so you can confirm SMOTE worked.
# the result should be something like {1: 7600, 0: 2400} before and {1:76000, 0:76000} after
