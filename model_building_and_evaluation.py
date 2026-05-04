# Phase 4 - Model Building & Evaluation
from datasetup_churn_definition import *
from meaningful_engineering import *
from exploratory_data_analysis import*

# 4.1 scale features
from sklearn.preprocessing import StandardScaler
import matplotlib as plt

# this is essential for logistic regression, which is sensitive to features being on different scales.
scaler = StandardScaler()           # # StandardScaler converts each feature so it has a mean of 0 and standard deviation of 1. 
X_train_scaled = scaler.fit_transform(X_train_bal)          # learn the mean and standard deviation from the training data then apply scaling          
X_test_scaled = scaler.transform(X_test)            # apl same scaling learned from training data to the data set.



# 4.2 baseline model: logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

lr = LogisticRegression(max_iter=1000, random_state=42)         # LogisticRegression() is used for binary or multiclass clasasification, used for predicting categories (churn vs not churn) with more iterations to converge (learn best weights), and random_state=42 ensures reproducibility
lr.fit(X_train_scaled, y_train_bal)         # fit() trains the model by finding the best relationship between inputs and targets, features (independent variable) that is already standardized (on similar ranges), and target variable (labels), balanced to handle oversampling like SMOTE or undersampling

y_pred_lr = lr.predict(X_test_scaled)           # hard predictions (score of 0 or 1) for each customer
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]           # probabiliy score of 0-1 for each customer.[:,1] takes the second column (probability of being churned). use these probabiltiies for ROC-AUC and threshold tuning

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))         # classification_report prints precision, recall, F1-score, and support for both classes in a formatted table. 
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):4f}")            # roc_auc_score computes the area under the ROC curve, the primary metric. 



# 4.3 strong model: random forest
from sklearn.ensemble import RandomForestClassifier         

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)         # build 200 individual decision trees and combine their votes, max_depth=10 each tree can be at most 10 levels deep prevents overfitting, n_jobs=-1 uses all available CPU cores to train in parallel (much faster)
rf.fit(X_train_bal, y_train_bal)            # fit() trains the model, input features X and target labels y

y_pred_rf = rf.predict(X_test)              # hard predictions (0 or 1) for each customer, random forest can handle unscaled data so we use X_test instead of X_test_scaled, result should  be 1
y_prob_rf = rf.predict_proba(X_test)[:, 1]      # probability that each sample belongs to the positive class, to measure the confidence of the model for each prediction

print("===Random Forest ===")
print(classification_report(y_test, y_pred_rf))         # returns a formatted string with precision, recall, and F-score values for each class
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")       # computes the Area under ROC curve, uses prob because ROC AUC needs probabilties to calculate the curve across all classificatin thresholds, formats to 4 decimal places



# 4.4 model XGBoost with hyperparameter tuning
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# dictionary of hyperparameter options to try. instead of manually guessing what settings work best, let the search try combinations
param_dist = {
    'n_estimators': [100, 200, 300],            # how many boosting rounds (trees)            
    'max_depth': [3, 5, 7],         # how complex each tree is
    'learning_rate': [0.01, 0.05, 0.1],         # how much each tree corrects the previous one
    'subsample': [0.7, 0.8, 1.0],           # what fraction of rows each tree sees
    'colsample_bytree': [0.7, 0.8, 1.0],            # what fraction of features each tree uses
    'scale_pos_weight': [1, 5, 10]      # how much extra weight to give the minority class
}

xgb_base = XGBClassifier(       # creates a base XGBoost classifier with default hyperparamters, a blank template that RandomizedSearch will experiment on,
    random_state=42,            # random_state=42 for reporducibility,
    eval_metric='logloss',          # eval_metric='logloss' tells XGBoost which metric to use internally when evaluating how well the model is learning measuring the confidence and correct the probability predictions are, the lower the more correct
    use_label_encoder=False)            # disables a deprecated label encoding step inside XGBoost that was removed in newer versions
xgb_search = RandomizedSearchCV(            # a hyperparameter tuning tool from sklearn, xgb base is the model template to tune
    xgb_base, param_dist, n_iter=30, cv=5,          # n_iter=30 tries 30 random combinations from param_dist (instead of every possible combination), cv=5 for each combination evaluate it with 5-fold cross-validation (splits training data into 5 chunks, trains on 4, and tests on 1, rotates 5 times), 5 performance scores per combination and averaged
    scoring='roc_auc', n_jobs=-1, random_state=42           #  scoring='roc_auc' picks the combination that scores highest on ROC-AUC, n_jobs=-1 uses all available CPU cores in parallel
)
xgb_search.fit(X_train_bal, y_train_bal)        # triggers the entire search process and runs all 30 combinations x 5 folds = 150 model training and evaluation cycles, X_train_bal is the balanced training to handle class imbalance

best_xgb = xgb_search.best_estimator_       # pulls the single best performing model configuration, best_estimator_ is the trained model with the winning parameters
y_pred_xgb = best_xgb.predict(X_test)
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

print("=== XGBoost (tuned) ===")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
print(f"Best params: {xgb_search.best_params}")



# 4.5 ROC curves comparison
fig, ax = plt.subplots(figsize=(8,6))

for name, y_prob in [('Logistic Regression', y_prob_lr),
                     ('Random Forest',       y_prob_rf),
                     ('XGBoost',             y_prob_xgb)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)         #fpr False Positive Rate (x-axis) - how often we wrongly flagged a retained customer as churned, tpr is the true positive rate (y-axis) how often we correctly flagged an atual churner
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0,1], [0,1], 'k--', alpha=0.4)     # draw a diagonal dashed line. this is the random baseline. a random guess would look like (AUC = 0.5). any model above this line is better than random the further the curves bow toward the top-left the better
ax.set_xlabel('False Positive Rate')            # label x and y axes and title
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves - model comparison')
ax.legend()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)



# 4.6 SHAP values
# explains why the model made each prediction, bridges data science to business storytelling
import shap

explainer = shap.TreeExplainer(best_xgb)      # TreeExplainer is optimized specificallyy for tree-based models like XGBoost, computing a SHAP value for every customer x every feature combination.
shap_values = explainer.shap_values(X_test)         # the SHAP value answers "how much did this feature push this customer's churn probability up (+) or down (-) compared to average?"
feature_names = X.columns.tolist()          # this line extracts the column names from feature datafame X and converts that index objkect into a plain Python list. SHAP does not accept a pandas Index, it needs a python list.

# Summary plot - global feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)     # generates a beeswarm summary plot that shows every customer as a dot. features are ranked by importance (top = most important), color shows the value (red = high, blue = low) position on x-axis shows the impact on churn probability. show=False prevents it from displaying immediately so you can save it first 
fig = plt.gcf()           # explicitly captures the figure shap drew into a variable
fig.tight_layout()          # calling fig directly on object ensures operation on the correct figure
fig.savefig('shap_summary.png', dpi=150, bbox_inches='tight')           # .savefig() is a matplotlib method that write the figure to a file on disk. called on fig directly to guarantee that the exact figure SHAP drew is being saved, not what matplotlib happens to think is the active figure. DPI (dots per inch) the higher the DPI the sharper the image and 150 is enough for this project. bbox_inches stands for bounding box, tight sizes the bounding box to fit all content
plt.clf()       # clears the figure so the next plot starts fresh

# Bar plot - mean absolute impact
# the bar chart is easier to explain to non-technical stakeholders: "these are the top 5 things driving churn in order of importance"
shap.summary_plot(shap_values, X_test,           
                  feature_names=feature_names,
                  plot_type='bar',          # the bar plot version shows a simpler view, just the average absolute SHAP value per feature
                    show=False)
fig = plt.gcf()         # explicitly captures the figure shap drew into a variable
fig.tight_layout()          # calling fig directly on object ensures operation on the correct figure
fig.savefig('shap_importance.png', dpi=150, bbox_inches='tight')        # .savefig() is a matplotlib method that write the figure to a file on disk. called on fig directly to guarantee that the exact figure SHAP drew is being saved, not what matplotlib happens to think is the active figure. DPI (dots per inch) the higher the DPI the sharper the image and 150 is enough for this project. bbox_inches stands for bounding box, tight sizes the bounding box to fit all content
plt.clf()       # clears the figure for the next plot



# 4.7 Precision-recall tradeoff (choose right threshold)
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_xgb)            # for every possible classification threshold between 0 and 1, compute what the precision and recall would be, return three arrays of equa length, 1 precision value, 1 recall, and 1 threshold value for each point.

fig, ax = plt.subplots(figsize=(8, 5))          # create a blank canvas fig as the picture frame, ax as the actual drawing area inside it. sets size of the entire figure (fig) to 8 inches wide and 5 inches tall. ax does not have the 8x5 size, it is usually a little smaller as it lives inside the fig
ax.plot(thresholds, precisions[:-1], label='Precision')         # ax.plot() draws a line on subplot (ax), thresholds (x-axis) are the cutoff values used to decide class labels, precisions[:-1] (y-axis) are the precision values corresponding to each threshold, [:-1] to take all elements except the last one to make the lengths equal so the plot works properly. this plot answers "how does precision change as the decision threshold changes?" higher threshold, fewer positive predictions (higher precision), lower threshold, more positive predictions (lower precision)
# recall = true positives (TP) / actual positives (TP + false negatives
ax.plot(thresholds, recalls[:-1],    label='Recall')        # ax.plot() draws a line graph on the subplot(ax), thresholds (x-axis) has cutoff values used to decide prediction class, recalls[:-1] (y-axis) are the recall values for each threshold,  recall has one more value than thresholds, slice out the last value to match lengths. this plot answers "how well does my model capture actual positives so I change the threshold?" 
ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.6, label='Threshold = 0.30')         # axvline = axis vertial line. draw a vertical line threshold 0.30. customers with churn probability above 0.3 are flagged 
ax.set_xlabel('Classification threshold')
ax.set_title('Precision vs Recall at different thresholds')
ax.legend()
plt.tight_layout()
plt.savefig('precision_recall_threshold.png', dpi=150)

# Use 0.3 as threshold for high recall
y_pred_custom = (y_prob_xgb >= 0.30).astype(int)            # apply custom threshold manually. y_prob_xgb produces a True/False array. astype(int) converts True to 1 and False to 0. Now there are predictions that flag more people as potential churners (higher recall)
print(classification_report(y_test, y_pred_custom))