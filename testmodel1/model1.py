#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:00:42 2024

@author: pangbo
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import shap
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from bayes_opt import BayesianOptimization
import seaborn as sns
import optuna
import warnings
warnings.filterwarnings('ignore')

# Read the data file
dataset_path = 'ame_msm.csv'
data = pd.read_csv(dataset_path, parse_dates=['date'])

# Unify the date format
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')

# Extracting all the attributes
all_columns = [
    'temp_diff_ame', 'rh_diff_ame', 'ws_avg_18_21_ame', 'wd_avg_18_21_ame', 'sundur_avg_12_15_ame',
    'temp_diff_chi_hato_ame', 'temp_diff_chi_yori_ame', 'press_diff_ame', 'rain_1d_ame', 
    'chi_sp', 'chi_t2m', 'chi_rh2m', 'chichibu_sp', 'yori_sp', 'hato_sp', 'chichibu_t2m',
    'yori_t2m', 'hato_t2m', 'chichibu_rh2m', 'yori_rh2m', 'hato_rh2m', 'tdiffchi1221',
    'tdiffchiyori21', 'tdiffchihato21', 'pressdiffchi1221', 'rhdiffchi1221', 'pressdiffchiyori21',
    'pressdiffchihato21', 'ncld_upper_00', 'ncld_mid_00', 'ncld_low_00', 'ncld_upper_06',
    'ncld_mid_06', 'ncld_low_06',
]

# Deleting rows containing missing values (rows that contain blank cells)
data_cleaned = data.dropna(subset=all_columns).reset_index(drop=True)

# extract all the features and give values to X and y, create predictors and predictand
X = data_cleaned[all_columns]
y = data_cleaned['fog_event']

# create training data and test data, build the dataset
train_data = data_cleaned[(data_cleaned['date'] >= '2013-10-01') & (data_cleaned['date'] <= '2018-12-31')].reset_index(drop=True)
test_data = data_cleaned[(data_cleaned['date'] >= '2019-10-01') & (data_cleaned['date'] <= '2019-12-31')].reset_index(drop=True)
X_train = train_data[all_columns]
y_train = train_data['fog_event']
X_test = test_data[all_columns]
y_test = test_data['fog_event']

# set timeseries cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# initialize the list in order to store all the results of cross-validation and folds
cv_results = []
miss_rates = []
false_alarm_rates = []
roc_aucs = []

# define the predictand
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 0.8),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.2, 0.8),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_uniform('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_uniform('lambda_l2', 0, 10),
        'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0, 5),
        #'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'verbose': -1
    }
    cv_scores = []
    for train_index, val_index in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Address the unbalanced training dataset, not deal with test data
        smoteenn = SMOTEENN(random_state=42)
        X_train_res, y_train_res = smoteenn.fit_resample(X_train_fold, y_train_fold)
        
        lgb_train = lgb.Dataset(X_train_res, label=y_train_res)
        lgb_eval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)
        
        gbm = lgb.train(param,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=[lgb_train, lgb_eval],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50, verbose=False),
                            lgb.log_evaluation(period=0)
                        ])
        
        y_pred = gbm.predict(X_val_fold, num_iteration=gbm.best_iteration)
        logloss = -np.mean(y_val_fold * np.log(y_pred + 1e-15) + (1 - y_val_fold) * np.log(1 - y_pred + 1e-15))
        cv_scores.append(logloss)
    
    return np.mean(cv_scores)

# create Optuna and optimize the model
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=1)  # n_trials can be adjusted

# aquire the best parameters
best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate':0.01,
    'verbose': -1
})
print(f"bestparam: {best_params}")

# Train the whole model again using training dataset
# Deal with unbalanced dataset
smoteenn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)

# use best parameters to train the whole model
#lgb_model_final = lgb.LGBMClassifier(**best_params, objective='binary', class_weight={0: 1, 1: 10}, boosting_type='gbdt', metric='binary_logloss')
lgb_model_final = lgb.LGBMClassifier(**best_params)
lgb_model_final.fit(X_train_res, y_train_res)

# Do prediction on test data
y_pred_proba_test = lgb_model_final.predict_proba(X_test)[:, 1]
y_pred_test = pd.Series((y_pred_proba_test >= 0.5).astype(int), index=X_test.index)

# calculate confusion matrix values (true positive, true nagetive…)
cm_test = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm_test.ravel()
miss_rate_test = fn / (fn + tp) if (fn + tp) > 0 else 0
false_alarm_rate_test = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"Missrate on test data: {miss_rate_test:.4f}, Falsealarm on test data: {false_alarm_rate_test:.4f}")
print("evaluation reports：")
print(classification_report(y_test, y_pred_test))

# output all the fog day that missed prediction（true is 1 but prediction is 0）
incorrect_dates = test_data[(y_test == 1) & (y_pred_test == 0)]['date']
print("miss days:")
print(incorrect_dates.values)

# save all the csv files
incorrect_dates.to_csv('result/incorrect_predicted_fog_days7.csv', index=False)

# calculate ROC and AUC (I will change this part into performance diagram)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)
roc_auc_test = auc(fpr_test, tpr_test)
print(f"test data ROC AUC: {roc_auc_test:.4f}")

# ROC
plt.figure()
lw = 2
plt.plot(fpr_test, tpr_test, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Test Set')
plt.legend(loc="lower right")
plt.savefig('result/test_ROC_curve7.png')
plt.close()
print("evaluation completed")

# SHAP (shapely additive explanations)
explainer = shap.TreeExplainer(lgb_model_final)
shap_values = explainer.shap_values(X_test)

# check the shape value and its shape
if isinstance(shap_values, list):
    # check if it is a list
    shap_values_for_class_1 = shap_values[1]
else:
    shap_values_for_class_1 = shap_values
# make sure that shap_values_for_class_1 is 2-dimension arrays
if len(shap_values_for_class_1.shape) == 1:
    # transform 1D arrays into 2D arrays
    shap_values_for_class_1 = shap_values_for_class_1.reshape(-1, 1)

# draw SHAP feature importance graph
plt.figure()
shap.summary_plot(shap_values_for_class_1, X_test, plot_type="bar", show=False)
plt.yticks(fontsize=9)
plt.savefig('result/1320shap_feature_importance_bar7.png')
plt.close()

# Draw SHAP summary plot
plt.figure()
shap.summary_plot(shap_values_for_class_1, X_test, show=False)
plt.savefig('result/1320shap_summary_plot7.png')
plt.close()


# The following steps are to check those who are fog days but were not correctly predicted
# extract missed fog days and see their features
incorrect_samples = test_data[(y_test == 1) & (y_pred_test == 0)]
incorrect_features = incorrect_samples[all_columns]

# check incorrect prediction statistical information 
print("feature statistical information of incorrect prediction:")
print(incorrect_features.describe())

# extract correct prediction data
correct_samples = test_data[(y_test == 1) & (y_pred_test == 1)]
correct_features = correct_samples[all_columns]

# check correct prediction statistical information
print("feature statistical information of correct prediction")
print(correct_features.describe())

# use SHAP to explain the incorrect samples
explainer = shap.TreeExplainer(lgb_model_final)

# calculate incorrect samples SHAP values
shap_values_incorrect = explainer.shap_values(incorrect_features)

# check shap_values_incorrect
if isinstance(shap_values_incorrect, list):
    shap_values_incorrect = shap_values_incorrect[1]
    
# draw SHAP summary plot
plt.figure()
shap.summary_plot(shap_values_incorrect, incorrect_features, show=False)
plt.savefig('result/incorrect_samples_shap_summary7.png')
plt.close()

print("task completed")
