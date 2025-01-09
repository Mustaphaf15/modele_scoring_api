import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

def business_cost(y_true, y_pred_probs, threshold=0.5, cost_fn=10, cost_fp=1):
    # Convertir les probabilités en classes avec le seuil donné
    y_pred = (y_pred_probs >= threshold).astype(int)
    # Calculer la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Coût total
    total_cost = cost_fn * fn + cost_fp * fp
    return total_cost


def custom_business_scorer(y_true, y_pred, cost_fp=1, cost_fn=10):
    false_positives = sum((y_true == 0) & (y_pred == 1))
    false_negatives = sum((y_true == 1) & (y_pred == 0))
    return cost_fp * false_positives + cost_fn * false_negatives


def optimal_threshold_scorer(estimator, X, y, thresholds=np.arange(0.1, 1.0, 0.1), cost_fn=10, cost_fp=1):
    y_pred_probs = estimator.predict_proba(X)[:, 1]
    costs = [business_cost(y, y_pred_probs, threshold=t, cost_fn=cost_fn, cost_fp=cost_fp) for t in thresholds]
    optimal_threshold = thresholds[np.argmin(costs)]
    return optimal_threshold

def create_new_features(df):
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df

def get_clean_data_train():
    path_train = '../data/application_train.csv'
    df = pd.read_csv(path_train)
    df = df.drop(columns=['SK_ID_CURR'], axis=1)
    df = create_new_features(df)
    var_num = df.drop(columns=['TARGET']).select_dtypes(exclude=['object']).columns.tolist()
    var_cat = df.drop(columns=['TARGET']).select_dtypes(include=['object']).columns.tolist()
    return df, var_num, var_cat
    