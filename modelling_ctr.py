import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import shap
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from pyffm import PyFFM
from scipy.stats import randint, uniform


def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2, shuffle=False, target_col='click'):
    X_train, X_test = train_test_split(data, test_size=test_size, shuffle=shuffle)
    y_train = X_train[target_col]
    y_test = X_test[target_col]
    return X_train, X_test, y_train, y_test

def drop_unused_columns(df, cols_to_drop):
    return df.drop(columns=cols_to_drop)

def response_fit(data, feature_name):
    df_vocab = data.groupby([feature_name, 'click']).size().unstack()
    df_vocab['CTR'] = df_vocab[1] / (df_vocab[0] + df_vocab[1])
    df_vocab.dropna(inplace=True)
    mean_CTR = df_vocab['CTR'].mean()
    vocab_dict = df_vocab['CTR'].to_dict()
    return vocab_dict, mean_CTR

def response_transform(x, vocab, mean_CTR):
    return [vocab.get(row, mean_CTR) for row in x]

def create_pCTR_features(X_train, X_test, feature_names):
    X_train_pCTR = pd.DataFrame()
    X_test_pCTR = pd.DataFrame()
    for name in tqdm(feature_names):
        vocab, mean = response_fit(X_train, name)
        X_train_pCTR[name] = response_transform(X_train[name], vocab, mean)
        X_test_pCTR[name] = response_transform(X_test[name], vocab, mean)
    return X_train_pCTR, X_test_pCTR


def get_param_distributions_rf():
    return {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }

def get_param_distributions_dt():
    return {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }

def get_param_distributions_lgb():
    return {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': [5, 10, -1],
        'num_leaves': randint(31, 127),
        'feature_fraction': uniform(0.6, 0.4),
        'bagging_fraction': uniform(0.6, 0.4)
    }

def get_param_distributions_xgb():
    return {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': [5, 10, 15],
        'colsample_bytree': uniform(0.6, 0.4),
        'subsample': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10)
    }


def tune_and_train_model(model, param_distributions, X_train, y_train, n_iter=10, scoring='neg_log_loss', random_state=42):
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, 
                                       n_iter=n_iter, scoring=scoring, n_jobs=-1, random_state=random_state)
    random_search.fit(X_train, y_train)
    print("Best Parameters:", random_search.best_params_)
    return random_search.best_estimator_

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_dist = get_param_distributions_rf()
    return tune_and_train_model(rf, param_dist, X_train, y_train)

def train_dt(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    param_dist = get_param_distributions_dt()
    return tune_and_train_model(dt, param_dist, X_train, y_train)

def train_lgb(X_train, y_train):
    lgb_gpu = lgb.LGBMClassifier(device='gpu', random_state=42)
    param_dist = get_param_distributions_lgb()
    return tune_and_train_model(lgb_gpu, param_dist, X_train, y_train)

def train_xgb(X_train, y_train):
    xgb_model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', random_state=42)
    param_dist = get_param_distributions_xgb()
    return tune_and_train_model(xgb_model, param_dist, X_train, y_train)

def find_best_sgd_alpha(X_train, y_train, X_test, y_test):
    alpha_values = [10 ** x for x in range(-5, 2)]
    log_error_array = []
    for i in alpha_values:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42, class_weight={0: 1, 1: 1.75})
        clf.fit(X_train, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X_train, y_train)
        predict_y = sig_clf.predict_proba(X_test)
        log_loss_value = log_loss(y_test, predict_y, eps=1e-15)
        log_error_array.append(log_loss_value)
    best_alpha = alpha_values[np.argmin(log_error_array)]
    return best_alpha

def train_sgd_with_best_alpha(X_train, y_train, best_alpha):
    final_clf = SGDClassifier(alpha=best_alpha, penalty='l2', loss='log', random_state=42, class_weight={0: 1, 1: 1.75})
    final_clf.fit(X_train, y_train)
    final_sig_clf = CalibratedClassifierCV(final_clf, method="sigmoid")
    final_sig_clf.fit(X_train, y_train)
    return final_sig_clf

def find_best_pyffm_lambda(X_train, y_train, X_test, y_test):
    lamda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100]
    log_error_array = []
    for i in lamda_values:
        training_params = {'epochs': 10, 'reg_lambda': i, 'sigmoid': True, "parallel": True, 'early_stop': True}
        pyffm_model = PyFFM(model='ffm', training_params=training_params)
        pyffm_model.train(X_train)
        predict_y = pyffm_model.predict(X_test)
        log_error_value = log_loss(y_test, predict_y, eps=1e-15)
        log_error_array.append(log_error_value)
    best_lambda = lamda_values[np.argmin(log_error_array)]
    return best_lambda

def train_pyffm_with_best_lambda(X_train, y_train, best_lambda):
    training_params = {'epochs': 10, 'reg_lambda': best_lambda, 'sigmoid': True, "parallel": True, 'early_stop': True}
    pyffm_final = PyFFM(model='ffm', training_params=training_params)
    pyffm_final.train(X_train)
    return pyffm_final


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)
    print(f"The test log loss is: {loss}")
    return loss

def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    A = (C.T / C.sum(axis=1)).T
    B = C / C.sum(axis=0)
    plt.figure(figsize=(20, 4))
    labels = [0, 1]
    cmap = sns.light_palette("green")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    plt.subplot(1, 3, 3)
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    plt.show()


if __name__ == "__main__":
    data = load_data('train_data.csv')
    X_train, X_test, y_train, y_test = split_data(data)
    
    cols_to_drop = ['device_ip_counts', 'device_id_counts', 'hour_of_day', 'day_of_week', 'hourly_user_count']
    X_test = drop_unused_columns(X_test, cols_to_drop)
    
    feature_names = [
        'site_id', 'site_domain', 'site_category', 'app_id', 'app_category', 'app_domain',
        'device_model', 'device_type', 'device_conn_type', 'device_id_counts', 'device_ip_counts',
        'banner_pos', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
        'hour_of_day', 'day_of_week', 'hourly_user_count'
    ]
    
    X_train_pCTR, X_test_pCTR = create_pCTR_features(X_train, X_test, feature_names)
    X_train_pCTR['click'] = y_train
    
    rf_best = train_rf(X_train_pCTR, y_train)
    evaluate_model(rf_best, X_test_pCTR, y_test)
    
    dt_best = train_dt(X_train_pCTR, y_train)
    evaluate_model(dt_best, X_test_pCTR, y_test)
    
    lgb_best = train_lgb(X_train_pCTR, y_train)
    evaluate_model(lgb_best, X_test_pCTR, y_test)
    
    xgb_best = train_xgb(X_train_pCTR, y_train)
    evaluate_model(xgb_best, X_test_pCTR, y_test)
    
    best_alpha = find_best_sgd_alpha(X_train_pCTR, y_train, X_test_pCTR, y_test)
    print(f"Best Alpha: {best_alpha}")
    sgd_model = train_sgd_with_best_alpha(X_train_pCTR, y_train, best_alpha)
    evaluate_model(sgd_model, X_test_pCTR, y_test)
    
    best_lambda = find_best_pyffm_lambda(X_train_pCTR, y_train, X_test_pCTR, y_test)
    print(f"Best Lambda: {best_lambda}")
    pyffm_final = train_pyffm_with_best_lambda(X_train_pCTR, y_train, best_lambda)
    y_pred_pyffm = pyffm_final.predict(X_test_pCTR)
    print("The test log loss (PyFFM) is:", log_loss(y_test, y_pred_pyffm, eps=1e-15))
    
    predict_y_val_pyffm = [1 if val >= 0.15 else 0 for val in y_pred_pyffm]
    plot_confusion_matrix(y_test, predict_y_val_pyffm)
