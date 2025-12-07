import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_graphviz
from src.data_utils import prepare_dataset, eval_classification, eval_regression

def _ensure_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

def run(dataset, data_path, run_dir):
    X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, task = prepare_dataset(dataset, data_path)
    X_train_t = _ensure_dense(pre.transform(X_train))
    X_val_t = _ensure_dense(pre.transform(X_val))
    X_test_t = _ensure_dense(pre.transform(X_test))
    candidates = []
    if task == "classification":
        for md in [None, 10, 20]:
            for msl in [1, 2, 5]:
                candidates.append(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=md, min_samples_leaf=msl))
    else:
        for md in [None, 10, 20]:
            for msl in [1, 2, 5]:
                candidates.append(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=md, min_samples_leaf=msl))
    best_score = -1e18
    best_est = None
    for cand in candidates:
        cand.fit(X_train_t, y_train)
        if task == "classification":
            val_proba = cand.predict_proba(X_val_t)[:, 1]
            score = roc_auc(val_proba, y_val)
        else:
            val_pred = cand.predict(X_val_t)
            score = -rmse(y_val, val_pred)
        if score > best_score:
            best_score = score
            best_est = cand
    X_tv_t = np.vstack([X_train_t, X_val_t])
    y_tv = np.concatenate([y_train, y_val])
    best_est.fit(X_tv_t, y_tv)
    if task == "classification":
        y_pred = best_est.predict(X_test_t)
        y_proba = best_est.predict_proba(X_test_t)[:, 1]
        metrics = eval_classification(y_test, y_pred, y_proba)
    else:
        y_pred = best_est.predict(X_test_t)
        metrics = eval_regression(y_test, y_pred)
    p = os.path.join(run_dir, "rf_structure.dot")
    if hasattr(best_est, "estimators_") and len(best_est.estimators_) > 0:
        export_graphviz(best_est.estimators_[0], out_file=p, feature_names=feature_names, filled=True, rounded=True)
    pd.DataFrame([{**{"dataset": dataset, "model": "RandomForest"}, **metrics}]).to_csv(os.path.join(run_dir, "rf_metrics.csv"), index=False)
    return metrics

def roc_auc(proba, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, proba)

def rmse(y, pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y, pred) ** 0.5
