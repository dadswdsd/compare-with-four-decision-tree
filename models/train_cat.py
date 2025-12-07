import os
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from src.data_utils import prepare_dataset, eval_classification, eval_regression

def _ensure_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

def run(dataset, data_path, run_dir):
    X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, task = prepare_dataset(dataset, data_path)
    X_train_t = _ensure_dense(pre.transform(X_train))
    X_val_t = _ensure_dense(pre.transform(X_val))
    X_test_t = _ensure_dense(pre.transform(X_test))
    if task == "classification":
        model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_state=42, verbose=False)
        model.fit(X_train_t, y_train, eval_set=(X_val_t, y_val), verbose=False, use_best_model=True)
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)[:, 1]
        metrics = eval_classification(y_test, y_pred, y_proba)
    else:
        model = CatBoostRegressor(iterations=300, learning_rate=0.1, depth=6, random_state=42, verbose=False, loss_function="RMSE")
        model.fit(X_train_t, y_train, eval_set=(X_val_t, y_val), verbose=False, use_best_model=True)
        y_pred = model.predict(X_test_t)
        metrics = eval_regression(y_test, y_pred)
    try:
        j = os.path.join(run_dir, "catboost_model.json")
        model.save_model(j, format="json")
    except Exception:
        pass
    pd.DataFrame([{**{"dataset": dataset, "model": "CatBoost"}, **metrics}]).to_csv(os.path.join(run_dir, "catboost_metrics.csv"), index=False)
    return metrics
