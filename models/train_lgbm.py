import os
import pandas as pd
import lightgbm as lgb
from src.data_utils import prepare_dataset, eval_classification, eval_regression

def run(dataset, data_path, run_dir):
    X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, task = prepare_dataset(dataset, data_path)
    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    if task == "classification":
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42)
        model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], eval_metric="logloss", callbacks=[lgb.early_stopping(50)])
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)[:, 1]
        metrics = eval_classification(y_test, y_pred, y_proba)
    else:
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42)
        model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], eval_metric="rmse", callbacks=[lgb.early_stopping(50)])
        y_pred = model.predict(X_test_t)
        metrics = eval_regression(y_test, y_pred)
    try:
        t = os.path.join(run_dir, "lightgbm_model.txt")
        model.booster_.save_model(t)
    except Exception:
        pass
    pd.DataFrame([{**{"dataset": dataset, "model": "LightGBM"}, **metrics}]).to_csv(os.path.join(run_dir, "lgbm_metrics.csv"), index=False)
    return metrics
