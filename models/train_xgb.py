import os
import pandas as pd
import xgboost as xgb
from src.data_utils import prepare_dataset, eval_classification, eval_regression

def run(dataset, data_path, run_dir):
    X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, task = prepare_dataset(dataset, data_path)
    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    if task == "classification":
        model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
        try:
            model.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_val_t, y_val)], verbose=False, callbacks=[xgb.callback.EarlyStopping(rounds=50)])
        except TypeError:
            model.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_val_t, y_val)], verbose=False)
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)[:, 1]
        metrics = eval_classification(y_test, y_pred, y_proba)
    else:
        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, objective="reg:squarederror")
        try:
            model.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_val_t, y_val)], verbose=False, callbacks=[xgb.callback.EarlyStopping(rounds=50)])
        except TypeError:
            model.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_val_t, y_val)], verbose=False)
        y_pred = model.predict(X_test_t)
        metrics = eval_regression(y_test, y_pred)
    ok = False
    p = os.path.join(run_dir, "xgb_structure.dot")
    try:
        g = xgb.to_graphviz(model, num_trees=0)
        g.save(p)
        ok = True
    except Exception:
        pass
    if not ok:
        try:
            j = os.path.join(run_dir, "xgb_model.json")
            model.get_booster().save_model(j)
        except Exception:
            pass
    pd.DataFrame([{**{"dataset": dataset, "model": "XGBoost"}, **metrics}]).to_csv(os.path.join(run_dir, "xgb_metrics.csv"), index=False)
    return metrics
