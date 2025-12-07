import argparse
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

def ensure_outputs_dir():
    os.makedirs("outputs", exist_ok=True)

def split_features(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return X, y, cat_cols, num_cols

def make_classifier_preprocessor(cat_cols, num_cols):
    categorical = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pre = ColumnTransformer(transformers=[("cat", categorical, cat_cols), ("num", numeric, num_cols)])
    return pre

def make_regressor_preprocessor(cat_cols, num_cols):
    categorical = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pre = ColumnTransformer(transformers=[("cat", categorical, cat_cols), ("num", numeric, num_cols)])
    return pre

def eval_classification(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}

def eval_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def ensure_structure_dir(dataset):
    ensure_outputs_dir()
    d = os.path.join("outputs", "structures", dataset)
    os.makedirs(d, exist_ok=True)
    return d

def prepare_dataset(dataset, data_path):
    if dataset == "bank":
        df = pd.read_csv(os.path.join(data_path, "bank.csv"))
        df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
        X, y, cat_cols, num_cols = split_features(df, "deposit")
        pre = make_classifier_preprocessor(cat_cols, num_cols)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
        pre.fit(X_train)
        feature_names = pre.get_feature_names_out()
        print(f"bank split -> train:{len(X_train)} ({len(X_train)/len(X):.2%}), val:{len(X_val)} ({len(X_val)/len(X):.2%}), test:{len(X_test)} ({len(X_test)/len(X):.2%})")
        return X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, "classification"
    if dataset == "boston":
        df = pd.read_csv(os.path.join(data_path, "boston.csv"))
        X, y, cat_cols, num_cols = split_features(df, "MEDV")
        pre = make_regressor_preprocessor(cat_cols, num_cols)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
        pre.fit(X_train)
        feature_names = pre.get_feature_names_out()
        print(f"boston split -> train:{len(X_train)} ({len(X_train)/len(X):.2%}), val:{len(X_val)} ({len(X_val)/len(X):.2%}), test:{len(X_test)} ({len(X_test)/len(X):.2%})")
        return X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, "regression"
    if dataset == "melb":
        df = pd.read_csv(os.path.join(data_path, "melb_data.csv"))
        df = df.dropna(subset=["Price"])
        keep_cols = [
            "Rooms","Type","Distance","Postcode","Bedroom2","Bathroom","Car","Landsize","BuildingArea","YearBuilt","CouncilArea","Lattitude","Longtitude","Regionname"
        ]
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols + ["Price"]]
        X, y, cat_cols, num_cols = split_features(df, "Price")
        pre = make_regressor_preprocessor(cat_cols, num_cols)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
        pre.fit(X_train)
        feature_names = pre.get_feature_names_out()
        print(f"melb split -> train:{len(X_train)} ({len(X_train)/len(X):.2%}), val:{len(X_val)} ({len(X_val)/len(X):.2%}), test:{len(X_test)} ({len(X_test)/len(X):.2%})")
        return X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, "regression"
    raise ValueError("unknown dataset")

def train_model(kind, estimator, pre, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, dataset, task, out_dir):
    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    best_est = estimator
    if kind == "rf":
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
        for cand in candidates:
            cand.fit(X_train_t, y_train)
            if task == "classification":
                val_proba = cand.predict_proba(X_val_t)[:, 1]
                score = roc_auc_score(y_val, val_proba)
            else:
                val_pred = cand.predict(X_val_t)
                score = -mean_squared_error(y_val, val_pred, squared=False)
            if score > best_score:
                best_score = score
                best_est = cand
        X_tv_t = np.vstack([X_train_t, X_val_t])
        y_tv = np.concatenate([y_train, y_val])
        best_est.fit(X_tv_t, y_tv)
    elif kind == "xgb":
        if task == "classification":
            best_est.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_val_t, y_val)], verbose=False, early_stopping_rounds=50)
        else:
            best_est.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_val_t, y_val)], verbose=False, early_stopping_rounds=50)
    elif kind == "lgbm":
        if task == "classification":
            best_est.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], eval_metric="logloss", callbacks=[lgb.early_stopping(50)])
        else:
            best_est.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], eval_metric="rmse", callbacks=[lgb.early_stopping(50)])
    elif kind == "cat":
        if task == "classification":
            best_est.fit(X_train_t, y_train, eval_set=(X_val_t, y_val), verbose=False, use_best_model=True)
        else:
            best_est.fit(X_train_t, y_train, eval_set=(X_val_t, y_val), verbose=False, use_best_model=True)
    if task == "classification":
        y_pred = best_est.predict(X_test_t)
        y_proba = best_est.predict_proba(X_test_t)[:, 1]
        metrics = eval_classification(y_test, y_pred, y_proba)
    else:
        y_pred = best_est.predict(X_test_t)
        metrics = eval_regression(y_test, y_pred)
    if kind == "rf":
        if hasattr(best_est, "estimators_") and len(best_est.estimators_) > 0:
            p = os.path.join(out_dir, "RandomForest_tree0.dot")
            export_graphviz(best_est.estimators_[0], out_file=p, feature_names=feature_names, filled=True, rounded=True)
    elif kind == "xgb":
        p = os.path.join(out_dir, "XGBoost_tree0.dot")
        ok = False
        try:
            g = xgb.to_graphviz(best_est, num_trees=0)
            g.save(p)
            ok = True
        except Exception:
            pass
        if not ok:
            try:
                j = os.path.join(out_dir, "XGBoost.json")
                best_est.get_booster().save_model(j)
            except Exception:
                pass
    elif kind == "lgbm":
        try:
            t = os.path.join(out_dir, "LightGBM.txt")
            best_est.booster_.save_model(t)
        except Exception:
            pass
    elif kind == "cat":
        try:
            j = os.path.join(out_dir, "CatBoost.json")
            best_est.save_model(j, format="json")
        except Exception:
            pass
    return metrics

def run_four_models(dataset, data_path):
    X_train, X_val, X_test, y_train, y_val, y_test, pre, feature_names, task = prepare_dataset(dataset, data_path)
    out_dir = ensure_structure_dir(dataset)
    if task == "classification":
        models = {
            "RandomForest": ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            "XGBoost": ("xgb", xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)),
            "LightGBM": ("lgbm", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42)),
            "CatBoost": ("cat", CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_state=42, verbose=False)),
        }
    else:
        models = {
            "RandomForest": ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            "XGBoost": ("xgb", xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, objective="reg:squarederror")),
            "LightGBM": ("lgbm", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42)),
            "CatBoost": ("cat", CatBoostRegressor(iterations=300, learning_rate=0.1, depth=6, random_state=42, verbose=False, loss_function="RMSE")),
        }
    rows = []
    for name, (kind, est) in models.items():
        m = train_model(kind, est, pre, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, dataset, task, out_dir)
        row = {"dataset": dataset, "model": name}
        row.update(m)
        rows.append(row)
    out_df = pd.DataFrame(rows)
    ensure_outputs_dir()
    out_df.to_csv(os.path.join("outputs", f"{dataset}_results.csv"), index=False)
    return out_df

def run_bank(data_path):
    return run_four_models("bank", data_path)

def run_boston(data_path):
    return run_four_models("boston", data_path)

def run_melb(data_path):
    return run_four_models("melb", data_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bank","boston","melb","all"], default="bank")
    parser.add_argument("--data_path", default=os.path.join(os.getcwd(), "data"))
    args = parser.parse_args()
    if args.dataset == "bank":
        df = run_bank(args.data_path)
        print(df)
    elif args.dataset == "boston":
        df = run_boston(args.data_path)
        print(df)
    elif args.dataset == "melb":
        df = run_melb(args.data_path)
        print(df)
    else:
        df1 = run_bank(args.data_path)
        df2 = run_boston(args.data_path)
        df3 = run_melb(args.data_path)
        print(pd.concat([df1, df2, df3], ignore_index=True))

if __name__ == "__main__":
    main()

