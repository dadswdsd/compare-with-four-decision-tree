import os
import time
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

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
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def _normalize_path(p):
    if p.startswith("/cygdrive/"):
        parts = p.split("/")
        drive = parts[2].upper()
        rest = "/".join(parts[3:])
        return f"{drive}:\\{rest.replace('/', '\\')}"
    return p

def prepare_dataset(dataset, data_path):
    data_path = _normalize_path(data_path)
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

def ensure_run_root():
    root = os.path.join("outputs", "runs")
    os.makedirs(root, exist_ok=True)
    return root

def make_run_dir(dataset):
    root = ensure_run_root()
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(root, ts, dataset)
    os.makedirs(d, exist_ok=True)
    return d
