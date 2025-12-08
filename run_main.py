import os
import argparse
import pandas as pd
from src.data_utils import make_run_dir
from models.train_rf import run as run_rf
from models.train_xgb import run as run_xgb
from models.train_lgbm import run as run_lgbm
from models.train_cat import run as run_cat

def run_one_dataset(dataset, data_path):
    d = make_run_dir(dataset)
    rows = []
    rows.append({"model": "RandomForest", **run_rf(dataset, data_path, d)})
    rows.append({"model": "XGBoost", **run_xgb(dataset, data_path, d)})
    rows.append({"model": "LightGBM", **run_lgbm(dataset, data_path, d)})
    rows.append({"model": "CatBoost", **run_cat(dataset, data_path, d)})
    pd.DataFrame(rows).to_csv(os.path.join(d, "combined_results.csv"), index=False)
    print(pd.DataFrame(rows))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bank","boston","melb","credit","all"], default="bank")
    parser.add_argument("--data_path", default=os.path.join(os.getcwd(), "data"))
    args = parser.parse_args()
    if args.dataset == "all":
        for ds in ["bank","boston","melb","credit"]:
            run_one_dataset(ds, args.data_path)
    else:
        run_one_dataset(args.dataset, args.data_path)

if __name__ == "__main__":
    main()
