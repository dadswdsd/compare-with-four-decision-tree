# 毕业设计：四种树模型的对比实验

## 项目概述
本项目对比了四类树模型在分类与回归任务上的表现：RandomForest、XGBoost、LightGBM、CatBoost。涵盖数据拆分、统一评估、早停与概率校准建议，并提供可复现实验脚本与甘特图。

## 目录结构
- src/：数据读取与通用工具（如 `data_utils.py`）
- models/：各模型训练脚本（`train_rf.py`、`train_xgb.py`、`train_lgbm.py`、`train_cat.py`）
- run_main.py：单数据集快速运行入口
- run_experiments.py：批量对比实验入口
- generate_gantt.py：生成项目进度甘特图
- data/：示例数据（`bank.csv`、`boston.csv`、`melb_data.csv`）
- outputs/gantt/：甘特图产出（`gantt.png`、`gantt.pdf`）

## 环境与安装
- 建议使用 Python 3.9+，并创建虚拟环境
- 安装依赖：
  - Windows PowerShell：`python -m pip install -r requirements.txt`
  - 如无 `requirements.txt`，可根据脚本需求安装：`numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn plotly`

## 数据集说明
- Bank Marketing（分类）：`data/bank.csv`
- Boston Housing（回归）：`data/boston.csv`
- Melbourne Housing（回归）：`data/melb_data.csv`
- 说明：仓库可仅保留小样本或下载指引，完整 CSV 可因体积较大不随代码提交。

## 快速开始
- 运行单数据集：
  - 分类（Bank）：`python run_main.py --dataset bank --data_path data`
  - 回归（Boston）：`python run_main.py --dataset boston --data_path data`
  - 回归（Melbourne）：`python run_main.py --dataset melb --data_path data`
- 批量对比：`python run_experiments.py`

## 当前进展与结果
- 数据划分统一为 70%/15%/15%（train/val/test），并记录类别/样本量与特征数。
- Bank 分类：
  - 验证集 LightGBM `logloss≈0.3113`，早停于约第 77 轮；测试集上 CatBoost 最优，`ROC-AUC≈0.927`、`accuracy≈0.851`，LightGBM 次之（`ROC-AUC≈0.924`）。四模型准确率集中在 `0.84–0.85`，区分能力较强。
- Boston 回归：
  - LightGBM 验证集早停于第 36 轮（`rmse≈2.89`）；测试集 XGBoost 最优（`RMSE≈3.22`、`MAE≈2.67`、`R²≈0.851`），RandomForest 次之，CatBoost/LightGBM 略逊。
- Melbourne 回归：
  - 验证集 LightGBM 最优迭代约 299；测试集四模型接近，`R²≈0.75–0.76`，以 LightGBM 略优（`RMSE≈308k`、`MAE≈157k`、`R²≈0.763`）。
- 统一评估指标：分类使用 `accuracy / F1 / ROC-AUC`，回归使用 `RMSE / MAE / R²`。
- 观察到的警告与处理：
  - `xgboost` 的 `num_trees` 已弃用，改用新 API；
  - `sklearn` 预测阶段出现“无特征名”警告，建议统一以带列名的 `pandas.DataFrame` 进行训练与预测。

## 复现实验与可视化
- 使用 `run_experiments.py` 自动跑完三数据集并输出控制台指标。
- 项目进度图：`outputs/gantt/gantt.png`、`gantt.pdf`。

## 后续计划
- 阈值调优：在验证集基于 ROC/PR 曲线选择最优分类阈值。
- 概率校准：对最优分类模型进行 Platt 或 Isotonic 校准，提升概率可解释性。
- 超参优化：
  - CatBoost：`depth`、`learning_rate`、`l2_leaf_reg`
  - LightGBM：`num_leaves`、`min_data_in_leaf`、`feature_fraction`
- 评估补充：混淆矩阵、PR-AUC、KS 值与按阈值的业务指标。

## 参考文献（精选）
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. NeurIPS.
- Hanley, J. A., & McNeil, B. J. (1982). The Meaning and Use of the Area under a ROC Curve. Radiology.

## 许可与致谢
- 许可：可根据需要添加 `LICENSE`（如 MIT）。
- 数据：感谢 UCI Bank Marketing、Boston Housing 与 Melbourne Housing 数据的提供与整理。
