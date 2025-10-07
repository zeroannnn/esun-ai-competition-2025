import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocess
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score

from typing import Tuple
import numpy as np


def LoadCSV(dir_path):
    """
    讀取挑戰賽提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    Args:
        dir_path (str): 資料夾，請把上述3個檔案放在同一個資料夾
    
    Returns:
        df_txn    : 交易資料 DataFrame
        df_alert  : 警示帳戶註記 DataFrame
        df_test   : 待預測帳戶清單 DataFrame
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    
    print("(Finish) Load Dataset.")
    return df_txn, df_alert, df_test


def PreProcessing(df):
    # 檢查欄位是否存在，做一些前置安全檢查
    required = {'from_acct', 'to_acct', 'txn_amt', 'from_acct_type', 'to_acct_type'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transaction CSV missing columns: {missing}")
    
    df_result = preprocess(df) 
    print("(Finish) PreProcessing.")
    return df_result

def TrainTestSplit(df, df_alert, df_test):
    """
    切分訓練集及測試集，並為訓練集的帳戶標上警示label (0為非警示、1為警示)
    
    備註:
        1. 測試集為待預測帳戶清單，你需要預測它們
        2. 此切分僅為範例，較標準的做法是基於訓練集再且分成train和validation，請有興趣的參賽者自行切分
        3. 由於待預測帳戶清單僅為玉山戶，所以我們在此範例僅使用玉山帳戶做訓練
    """  
    X_train = df[(~df['acct'].isin(df_test['acct'])) & (df['is_esun']==1)].drop(columns=['is_esun']).copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)
    X_test = df[df['acct'].isin(df_test['acct'])].drop(columns=['is_esun']).copy()
    
    print(f"(Finish) Train-Test-Split")
    return X_train, X_test, y_train

# def Modeling(X_train, y_train, X_test):
#     """
#     Decision Tree的範例程式，參賽者可以在這裡實作自己需要的方法
#     """
#     model = DecisionTreeClassifier(random_state=42, max_depth=4)
#     model.fit(X_train.drop(columns=['acct']), y_train)
#     y_pred = model.predict(X_test.drop(columns=['acct']))  

    
#     print(f"(Finish) Modeling")
#     return y_pred


def Modeling(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
            ) -> Tuple[np.ndarray, DecisionTreeClassifier]:
    """
    1) 以訓練資料再切出 validation，做 F1 驗證
    2) 用 GridSearchCV 尋找最佳超參數（以 F1 為 scoring）
    3) 印出最佳參數與 validation F1
    4) 用最佳模型對 X_test 預測
    5) return y_pred_test
    """

    # 準備資料，需去掉 ID 欄位
    feat_cols = [c for c in X_train.columns if c != "acct"]
    X_tr = X_train[feat_cols].copy()
    X_te = X_test[feat_cols].copy()

    # 先切一個 validation 來評估 F1
    X_tr_in, X_val, y_tr_in, y_val = train_test_split(
        X_tr, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # GridSearchCV（以 F1 為指標）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_split": [2, 20, 50, 100, 200],
        "min_samples_leaf": [1, 10, 20, 50],
        "class_weight": [None, "balanced"],
        # "criterion": ["gini", "entropy", "log_loss"],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring="f1",   # 以 F1 挑最佳
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_tr_in, y_tr_in)

    best_model: DecisionTreeClassifier = grid.best_estimator_
    # 用 validation 看真實 F1
    y_val_pred = best_model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)

    print("\n[GridSearch] Best CV F1:", grid.best_score_)
    print("[GridSearch] Best params:", grid.best_params_)
    print("[Holdout]   Validation F1:", val_f1)

    # 用最佳模型對 X_test 預測
    y_pred_test = best_model.predict(X_te)

    print("(Finish) Modeling with GridSearchCV + F1 evaluation + Visualization")
    return y_pred_test

def OutputCSV(path, df_test, X_test, y_pred, df_alert):
    """
    根據測試資料集及預測結果，產出預測結果之CSV，該CSV可直接上傳於TBrain    
    """
    print("len(X_test) =", len(X_test))
    print("len(X_test['acct']) =", len(X_test['acct']))
    print("len(y_pred) =", len(y_pred))

    df_pred = pd.DataFrame({
        'acct': X_test['acct'].values,
        'label': y_pred
    })
    
    df_out = df_test[['acct']].merge(df_pred, on='acct', how='left')
    df_out.to_csv(path, index=False)    
    print(f"(Finish) Output saved to {path}")

if __name__ == "__main__":
    dir_path = "./preliminary_data/"
    df_txn, df_alert, df_test = LoadCSV(dir_path)
    df_X = PreProcessing(df_txn)
    X_train, X_test, y_train = TrainTestSplit(df_X, df_alert, df_test)
    y_pred = Modeling(X_train, y_train, X_test)
    out_path = "result.csv"
    OutputCSV(out_path, df_test, X_test, y_pred, df_alert)   
    