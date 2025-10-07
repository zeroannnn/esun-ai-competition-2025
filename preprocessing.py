import pandas as pd
from typing import List, Callable, Dict
import os

# 每個帳戶轉出/轉入的總額
def total_send_recv_amt(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 'total_send/recv_amt': total amount sent/received by each acct
    send = df.groupby('from_acct')['txn_amt'].sum().rename('total_send_amt')
    recv = df.groupby('to_acct')['txn_amt'].sum().rename('total_recv_amt')
    out = pd.concat([send, recv], axis=1).fillna(0).reset_index()
    out = out.rename(columns={'index':'acct', 'from_acct':'acct', 'to_acct':'acct'})
    print(f"(Finish) func: total_send_recv_amt")
    return out

# 標記使用大於一種 currency 的轉入轉出戶
def is_multi_curr_acount(df):
    from_multi = df.groupby('from_acct')['currency_type'].nunique().rename('send_currency_count')
    to_multi   = df.groupby('to_acct')['currency_type'].nunique().rename('recv_currency_count')

    multi_flag = pd.concat([from_multi, to_multi], axis=1).fillna(0).astype(int)
    multi_flag['multi_currency_flag'] = ((multi_flag['send_currency_count'] > 1) | 
                                         (multi_flag['recv_currency_count'] > 1)).astype(int)
    multi_flag = multi_flag.reset_index().rename(columns={'index': 'acct'})
    print(f"(Finish) func: is_multi_curr_acount")
    return multi_flag

# 每一個帳戶轉出/轉入的最大、最小、平均價格
def calculate_max_min_avg(df: pd.DataFrame) -> pd.DataFrame:
    # 2. max, min, avg txn_amt for each account
    max_send = df.groupby('from_acct')['txn_amt'].max().rename('max_send_amt')
    min_send = df.groupby('from_acct')['txn_amt'].min().rename('min_send_amt')
    avg_send = df.groupby('from_acct')['txn_amt'].mean().rename('avg_send_amt')
    
    max_recv = df.groupby('to_acct')['txn_amt'].max().rename('max_recv_amt')
    min_recv = df.groupby('to_acct')['txn_amt'].min().rename('min_recv_amt')
    avg_recv = df.groupby('to_acct')['txn_amt'].mean().rename('avg_recv_amt')
    out = pd.concat(
        [max_send, min_send, avg_send, max_recv, min_recv, avg_recv],
        axis=1
    ).fillna(0).reset_index()
    out = out.rename(columns={'index':'acct', 'from_acct':'acct', 'to_acct':'acct'})
    print(f"(Finish) func: calculate_max_min_avg")
    return out

# 是不是玉山的帳戶
def build_is_esun(df: pd.DataFrame) -> pd.DataFrame:
    # 2. 'is_esun': is esun account or not
    df_from = df[['from_acct', 'from_acct_type']].rename(
        columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'}
    )
    df_to = df[['to_acct', 'to_acct_type']].rename(
        columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'}
    )
    df_acc = pd.concat([df_from, df_to], ignore_index=True)
    df_acc = df_acc.groupby('acct')['is_esun'].max().reset_index()
    print(f"(Finish) func: build_is_esun")
    return df_acc

# 每個帳號的日交易量
def daily_trading_volume_of_an_account(df):
    # 每一個帳號的交易總次數
    vol = (df.groupby("from_acct")
        .agg(txn_count=("txn_amt","size"))
        .reset_index())
    
    # df.groupby(["from_acct","txn_date"]).size() : 每一個帳號的某天的交易數
    # .groupby("from_acct").size() : 每一個帳號有幾天有交易
    # .rename("txn_days") : 改變欄位的名稱至 txn_days
    active = (df.groupby(["from_acct","txn_date"]).size()
                .groupby("from_acct").size()
                .rename("txn_days").reset_index())
    # 一個帳號的天交易量
    vol = vol.merge(active, on="from_acct", how="left")
    vol["txn_days"] = vol["txn_days"].fillna(0)

    vol["txn_per_day"] = vol["txn_count"] / vol["txn_days"].clip(lower=1)
    vol = vol.rename(columns={"from_acct": "acct"})
    print(f"(Finish) func: daily_trading_volume_of_an_account")
    return vol

# 每個帳戶的收款人數量
def number_of_payees(df):   
    recv_count = (df.groupby("from_acct")["to_acct"]
                    .nunique()
                    .reset_index(name="n_receivers"))
    recv_count = recv_count.rename(columns={"from_acct": "acct"})
    print(f"(Finish) func: number_of_payees")
    return recv_count

# 每個帳戶在自定義時段下，各時段的交易量以及比例
def transform_txn_time(df):
    # 從 txn_time 取出小時的數字
    df['txn_hour'] = pd.to_datetime(df['txn_time'], format='%H:%M:%S').dt.hour

    # 根據銀行活動時間定義區間
    bins = [0, 6, 9, 12, 14, 17, 21, 24]
    labels = ['LateNight', 'EarlyMorning', 'MorningWork', 'LunchBreak',
            'AfternoonWork', 'EveningTrade', 'NightTrade']

    # 針對每一筆交易時間作區間離散化
    df['txn_period'] = pd.cut(df['txn_hour'], bins=bins, labels=labels, right=False, include_lowest=True)

    # 對時段做 one-hot encoding，讓模型可以學習在哪一個時段交易是可疑的
    # 帳戶在各時段的交易筆數
    period_dum = pd.get_dummies(df['txn_period'], dummy_na=False)
    by_acct_cnt = (
        pd.concat([df[['from_acct']], period_dum], axis=1)
          .groupby('from_acct', as_index=False).sum()
          .rename(columns={'from_acct':'acct'})
          .add_prefix('cnt_') # 替每一個 attribute 都加上 'cnt_'
    )
    by_acct_cnt = by_acct_cnt.rename(columns={'cnt_acct':'acct'})  # 還原 acct 的 column name，之後才能直接跟其他資料表對齊並 group

    # 計算比例且避免除以 0
    # 有助於模型判斷異常行為，像是"某帳戶 80% 交易都發生在深夜"比"深夜有 10 筆交易"更有意義。
    # 帳戶的交易時段分佈比例
    cnt_cols = [c for c in by_acct_cnt.columns if c.startswith('cnt_') and c != 'cnt_acct']
    total_cnt = by_acct_cnt[cnt_cols].sum(axis=1).replace(0, 1)
    ratio = (by_acct_cnt[cnt_cols].div(total_cnt, axis=0)
             .add_prefix('ratio_'))
    
    # 平均交易小時，可以看出是正常戶還是可疑戶
    mean_hour = (df.groupby('from_acct')['txn_hour']
                   .mean()
                   .rename('mean_txn_hour')
                   .reset_index()
                   .rename(columns={'from_acct':'acct'}))

    # 深夜交易占比（00-06），單獨列出來針對決策有比較大的影響力
    night_ratio = (by_acct_cnt[['acct','cnt_LateNight']]
                   .assign(total=total_cnt)
                   .eval('night_ratio = cnt_LateNight / total')[['acct','night_ratio']])

    # Merge 表格
    out = by_acct_cnt.merge(ratio, left_index=True, right_index=True)
    out = out.merge(mean_hour, on='acct', how='left')
    out = out.merge(night_ratio, on='acct', how='left')

    # 缺值補 0
    num_cols = out.columns.difference(['acct'])
    out[num_cols] = out[num_cols].fillna(0)

    print("(Finish) func: transform_txn_time")
    return out

FeatureFunc = Callable[[pd.DataFrame], pd.DataFrame]
FEATURES: Dict[str, FeatureFunc] = {
    "send_recv": total_send_recv_amt,
    "is_multi_curr_acount": is_multi_curr_acount,
    "agg_stats": calculate_max_min_avg,
    "is_esun": build_is_esun,
    "daily_trading": daily_trading_volume_of_an_account,
    "number_of_payees": number_of_payees,
    "txn_period": transform_txn_time,
}

# 將選定特徵用 acct 為 key 逐一 merge 起來
def preprocess(df: pd.DataFrame, feature_list: List[str] = None) -> pd.DataFrame:
    if feature_list is None:
        feature_list = ["send_recv", "is_multi_curr_acount", "agg_stats", "is_esun", "daily_trading", "number_of_payees", "txn_period"]

    # 進去 FEATURES 跑每一個 func
    tables = [FEATURES[name](df) for name in feature_list]
    # 依序 merge
    out = tables[0]
    for t in tables[1:]:
        out = out.merge(t, on='acct', how='left')

    return out.fillna(0)

def _make_mock(path="./preliminary_data/acct_transaction.csv", n=30):
    """
    從 acct_transaction.csv 讀取前 n 筆資料
    預設 n=30
    """
    # 為了避免整份檔案都載入，可以用 nrows 參數只讀前 n 筆
    dfm = pd.read_csv(path, nrows=n)
    print("n_txn =", len(dfm))
    print("n_from_acct =", dfm["from_acct"].nunique())
    print("n_to_acct   =", dfm["to_acct"].nunique())
    print("n_union_acct =", len(set(dfm["from_acct"]) | set(dfm["to_acct"])))
    return dfm
    
if __name__ == "__main__":
    dfm = _make_mock()
    out = preprocess(dfm)
    print(out)