import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display

display.set_matplotlib_formats("svg")

from meta import config
from meta.data_processor import DataProcessor
from main import check_and_make_directories
from meta.data_processors.baostock import ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
from meta.config_tickers import DOW_30_TICKER
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)
import pyfolio
from pyfolio import timeseries

import os.path

pd.options.display.max_columns = None

print("ALL Modules have been imported!")


### Create folders

"""
use check_and_make_directories() to replace the following

if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")
"""

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)


### Download data, cleaning and feature engineering

ticker_list = [
    "600000.SH",
    "600009.SH",
    "600016.SH",
    "600028.SH",
    "600030.SH",
    "600031.SH",
    "600036.SH",
    "600050.SH",
    "600104.SH",
    "600196.SH",
    "600276.SH",
    "600309.SH",
    "600519.SH",
    "600547.SH",
    "600570.SH",
]

TRAIN_START_DATE = "2019-01-01"  # 历史训练开始日期
TRAIN_END_DATE = "2024-03-01"    # 历史训练结束日期
TRADE_START_DATE = "2024-03-01"  # 交易评估开始日期
TRADE_END_DATE = "2025-04-15"    # 尽可能接近当前日期

from meta.data_processors._base import DataSource

TIME_INTERVAL = "1d"
kwargs = {}
kwargs["token"] = "2739bd3af641326a97a330c4f0890b5a7c7992b252e310be176d310e"
p = DataProcessor(
    data_source=DataSource.baostock,
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL,
    **kwargs,
)

# download and clean
p.download_data(ticker_list=ticker_list)
p.clean_data()
p.fillna()

# add_technical_indicator
p.add_technical_indicator(config.INDICATORS)
p.fillna()
print(f"p.dataframe: {p.dataframe}")

# 定义模型前缀 - 将此行从下面移到这里
MODEL_PREFIX = f"china_stock_{len(ticker_list)}_stocks"

# 保存处理好的数据集
PROCESSED_DATA_PATH = os.path.join(DATA_SAVE_DIR, f"{MODEL_PREFIX}_processed_data.csv")

# 保存数据处理
if not os.path.exists(PROCESSED_DATA_PATH):
    p.dataframe.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"处理后的数据已保存至: {PROCESSED_DATA_PATH}")
else:
    print(f"使用已保存的处理数据: {PROCESSED_DATA_PATH}")
    # 如果需要直接加载，可以使用以下代码:
    # p.dataframe = pd.read_csv(PROCESSED_DATA_PATH)

### Split traning dataset

train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)
print(f"len(train.tic.unique()): {len(train.tic.unique())}")

print(f"train.tic.unique(): {train.tic.unique()}")

print(f"train.head(): {train.head()}")

print(f"train.shape: {train.shape}")

stock_dimension = len(train.tic.unique())
state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

### Train

# 定义模型保存路径 - 移除此行的MODEL_PREFIX定义，因为已经在上面定义了
DDPG_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, f"{MODEL_PREFIX}_ddpg.zip")
A2C_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, f"{MODEL_PREFIX}_a2c.zip")

# 定义模型参数
DDPG_PARAMS = {
    "batch_size": 256,
    "buffer_size": 50000,
    "learning_rate": 0.0005,
    "action_noise": "normal",
}
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))

# 保存训练参数信息
model_params = {
    "ticker_list": ticker_list,
    "train_start_date": TRAIN_START_DATE,
    "train_end_date": TRAIN_END_DATE,
    "indicators": config.INDICATORS,
    "ddpg_params": DDPG_PARAMS,
    "policy_kwargs": POLICY_KWARGS
}
params_path = os.path.join(TRAINED_MODEL_DIR, f"{MODEL_PREFIX}_params.json")
import json
with open(params_path, 'w') as f:
    json.dump(model_params, f, indent=4, default=str)

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 100000,
    "buy_cost_pct": 6.87e-5,
    "sell_cost_pct": 1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": True,
    "hundred_each_trade": True,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)

## DDPG
print(f"未找到已保存的DDPG模型，开始训练新模型...")
agent = DRLAgent(env=e_train_gym)
# DDPG_PARAMS 和 POLICY_KWARGS 已在前面定义
print(f"未找到已保存的DDPG模型，开始训练新模型...")
agent = DRLAgent(env=e_train_gym)
DDPG_PARAMS = {
    "batch_size": 256,
    "buffer_size": 50000,
    "learning_rate": 0.0005,
    "action_noise": "normal",
}
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
model_ddpg = agent.get_model(
    "ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS
)

trained_ddpg = agent.train_model(
    model=model_ddpg, tb_log_name="ddpg", total_timesteps=10000
)

# 保存训练好的模型
trained_ddpg.save(DDPG_MODEL_PATH)
print(f"DDPG模型已保存至: {DDPG_MODEL_PATH}")

## A2C
# 检查是否有已保存的A2C模型
if os.path.exists(A2C_MODEL_PATH):
    print(f"加载已存在的A2C模型: {A2C_MODEL_PATH}")
    from stable_baselines3 import A2C
    trained_a2c = A2C.load(A2C_MODEL_PATH)
else:
    print(f"未找到已保存的A2C模型，开始训练新模型...")
    agent = DRLAgent(env=e_train_gym)
    model_a2c = agent.get_model("a2c")

    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=50000
    )

    # 保存训练好的模型
    trained_a2c.save(A2C_MODEL_PATH)
    print(f"A2C模型已保存至: {A2C_MODEL_PATH}")

### Trade

trade = p.data_split(p.dataframe, TRADE_START_DATE, TRADE_END_DATE)

# 在回测开始前添加检查
print(f"\n交易日期范围: {TRADE_START_DATE} 到 {TRADE_END_DATE}")
print(f"交易数据点数量: {len(trade)}")
if len(trade) < 10:  # 假设需要至少10个交易日
    print(f"警告: 交易数据点数量太少 ({len(trade)}), 可能导致统计结果不可靠")

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    "buy_cost_pct": 6.87e-5,
    "sell_cost_pct": 1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": False,
    "hundred_each_trade": True,
}
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, environment=e_trade_gym
)

df_actions.to_csv("action.csv", index=False)
print(f"df_actions: {df_actions}")

### Backtest

# matplotlib inline
plotter = ReturnPlotter(df_account_value, trade, TRADE_START_DATE, TRADE_END_DATE)

# 获取基准数据并确保时间维度一致
baseline_df = plotter.get_baseline("399300")
account_dates = pd.to_datetime(df_account_value['date'])
# 修改这行，使用 baseline_df 中实际存在的 'time' 列
baseline_dates = baseline_df['time']  # 而不是 pd.to_datetime(baseline_df['date'])

# 找到共同的日期范围
common_dates = set(account_dates).intersection(set(baseline_dates))
common_dates = sorted(list(common_dates))

# 过滤数据，仅保留共同日期的数据
df_account_value = df_account_value[account_dates.isin(common_dates)].reset_index(drop=True)
baseline_df = baseline_df[baseline_dates.isin(common_dates)].reset_index(drop=True)

# 更新绘图器的账户数据
plotter.account_value = df_account_value

# 然后绘图
plotter.plot()

# matplotlib inline
# # ticket: SSE 50：000016
# plotter.plot("000016")

#### Use pyfolio

# CSI 300
baseline_df = plotter.get_baseline("399300")
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym)

# 确保日期维度匹配
account_dates = pd.to_datetime(df_account_value['date'])
baseline_dates = baseline_df['time']  # 修改这里，使用正确的列名

# 找到共同日期范围
common_dates = set(account_dates).intersection(set(baseline_dates))
common_dates = sorted(list(common_dates))

# 过滤数据
df_account_value = df_account_value[account_dates.isin(common_dates)].reset_index(drop=True)
baseline_df = baseline_df[baseline_dates.isin(common_dates)].reset_index(drop=True)

# 确保两个DataFrame都有time列用于get_return方法
if 'date' in df_account_value.columns and 'time' not in df_account_value.columns:
    df_account_value['time'] = pd.to_datetime(df_account_value['date'])

if 'dt' in baseline_df.columns and 'time' not in baseline_df.columns:
    baseline_df['time'] = pd.to_datetime(baseline_df['dt'])

# 在调用 get_return 前添加调试代码
print("\n==============数据检查==============")
print(f"df_account_value 列名: {df_account_value.columns.tolist()}")
print(f"df_account_value['account_value'] 前5个值: {df_account_value['account_value'].head()}")
print(f"df_account_value['account_value'] 含NaN: {df_account_value['account_value'].isna().sum()}")
print(f"df_account_value['time'] 前5个值: {df_account_value['time'].head()}")

print(f"baseline_df 列名: {baseline_df.columns.tolist()}")
print(f"baseline_df['close'] 前5个值: {baseline_df['close'].head()}")
print(f"baseline_df['close'] 含NaN: {baseline_df['close'].isna().sum()}")
print(f"baseline_df['time'] 前5个值: {baseline_df['time'].head()}")
print(f"数据点数量: {len(df_account_value)}")

# 计算收益率
daily_return = plotter.get_return(df_account_value)
print(f"daily_return 前5个值: {daily_return.head()}")
print(f"daily_return 含NaN: {daily_return.isna().sum()}/{len(daily_return)}")

daily_return_base = plotter.get_return(baseline_df, value_col_name="close")
print(f"daily_return_base 前5个值: {daily_return_base.head()}")
print(f"daily_return_base 含NaN: {daily_return_base.isna().sum()}/{len(daily_return_base)}")

# 在计算性能指标前检查收益率有效性
if daily_return.isna().all() or daily_return_base.isna().all():
    print("警告: 收益率序列全为NaN，无法计算性能指标")
else:
    # 删除NaN值以避免统计问题
    daily_return = daily_return.dropna()
    daily_return_base = daily_return_base.dropna()

    if len(daily_return) < 2 or len(daily_return_base) < 2:
        print(f"警告: 有效收益率数据点太少 (策略: {len(daily_return)}, 基准: {len(daily_return_base)})")
    else:
        perf_func = timeseries.perf_stats
        perf_stats_all = perf_func(
            returns=daily_return,
            factor_returns=daily_return_base,
            positions=None,
            transactions=None,
            turnover_denom="AGB",
        )
        print("==============DRL Strategy Stats===========")
        print(f"perf_stats_all: {perf_stats_all}")

daily_return = plotter.get_return(df_account_value)
daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=daily_return_base,
    factor_returns=daily_return_base,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print("==============Baseline Strategy Stats===========")

print(f"perf_stats_all: {perf_stats_all}")

# 获取最新交易信号
latest_action = df_actions.iloc[-1]
print("\n==============最新交易信号==============")
for stock, action in latest_action.items():
    if stock != 'date':
        signal = "买入" if action > 0 else "卖出" if action < 0 else "持有"
        strength = abs(action)
        print(f"{stock}: {signal} (信号强度: {strength:.4f})")
