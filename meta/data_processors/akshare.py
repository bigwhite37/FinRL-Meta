import copy
import os
import time
import warnings
import matplotlib.pyplot as plt
import akshare as ak

warnings.filterwarnings("ignore")
from typing import List

import pandas as pd
from tqdm import tqdm

import stockstats
import talib
from meta.data_processors._base import _Base

import akshare as ak  # pip install akshare


class Akshare(_Base):
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        start_date = self.transfer_date(start_date)
        end_date = self.transfer_date(end_date)

        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

        if "adj" in kwargs.keys():
            self.adj = kwargs["adj"]
            print(f"Using {self.adj} method.")
        else:
            self.adj = ""

        if "period" in kwargs.keys():
            self.period = kwargs["period"]
        else:
            self.period = "daily"

    def get_data(self, id) -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=id,
            period=self.time_interval,
            start_date=self.start_date,
            end_date=self.end_date,
            adjust=self.adj,
        )

    def download_data(
        self, ticker_list: List[str], save_path: str = "./data/dataset.csv"
    ):
        """
        `pd.DataFrame`
            7 columns: A tick symbol, time, open, high, low, close and volume
            for the specified stock ticker
        """
        assert self.time_interval in [
            "daily",
            "weekly",
            "monthly",
        ], "Not supported currently"

        self.ticker_list = ticker_list

        self.dataframe = pd.DataFrame()
        for i in tqdm(ticker_list, total=len(ticker_list)):
            nonstandard_id = self.transfer_standard_ticker_to_nonstandard(i)
            df_temp = self.get_data(nonstandard_id)
            df_temp["tic"] = i
            # df_temp = self.get_data(i)
            self.dataframe = pd.concat([self.dataframe, df_temp])
            # self.dataframe = self.dataframe.append(df_temp)
            # print("{} ok".format(i))
            time.sleep(0.25)

        self.dataframe.columns = [
            "time",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "change",
            "turnover",
            "tic",
        ]

        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.dataframe = self.dataframe[
            ["tic", "time", "open", "high", "low", "close", "volume"]
        ]
        # self.dataframe.loc[:, 'tic'] = pd.DataFrame((self.dataframe['tic'].tolist()))
        self.dataframe["time"] = pd.to_datetime(
            self.dataframe["time"], format="%Y-%m-%d"
        )
        self.dataframe["day"] = self.dataframe["time"].dt.dayofweek
        self.dataframe["time"] = self.dataframe.time.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

    def data_split(self, df, start, end, target_date_col="time"):
        """
        split the dataset into training or testing using time
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        # "600000.XSHG" -> "600000"
        # "000612.XSHE" -> "000612"
        # "600000.SH" -> "600000"
        # "000612.SZ" -> "000612"
        if "." in ticker:
            n, alpha = ticker.split(".")
            # assert alpha in ["XSHG", "XSHE"], "Wrong alpha"
        return n

    def transfer_date(self, time: str) -> str:
        if "-" in time:
            time = "".join(time.split("-"))
        elif "." in time:
            time = "".join(time.split("."))
        elif "/" in time:
            time = "".join(time.split("/"))
        return time


class ReturnPlotter:
    """
    An easy-to-use plotting tool to plot cumulative returns over time.
    Baseline supports equal weighting(default) and any stocks you want to use for comparison.
    使用 akshare 接口获取基准数据
    """

    def __init__(self, df_account_value, df_trade, start_date, end_date):
        self.start = start_date
        self.end = end_date
        self.trade = df_trade
        self.df_account_value = df_account_value.copy()

        # 检查并确保有 'time' 列
        if 'date' in self.df_account_value.columns and 'time' not in self.df_account_value.columns:
            self.df_account_value.rename(columns={'date': 'time'}, inplace=True)

    def get_baseline(self, ticket):
        """获取基准指数数据，使用 akshare API

        参数:
            ticket (str): 指数代码，例如 "399300" 代表沪深300指数

        返回:
            pandas.DataFrame: 包含基准数据的数据框
        """
        # 处理指数代码格式
        if ticket == "399300":
            index_code = "sz399300"  # akshare 使用的沪深300代码
        elif ticket == "000016":
            index_code = "sh000016"  # akshare 使用的上证50代码
        else:
            index_code = ticket

        try:
            # 使用 akshare 获取指数日线数据
            print(f"正在获取 {index_code} 指数数据...")
            df = ak.stock_zh_index_daily(symbol=index_code)

            # 转换数据格式以匹配原有代码
            if not df.empty:
                # 重命名列
                df = df.rename(columns={
                    'date': 'dt',
                    'volume': 'volume',
                    'close': 'close'
                })

                # 格式转换
                df['time'] = pd.to_datetime(df['dt'])

                # 过滤日期范围
                df = df[(df['time'] >= pd.to_datetime(self.start)) &
                         (df['time'] <= pd.to_datetime(self.end))]

                # 排序并重置索引
                df.sort_values(by='time', ascending=True, inplace=True)
                df.reset_index(drop=True, inplace=True)

                return df
            else:
                print(f"无法获取 {index_code} 数据")
                return None
        except Exception as e:
            print(f"获取 {index_code} 数据时出错: {e}")
            return None

    def plot(self, baseline_ticket=None):
        """
        Plot cumulative returns over time.
        use baseline_ticket to specify stock you want to use for comparison
        (default: equal weighted returns)
        """
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        if baseline_ticket:
            # 使用指定ticket作为baseline
            baseline_df = self.get_baseline(baseline_ticket)
            if baseline_df is None or baseline_df.empty:
                print(f"无法获取 {baseline_ticket} 数据，使用等权重投资组合")
                baseline_ticket = None
            else:
                baseline_date_list = baseline_df.time.dt.strftime("%Y-%m-%d").tolist()
                df_date_list = self.df_account_value.time.tolist()
                df_account_value = self.df_account_value[
                    self.df_account_value.time.isin(baseline_date_list)
                ]
                baseline_df = baseline_df[baseline_df.time.isin(df_date_list)]
                baseline = baseline_df.close.tolist()
                baseline_label = tic2label.get(baseline_ticket, baseline_ticket)
                ours = df_account_value.account_value.tolist()

        if not baseline_ticket:
            # 使用等权重投资组合作为基准
            all_date = self.trade.time.unique().tolist()
            baseline = []
            for day in all_date:
                day_close = self.trade[self.trade["time"] == day].close.tolist()
                avg_close = sum(day_close) / len(day_close)
                baseline.append(avg_close)
            ours = self.df_account_value.account_value.tolist()

        # 确保数据长度一致
        min_len = min(len(ours), len(baseline))
        ours = ours[:min_len]
        baseline = baseline[:min_len]

        # 计算百分比变化
        ours = self.pct(ours)
        baseline = self.pct(baseline)

        # 绘图
        days_per_tick = 60  # 根据总交易天数调整此变量
        time = list(range(len(ours)))
        datetimes = self.df_account_value.time.tolist()[:len(ours)]
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]

        plt.figure(figsize=(12, 6))
        plt.title("Cumulative Returns")
        plt.plot(time, ours, label="RL Agent", color="green")
        plt.plot(time, baseline, label=baseline_label, color="grey")
        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f"plot_{baseline_ticket or 'equal_weight'}.png", dpi=300)
        plt.show()

    def plot_all(self):
        """绘制与多个基准的比较图"""
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        # 获取算法账户价值的日期列表
        df_date_list = self.df_account_value.time.tolist()

        # 获取沪深300指数数据
        csi300_df = self.get_baseline("399300")
        if csi300_df is not None and not csi300_df.empty:
            csi300_date_list = csi300_df.time.dt.strftime("%Y-%m-%d").tolist()
        else:
            csi300_date_list = []

        # 获取上证50指数数据
        sh50_df = self.get_baseline("000016")
        if sh50_df is not None and not sh50_df.empty:
            sh50_date_list = sh50_df.time.dt.strftime("%Y-%m-%d").tolist()
        else:
            sh50_date_list = []

        # 找到交集日期
        all_date = sorted(
            list(set(df_date_list) & set(csi300_date_list) & set(sh50_date_list))
        )

        if not all_date:
            print("没有共同的交易日期，无法绘制比较图")
            return

        # 过滤数据
        csi300_df = csi300_df[csi300_df.time.isin(all_date)]
        baseline_300 = csi300_df.close.tolist()
        baseline_label_300 = tic2label["399300"]

        sh50_df = sh50_df[sh50_df.time.isin(all_date)]
        baseline_50 = sh50_df.close.tolist()
        baseline_label_50 = tic2label["000016"]

        # 均等权重基准
        baseline_equal_weight = []
        for day in all_date:
            day_close = self.trade[self.trade["time"] == day].close.tolist()
            avg_close = sum(day_close) / len(day_close)
            baseline_equal_weight.append(avg_close)

        # 过滤账户价值数据
        df_account_value = self.df_account_value[
            self.df_account_value.time.isin(all_date)
        ]
        ours = df_account_value.account_value.tolist()

        # 计算百分比变化
        ours = self.pct(ours)
        baseline_300 = self.pct(baseline_300)
        baseline_50 = self.pct(baseline_50)
        baseline_equal_weight = self.pct(baseline_equal_weight)

        # 绘图
        days_per_tick = 60
        time = list(range(len(ours)))
        datetimes = df_account_value.time.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]

        plt.figure(figsize=(12, 6))
        plt.title("Cumulative Returns Comparison")
        plt.plot(time, ours, label="RL Agent", color="darkorange")
        plt.plot(time, baseline_equal_weight, label=baseline_label, color="cornflowerblue")
        plt.plot(time, baseline_300, label=baseline_label_300, color="lightgreen")
        plt.plot(time, baseline_50, label=baseline_label_50, color="silver")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("plot_all.png", dpi=300)
        plt.show()

    def pct(self, l):
        """计算百分比变化"""
        if not l:
            return []
        base = l[0]
        if base == 0:
            return [0] * len(l)
        return [x / base for x in l]

    def get_return(self, df, value_col_name="account_value"):
        """计算日收益率"""
        df = copy.deepcopy(df)
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df["daily_return"], index=df.index)
