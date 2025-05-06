import copy
import os
import time
import warnings

warnings.filterwarnings("ignore")
from typing import List

import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

        # self.dataframe.columns = [
        #     "time",
        #     "open",
        #     "close",
        #     "high",
        #     "low",
        #     "volume",
        #     "amount",
        #     "amplitude",
        #     "pct_chg",
        #     "change",
        #     "turnover",
        #     "tic",
        # ]

        self.dataframe.rename(columns={
            "日期": "time",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover"
        }, inplace=True)

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
    """

    def __init__(self, df_account_value, df_trade, start_date, end_date):
        self.start = start_date
        self.end = end_date
        self.trade = df_trade
        self.df_account_value = df_account_value
        self.df_account_value.rename(columns={"date": "time"}, inplace=True)

    def get_baseline(self, ticket):
        # 使用 akshare 获取历史数据
        df = ak.stock_zh_a_hist(
            symbol=ticket,
            start_date=self.start.replace('-', ''),
            end_date=self.end.replace('-', ''),
            adjust="qfq"
        )
        # 将列名重命名以匹配原有格式
        df.rename(columns={"日期": "dt"}, inplace=True)
        # 删除将索引赋值给dt列的代码
        # 保留原始的日期数据
        df.sort_values(axis=0, by="dt", ascending=True, inplace=True)
        df["time"] = pd.to_datetime(df["dt"])  # 让pandas自动识别日期格式
        return df

    def plot(self, baseline_ticket=None):
        """
        Plot cumulative returns over time.
        use baseline_ticket to specify stock you want to use for comparison
        (default: equal weighted returns)
        """
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}
        if (baseline_ticket):
            # 使用指定ticket作为baseline
            baseline_df = self.get_baseline(baseline_ticket)
            baseline_date_list = baseline_df.time.dt.strftime("%Y-%m-%d").tolist()
            df_date_list = self.df_account_value.time.tolist()
            df_account_value = self.df_account_value[
                self.df_account_value.time.isin(baseline_date_list)
            ]
            baseline_df = baseline_df[baseline_df.time.isin(df_date_list)]
            baseline = baseline_df.close.tolist()
            baseline_label = tic2label.get(baseline_ticket, baseline_ticket)
            ours = df_account_value.account_value.tolist()
        else:
            # 均等权重
            all_date = self.trade.time.unique().tolist()
            baseline = []
            for day in all_date:
                day_close = self.trade[self.trade["time"] == day].close.tolist()
                avg_close = sum(day_close) / len(day_close)
                baseline.append(avg_close)
            ours = self.df_account_value.account_value.tolist()

        ours = self.pct(ours)
        baseline = self.pct(baseline)

        days_per_tick = (
            60  # you should scale this variable accroding to the total trading days
        )
        time = list(range(len(ours)))
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(self.df_account_value.head())
        datetimes = self.df_account_value.time.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=time, y=ours, mode='lines', name='DDPG Agent', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=time, y=baseline, mode='lines', name=baseline_label, line=dict(color='grey')))
        fig.update_layout(
            title="Cumulative Returns",
            xaxis=dict(
                tickmode='array',
                tickvals=[i * days_per_tick for i in range(len(ticks))],
                ticktext=ticks,
                tickfont=dict(size=7)
            ),
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            legend=dict(x=0, y=1)
        )
        fig.show()
        fig.write_image(f"plot_{baseline_ticket}.png")

    def plot_all(self):
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        # time lists
        # algorithm time list
        df_date_list = self.df_account_value.time.tolist()

        # 399300 time list
        csi300_df = self.get_baseline("399300")
        csi300_date_list = csi300_df.time.dt.strftime("%Y-%m-%d").tolist()

        # 000016 time list
        sh50_df = self.get_baseline("000016")
        sh50_date_list = sh50_df.time.dt.strftime("%Y-%m-%d").tolist()

        # find intersection
        all_date = sorted(
            list(set(df_date_list) & set(csi300_date_list) & set(sh50_date_list))
        )

        # filter data
        csi300_df = csi300_df[csi300_df.time.isin(all_date)]
        baseline_300 = csi300_df.close.tolist()
        baseline_label_300 = tic2label["399300"]

        sh50_df = sh50_df[sh50_df.time.isin(all_date)]
        baseline_50 = sh50_df.close.tolist()
        baseline_label_50 = tic2label["000016"]

        # 均等权重
        baseline_equal_weight = []
        for day in all_date:
            day_close = self.trade[self.trade["time"] == day].close.tolist()
            avg_close = sum(day_close) / len(day_close)
            baseline_equal_weight.append(avg_close)

        df_account_value = self.df_account_value[
            self.df_account_value.time.isin(all_date)
        ]
        ours = df_account_value.account_value.tolist()

        ours = self.pct(ours)
        baseline_300 = self.pct(baseline_300)
        baseline_50 = self.pct(baseline_50)
        baseline_equal_weight = self.pct(baseline_equal_weight)

        days_per_tick = (
            60  # you should scale this variable accroding to the total trading days
        )
        time = list(range(len(ours)))
        datetimes = self.df_account_value.time.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=time, y=ours, mode='lines', name='DDPG Agent', line=dict(color='darkorange')))
        fig.add_trace(go.Scatter(x=time, y=baseline_equal_weight, mode='lines', name=baseline_label, line=dict(color='cornflowerblue')))
        fig.add_trace(go.Scatter(x=time, y=baseline_300, mode='lines', name=baseline_label_300, line=dict(color='lightgreen')))
        fig.add_trace(go.Scatter(x=time, y=baseline_50, mode='lines', name=baseline_label_50, line=dict(color='silver')))
        fig.update_layout(
            title="Cumulative Returns",
            xaxis=dict(
                tickmode='array',
                tickvals=[i * days_per_tick for i in range(len(ticks))],
                ticktext=ticks,
                tickfont=dict(size=7)
            ),
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            legend=dict(x=0, y=1)
        )
        fig.show()
        fig.write_image("./plot_all.png")

    def pct(self, l):
        """Get percentage"""
        base = l[0]
        return [x / base for x in l]

    def get_return(self, df, value_col_name="account_value"):
        df = copy.deepcopy(df)
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")
        df.set_index("time", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df["daily_return"], index=df.index)
