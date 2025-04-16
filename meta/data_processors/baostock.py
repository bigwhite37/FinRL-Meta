from typing import List

import baostock as bs
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

try:
    import pandas_market_calendars as tc
except:
    print(
        "Cannot import pandas_market_calendars.",
        "If you are using python>=3.7, please install it.",
    )
    import trading_calendars as tc

    print("Use trading_calendars instead for yahoofinance processor..")
# from basic_processor import _Base
from meta.data_processors._base import _Base
from meta.data_processors._base import calc_time_zone

from meta.config import (
    TIME_ZONE_SHANGHAI,
    TIME_ZONE_USEASTERN,
    TIME_ZONE_PARIS,
    TIME_ZONE_BERLIN,
    TIME_ZONE_JAKARTA,
    TIME_ZONE_SELFDEFINED,
    USE_TIME_ZONE_SELFDEFINED,
    BINANCE_BASE_URL,
)


class Baostock(_Base):
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        # 添加时间间隔格式转换
        if self.time_interval == "1d":
            self.time_interval = "d"
        elif self.time_interval == "1w":
            self.time_interval = "w"
        elif self.time_interval == "1M":
            self.time_interval = "m"
        elif self.time_interval == "5m":
            self.time_interval = "5"
        elif self.time_interval == "15m":
            self.time_interval = "15"
        elif self.time_interval == "30m":
            self.time_interval = "30"
        elif self.time_interval == "60m":
            self.time_interval = "60"

    # 日k线、周k线、月k线，以及5分钟、15分钟、30分钟和60分钟k线数据
    # ["5m", "15m", "30m", "60m", "1d", "1w", "1M"]
    def download_data(
        self, ticker_list: List[str], save_path: str = "./data/dataset.csv"
    ):
        lg = bs.login()
        print("baostock login respond error_code:" + lg.error_code)
        print("baostock login respond  error_msg:" + lg.error_msg)

        self.time_zone = calc_time_zone(
            ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED
        )
        self.dataframe = pd.DataFrame()
        for ticker in ticker_list:
            nonstandrad_ticker = self.transfer_standard_ticker_to_nonstandard(ticker)
            # All supported: "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
            rs = bs.query_history_k_data_plus(
                nonstandrad_ticker,
                "date,code,open,high,low,close,volume",
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.time_interval,
                adjustflag="3",
            )

            print("baostock download_data respond error_code:" + rs.error_code)
            print("baostock download_data respond  error_msg:" + rs.error_msg)

            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            df.loc[:, "code"] = [ticker] * df.shape[0]
            self.dataframe = pd.concat([self.dataframe, df])
        self.dataframe = self.dataframe.sort_values(by=["date", "code"]).reset_index(
            drop=True
        )
        bs.logout()

        self.dataframe.open = self.dataframe.open.astype(float)
        self.dataframe.high = self.dataframe.high.astype(float)
        self.dataframe.low = self.dataframe.low.astype(float)
        self.dataframe.close = self.dataframe.close.astype(float)
        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

    def get_trading_days(self, start, end):
        lg = bs.login()
        print("baostock login respond error_code:" + lg.error_code)
        print("baostock login respond  error_msg:" + lg.error_msg)
        result = bs.query_trade_dates(start_date=start, end_date=end)
        bs.logout()
        return result

    # "600000.XSHG" -> "sh.600000"
    # "000612.XSHE" -> "sz.000612"
    # "sh.600000" -> "sh.600000" (already standard)
    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        if ticker.startswith("sh.") or ticker.startswith("sz."):
            # Already in non-standard format, return as is
            return ticker

        parts = ticker.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid ticker format: {ticker}")

        n, alpha = parts
        if alpha == "XSHG" or alpha == "SH":
            nonstandard_ticker = "sh." + n
        elif alpha == "XSHE" or alpha == "SZ":
            nonstandard_ticker = "sz." + n
        else:
            raise ValueError(f"Unsupported exchange code: {alpha}")
        return nonstandard_ticker

import copy
import matplotlib.pyplot as plt
import pandas as pd


class ReturnPlotter:
    """
    An easy-to-use plotting tool to plot cumulative returns over time.
    Baseline supports equal weighting(default) and any stocks you want to use for comparison.
    使用 BaoStock 接口获取基准数据
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
        """获取基准指数数据，使用 BaoStock API

        参数:
            ticket (str): 指数代码，例如 "399300" 代表沪深300指数

        返回:
            pandas.DataFrame: 包含基准数据的数据框
        """
        # BaoStock登录
        lg = bs.login()
        print("BaoStock login respond error_code:" + lg.error_code)
        print("BaoStock login respond error_msg:" + lg.error_msg)

        # 处理指数代码格式
        if ticket == "399300":
            index_code = "sz.399300"  # 沪深300指数
        elif ticket == "000016":
            index_code = "sh.000016"  # 上证50指数
        else:
            if ticket.startswith("0") and len(ticket) == 6:
                index_code = "sz." + ticket  # 深证指数
            elif ticket.startswith("1") or ticket.startswith("5"):
                index_code = "sz." + ticket  # 深证指数
            else:
                index_code = "sh." + ticket  # 上证指数

        try:
            # 获取指数K线数据
            rs = bs.query_history_k_data_plus(
                index_code,
                "date,open,high,low,close,volume",
                start_date=self.start,
                end_date=self.end,
                frequency="d",  # 日k线
                adjustflag="3"  # 不复权
            )

            print("BaoStock query_history_k_data_plus respond error_code:" + rs.error_code)
            print("BaoStock query_history_k_data_plus respond error_msg:" + rs.error_msg)

            # 处理结果集
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            # 转换为DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)

            # 类型转换
            if not df.empty:
                df["open"] = df["open"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["close"] = df["close"].astype(float)
                df["volume"] = df["volume"].astype(float)

                # 添加时间列
                df.rename(columns={"date": "dt"}, inplace=True)
                df["time"] = pd.to_datetime(df["dt"])
                df.sort_values(by="time", ascending=True, inplace=True)
                df.reset_index(drop=True, inplace=True)

            # 登出
            bs.logout()
            return df

        except Exception as e:
            print(f"获取指数数据时发生错误: {e}")
            bs.logout()  # 确保错误发生时也能正确登出
            return pd.DataFrame()  # 返回空DataFrame

    def plot(self, baseline_ticket=None):
        """
        绘制累计收益曲线。
        使用 baseline_ticket 指定要比较的基准指数
        (默认: 等权重投资组合)
        """
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        if baseline_ticket:
            # 使用指定ticket作为baseline
            baseline_df = self.get_baseline(baseline_ticket)

            if baseline_df.empty:
                print(f"无法获取 {baseline_ticket} 数据，使用等权重投资组合")
                baseline_ticket = None
            else:
                baseline_date_list = baseline_df.time.dt.strftime("%Y-%m-%d").tolist()
                df_date_list = self.df_account_value.time.tolist()

                # 确保日期对齐
                df_account_value = self.df_account_value[
                    self.df_account_value.time.isin(baseline_date_list)
                ]
                baseline_df = baseline_df[baseline_df.time.isin(df_date_list)]

                # 检查是否有足够的对齐数据
                if len(baseline_df) < 2:
                    print(f"对齐后 {baseline_ticket} 数据不足，使用等权重投资组合")
                    baseline_ticket = None
                else:
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

        # 绘图设置
        days_per_tick = 60  # 根据总交易天数调整此变量
        time = list(range(len(ours)))
        datetimes = self.df_account_value.time.tolist()[:min_len]
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

        save_name = f"plot_{baseline_ticket or 'equal_weight'}.png"
        plt.savefig(save_name)
        print(f"图表已保存为 {save_name}")
        plt.show()

    def plot_all(self):
        """同时绘制多个基准的比较图"""
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        # 获取账户数据日期列表
        df_date_list = self.df_account_value.time.tolist()

        # 获取沪深300指数数据
        print("\n获取沪深300指数数据...")
        csi300_df = self.get_baseline("399300")
        if not csi300_df.empty:
            csi300_date_list = csi300_df.time.dt.strftime("%Y-%m-%d").tolist()
        else:
            print("无法获取沪深300指数数据")
            csi300_date_list = []

        # 获取上证50指数数据
        print("\n获取上证50指数数据...")
        sh50_df = self.get_baseline("000016")
        if not sh50_df.empty:
            sh50_date_list = sh50_df.time.dt.strftime("%Y-%m-%d").tolist()
        else:
            print("无法获取上证50指数数据")
            sh50_date_list = []

        # 找出共同的日期
        common_dates = set(df_date_list)
        if csi300_date_list:
            common_dates = common_dates.intersection(set(csi300_date_list))
        if sh50_date_list:
            common_dates = common_dates.intersection(set(sh50_date_list))
        all_date = sorted(list(common_dates))

        if len(all_date) < 2:
            print("没有足够的共同交易日期，无法绘制比较图")
            return

        # 过滤和准备数据
        df_account_value = self.df_account_value[
            self.df_account_value.time.isin(all_date)
        ]
        ours = df_account_value.account_value.tolist()

        # 均等权重基准
        baseline_equal_weight = []
        for day in all_date:
            day_close = self.trade[self.trade["time"] == day].close.tolist()
            if day_close:  # 确保有数据
                avg_close = sum(day_close) / len(day_close)
                baseline_equal_weight.append(avg_close)
            else:
                # 如果没有数据，使用前一天的值或默认值
                baseline_equal_weight.append(
                    baseline_equal_weight[-1] if baseline_equal_weight else 0
                )

        # 处理指数数据
        lines_to_plot = [(ours, "RL Agent", "darkorange")]

        if baseline_equal_weight:
            lines_to_plot.append(
                (baseline_equal_weight, baseline_label, "cornflowerblue")
            )

        if not csi300_df.empty:
            csi300_df = csi300_df[csi300_df.time.isin(all_date)]
            baseline_300 = csi300_df.close.tolist()
            lines_to_plot.append(
                (baseline_300, tic2label["399300"], "lightgreen")
            )

        if not sh50_df.empty:
            sh50_df = sh50_df[sh50_df.time.isin(all_date)]
            baseline_50 = sh50_df.close.tolist()
            lines_to_plot.append(
                (baseline_50, tic2label["000016"], "silver")
            )

        # 计算百分比变化并绘图
        plt.figure(figsize=(12, 6))
        plt.title("Cumulative Returns Comparison")

        days_per_tick = 60
        time_axis = list(range(len(ours)))
        datetimes = df_account_value.time.tolist()
        ticks = [tick for t, tick in zip(time_axis, datetimes) if t % days_per_tick == 0]

        for values, label, color in lines_to_plot:
            # 确保长度一致
            values = values[:len(ours)]
            # 计算百分比变化
            values_pct = self.pct(values)
            plt.plot(time_axis, values_pct, label=label, color=color)

        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig("plot_all.png")
        print("所有基准比较图已保存为 plot_all.png")
        plt.show()

    def pct(self, l):
        """计算百分比变化"""
        if not l:
            return []
        base = l[0]
        if base == 0:
            base = 1e-5  # 避免除以0错误
        return [x / base for x in l]

    def get_return(self, df, value_col_name="account_value"):
        """计算日收益率"""
        import copy
        df = copy.deepcopy(df)

        # 检查值列是否存在
        if value_col_name not in df.columns:
            print(f"错误: '{value_col_name}' 列不存在！可用列: {df.columns.tolist()}")
            return pd.Series(dtype=float)  # 返回空Series

        # 计算日收益率
        df["daily_return"] = df[value_col_name].pct_change(1)

        # 移除第一行NaN
        df = df.dropna(subset=["daily_return"])

        if len(df) == 0:
            print(f"警告: 没有有效的收益率数据")
            return pd.Series(dtype=float)

        # 尝试找到可用的时间列
        time_col = None
        for col in ['time', 'date', 'dt']:
            if col in df.columns:
                time_col = col
                break

        # 如果找到时间列，则设置为索引
        if time_col:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                if hasattr(df.index, 'tz_localize'):
                    df.index = df.index.tz_localize(None)  # 移除时区以避免可能的问题
                return pd.Series(df["daily_return"], index=df.index)
            except Exception as e:
                print(f"设置时间索引时出错: {e}")
                return pd.Series(df["daily_return"])
        else:
            print("未找到时间列，返回无索引的收益率序列")
            return pd.Series(df["daily_return"])
