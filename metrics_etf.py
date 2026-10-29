"""绩效指标（精简版）。"""
import numpy as np
import pandas as pd

TD = 252


def daily(nav):
    return nav.pct_change(fill_method=None).dropna()


def ann_return(nav):
    if len(nav) < 2:
        return np.nan
    years = len(nav) / TD
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1


def ann_vol(nav):
    return daily(nav).std(ddof=1) * np.sqrt(TD)


def sharpe(nav, rf=0.0):
    r = daily(nav)
    if r.std(ddof=1) == 0:
        return np.nan
    return (r - rf / TD).mean() / r.std(ddof=1) * np.sqrt(TD)


def max_dd(nav):
    return (nav / nav.cummax() - 1).min()


def calmar(nav):
    m = max_dd(nav)
    return np.nan if not m else ann_return(nav) / abs(m)


def summarize(nav):
    return pd.Series({
        "累计收益": nav.iloc[-1] / nav.iloc[0] - 1,
        "年化收益": ann_return(nav),
        "年化波动": ann_vol(nav),
        "夏普比率": sharpe(nav),
        "最大回撤": max_dd(nav),
        "Calmar": calmar(nav),
        "日胜率": (daily(nav) > 0).mean(),
    })


def yearly(nav):
    y = nav.resample("YE").last()
    out, prev = {}, nav.iloc[0]
    for ts, v in y.items():
        out[ts.year] = v / prev - 1
        prev = v
    return pd.Series(out)
