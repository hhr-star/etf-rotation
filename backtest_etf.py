"""ETF 轮动回测引擎。

策略：月末按「风险调整动量」排序，取前 K 只等权持有，次一交易日开盘换仓。
      可选：最强动量 < 0 时空仓（持现金）。

时序：月末收盘算信号 -> 次日开盘成交 -> 持有到下个月末。
"""
import numpy as np
import pandas as pd

import config_etf as C


def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")


def month_ends(dates):
    s = pd.Series(dates, index=pd.DatetimeIndex(dates))
    return list(s.groupby([s.index.year, s.index.month]).last().sort_values())


def run(panel, mom_window=None, top_k=None, vol_adjust=None, cash_neg=None):
    mom_window = mom_window or C.MOM_WINDOW
    top_k = top_k or C.TOP_K
    vol_adjust = C.USE_VOL_ADJUST if vol_adjust is None else vol_adjust
    cash_neg = C.CASH_WHEN_NEGATIVE if cash_neg is None else cash_neg

    open_ = pivot(panel, "open")
    close = pivot(panel, "close")
    dates = list(close.index)

    mom = close / close.shift(mom_window) - 1
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(C.VOL_WINDOW, min_periods=max(5, C.VOL_WINDOW // 2)).std()
    score = (mom / vol.replace(0, np.nan)) if vol_adjust else mom

    mends = month_ends(dates)
    nav, weights = 1.0, pd.Series(dtype=float)
    rows, trades = [], []

    for i, today in enumerate(dates):
        prev = dates[i - 1] if i > 0 else None
        cost = 0.0

        if prev in mends:
            sc = score.loc[prev].dropna()
            sc = sc[sc.index.intersection(open_.loc[today].dropna().index)]
            if len(sc):
                top = sc.nlargest(min(top_k, len(sc)))
                if cash_neg and mom.loc[prev, top.index].max() <= 0:
                    target = pd.Series(dtype=float)          # 空仓
                else:
                    target = pd.Series(1.0 / len(top), index=top.index)
            else:
                target = weights.copy()

            all_sym = weights.index.union(target.index)
            old = weights.reindex(all_sym).fillna(0.0)
            new = target.reindex(all_sym).fillna(0.0)
            buy = float((new - old).clip(lower=0).sum())
            sell = float((old - new).clip(lower=0).sum())
            cost = buy * (C.COMMISSION + C.SLIPPAGE) + sell * (C.COMMISSION + C.SLIPPAGE + C.STAMP_TAX)

            nav *= (1 - cost)
            weights = target
            trades.append({"date": today, "n": len(target), "cost": cost,
                           "symbols": ",".join(f"{s}" for s in target.index)})

            px = (close.loc[today] / open_.loc[today]).reindex(weights.index).fillna(1.0)
            r = float((weights * (px - 1)).sum()) if len(weights) else 0.0
        elif i == 0 or len(weights) == 0:
            r = 0.0
        else:
            px = (close.loc[today] / close.loc[prev]).reindex(weights.index)
            ok = px.notna()
            if ok.sum() == 0:
                r = 0.0
            else:
                w, rr = weights[ok], px[ok] - 1
                r = float((w * rr).sum())
                nv = w * (1 + rr)
                tot = nv.sum() + (1 - weights.sum())
                if tot > 0:
                    weights = nv / tot

        nav *= (1 + r)
        rows.append({"date": today, "nav": nav, "ret": r, "cost": cost})

    df = pd.DataFrame(rows).set_index("date")
    return df, pd.DataFrame(trades)
