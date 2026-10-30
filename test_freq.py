"""测试不同调仓频率在 1000 块本金下的净收益。

最低佣金的成本按绝对金额算，不按比例：
    每笔成本 = max(交易金额 * 费率, 5 元)
持仓 N 只时，一次换仓最多 2N 笔。
"""
import pandas as pd

import backtest_etf as bt
import config_etf as C
import metrics_etf as M

PRINCIPAL = 1000.0
MIN_COMM = 5.0
RATE = C.COMMISSION + C.SLIPPAGE

CASES = [
    ("月度", None, 3),
    ("季度", [1, 4, 7, 10], 2),
    ("半年", [6, 12], 2),
    ("半年", [6, 12], 3),
    ("年度", [12], 1),
    ("年度", [12], 2),
]


def main():
    panel = pd.read_csv("data/etf.csv.gz", parse_dates=["date"], dtype={"symbol": str})
    print(f"{'频率':<6}{'持仓':<4}{'毛年化':>9}{'调仓次数':>9}{'年成本':>9}{'净年化':>9}")
    print("-" * 50)

    for label, months, k in CASES:
        res, trades = bt.run(panel, top_k=k, rebalance_months=months)
        nav = res["nav"]
        gross = M.ann_return(nav)
        years = len(nav) / 252

        # 按绝对金额估最低佣金成本：每次换仓按 2K 笔算（全换）
        per_year = len(trades) / years
        cost_per_trade = 2 * k * MIN_COMM
        annual_cost = per_year * cost_per_trade / PRINCIPAL
        net = gross - annual_cost

        print(f"{label:<6}{k:<4}{gross*100:>8.2f}%{len(trades):>9}{annual_cost*100:>8.2f}%{net*100:>8.2f}%")


if __name__ == "__main__":
    main()
