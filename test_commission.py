"""真实佣金成本测算：最低 5 元 vs 免五。

回测引擎里的成本是纯比例的 —— 每笔 333 元 × 万2.5 = 0.08 元。
真实券商普遍有最低收费（常见 5 元/笔），1000 块本金下这一条直接决定策略能不能做。

这个脚本按真实账户规则重放回测的调仓记录：
    - 买入按 100 份取整，买不满的钱挂着
    - 佣金 = max(金额 × 费率, 最低收费)
    - 滑点单边 5bp
    - 一手都买不起的标的直接跳过

用法：
    python test_commission.py                          # 默认 1000 块、万2.5 + 最低5元
    python test_commission.py --principal 10000        # 换本金
    python test_commission.py --rate 5e-05 --min-comm 0  # 换成查到真实佣金后
"""
import argparse
import sys

import pandas as pd

import backtest_etf as bt
import config_etf as C

LOT = 100          # ETF 最小交易单位：100 份
TOP_K = 3


def load():
    panel = pd.read_csv("data/etf.csv.gz", parse_dates=["date"], dtype={"symbol": str})
    open_ = panel.pivot_table(index="date", columns="symbol", values="open", aggfunc="last")
    close = panel.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    _, trades = bt.run(panel, top_k=TOP_K)
    return open_, close, trades


def simulate(trades, open_, close, rate, min_comm, principal=1000.0, rebal=True):
    """按真实账户规则重放调仓记录。

    rebal=False 时不动已有持仓，只在标的更换时交易 —— 省掉每月等权再平衡的摩擦。
    """
    cash, hold = principal, {}
    fees, n_trades, missed, idle = 0.0, 0, 0, []

    for _, t in trades.iterrows():
        today = t["date"]
        syms = [s for s in str(t["symbols"]).split(",") if s]

        # 1) 卖出不在目标名单里的
        for s in list(hold):
            if s not in syms:
                px = open_.loc[today, s] * (1 - C.SLIPPAGE)
                amt = hold[s] * px
                fee = max(amt * rate, min_comm)
                cash += amt - fee
                fees += fee
                n_trades += 1
                del hold[s]

        total = cash + sum(h * open_.loc[today, s] for s, h in hold.items())
        if not syms:                     # 空仓信号
            idle.append(cash / total if total else 0.0)
            continue

        # 2) 等权分配，剔除一手都买不起的
        target = total / len(syms)
        ok = [s for s in syms
              if open_.loc[today, s] * LOT * (1 + C.SLIPPAGE) <= target]
        if ok:
            missed += len(syms) - len(ok)
        else:                            # 全都买不起，退而求其次买最便宜的一只
            ok = [min(syms, key=lambda s: open_.loc[today, s])]
            missed += len(syms) - 1

        # 3) 按手数取整下单
        target = total / len(ok)
        for s in ok:
            px = open_.loc[today, s] * (1 + C.SLIPPAGE)
            cur = hold.get(s, 0)
            if not rebal and cur > 0:
                continue
            delta = target - cur * px
            lots = int(abs(delta) // (px * LOT))
            if not lots:
                continue
            amt = lots * LOT * px
            fee = max(amt * rate, min_comm)
            if delta > 0:
                # 现金不够就少买几手，不要整笔跳过（否则闲置现金会放大噪声）
                while lots > 0:
                    amt = lots * LOT * px
                    fee = max(amt * rate, min_comm)
                    if amt + fee <= cash:
                        break
                    lots -= 1
                if lots:
                    amt = lots * LOT * px
                    fee = max(amt * rate, min_comm)
                    cash -= amt + fee
                    hold[s] = cur + lots * LOT
                    fees += fee
                    n_trades += 1
            else:
                cash += amt - fee
                hold[s] = cur - lots * LOT
                fees += fee
                n_trades += 1
                if hold[s] <= 0:
                    del hold[s]

        after = cash + sum(h * open_.loc[today, s] for s, h in hold.items())
        idle.append(cash / after if after else 0.0)

    last = close.index[-1]
    final = cash + sum(h * close.loc[last, s] for s, h in hold.items())
    years = len(close) / 252
    return {
        "final": final,
        "ann": (final / principal) ** (1 / years) - 1,
        "fees": fees,
        "n_trades": n_trades,
        "idle": sum(idle) / len(idle) if idle else 0.0,
        "missed": missed,
    }


def show_policies(trades, open_, close, principal):
    print(f"\n【一】不同佣金政策 —— 本金 {principal:,.0f} 元")
    print("%-30s%10s%9s%10s%9s" % ("政策", "终值", "年化", "总费用", "笔数"))
    print("-" * 70)
    policies = [
        ("默认 万2.5 + 最低5元", 0.00025, 5.0),
        ("万1 + 最低5元", 0.0001, 5.0),
        ("万2.5 免五", 0.00025, 0.0),
        ("万1 免五", 0.0001, 0.0),
        ("万0.5 免五（渠道极限）", 0.00005, 0.0),
    ]
    for label, rate, mc in policies:
        r = simulate(trades, open_, close, rate, mc, principal)
        print("%-28s%10.0f%8.1f%%%10.0f%9d"
              % (label, r["final"], r["ann"] * 100, r["fees"], r["n_trades"]))
    print("\n  参考：回测（纯比例成本、资金无约束）年化 11.43%")


def show_missed(trades, open_, close, principal):
    """哪些调仓因为一手都买不起而无法跟随信号。"""
    target = principal / TOP_K
    last_px = close.iloc[-1]
    hard = sorted([s for s in last_px.index
                   if last_px[s] * LOT * (1 + C.SLIPPAGE) > target])
    if not hard:
        return

    print(f"\n【二】买不起的标的 —— 每只分到 {target:,.0f} 元，一手超过就不行")
    for s in hard:
        n = sum(1 for x in trades["symbols"] if s in str(x).split(","))
        print("  %s  最新价 %6.3f  一手 %7.1f 元  信号选中 %d 次（%.0f%%）"
              % (s, last_px[s], last_px[s] * LOT, n, n / len(trades) * 100))

    n_any = sum(1 for x in trades["symbols"] if set(hard) & set(str(x).split(",")))
    print("  合计 %d 次调仓（%.0f%%）会缺标，持仓偏离策略"
          % (n_any, n_any / len(trades) * 100))


def show_principal(trades, open_, close, rate, min_comm):
    print(f"\n【三】本金门槛 —— 费率 {rate*1e4:.1f}bp，最低收费 {min_comm:,.0f} 元")
    print("%-12s%9s%10s%11s%9s" % ("本金", "年化", "总费用", "费用/本金", "闲置"))
    print("-" * 55)
    for p0 in [1000, 2000, 3000, 5000, 10000, 30000, 100000]:
        r = simulate(trades, open_, close, rate, min_comm, p0)
        print("%-12s%8.1f%%%10.0f%10.0f%%%8.0f%%"
              % (f"{p0:,}", r["ann"] * 100, r["fees"],
                 r["fees"] / p0 * 100, r["idle"] * 100))
    print("\n  躺平对照：沪深300ETF 买着不动，同期年化 7.67%")


def show_rebal(trades, open_, close, rate, min_comm):
    print(f"\n【四】每月等权再平衡 vs 只在换标的时交易")
    print("%-12s%12s%12s%9s" % ("本金", "每月再平衡", "只换标的", "提升"))
    print("-" * 48)
    for p0 in [1000, 3000, 5000, 10000]:
        a = simulate(trades, open_, close, rate, min_comm, p0, rebal=True)["ann"]
        b = simulate(trades, open_, close, rate, min_comm, p0, rebal=False)["ann"]
        print("%-12s%11.1f%%%11.1f%%%8.1f%%"
              % (f"{p0:,}", a * 100, b * 100, (b - a) * 100))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="ETF 轮动的真实佣金成本测算")
    ap.add_argument("--principal", type=float, default=1000.0, help="本金，默认 1000")
    ap.add_argument("--rate", type=float, default=C.COMMISSION, help="佣金费率，默认万2.5")
    ap.add_argument("--min-comm", type=float, default=5.0, help="每笔最低收费，默认 5 元")
    a = ap.parse_args()

    open_, close, trades = load()
    print(f"回测区间 {close.index[0]:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}，"
          f"{len(trades)} 次调仓，持 {TOP_K} 只")

    show_policies(trades, open_, close, a.principal)
    show_missed(trades, open_, close, a.principal)
    show_principal(trades, open_, close, a.rate, a.min_comm)
    show_rebal(trades, open_, close, a.rate, a.min_comm)
    print()


if __name__ == "__main__":
    main()
