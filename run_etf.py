"""ETF 轮动：主流程。

用法：
    python run_etf.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_etf as bt
import config_etf as C
import metrics_etf as M

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAME = dict(C.ETF_LIST)


def load():
    panel = pd.read_csv(os.path.join(C.DATA_DIR, "etf.csv.gz"),
                        parse_dates=["date"], dtype={"symbol": str})
    return panel


def md_table(df, index_name="指标", pct_mode="auto"):
    PCT = {"累计收益", "年化收益", "年化波动", "最大回撤", "日胜率"}
    cols = [str(c) for c in df.columns]
    head = "| " + index_name + " | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for idx, row in df.iterrows():
        is_pct = pct_mode == "all" or (pct_mode == "auto" and str(idx) in PCT)
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("-" if not np.isfinite(v) else
                             (f"{v*100:.2f}%" if is_pct else f"{v:.3f}"))
            else:
                cells.append(str(v))
        rows.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def main():
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    panel = load()
    print(f"[data] {panel['symbol'].nunique()} 只 / {len(panel)} 行")

    res, trades = bt.run(panel)
    nav = res["nav"]
    print(f"[backtest] 净值 {nav.iloc[-1]:.3f}，调仓 {len(trades)} 次")

    # 基准：沪深300ETF
    close = bt.pivot(panel, "close")
    bench = close["510300"].reindex(nav.index).ffill()
    bench_nav = bench / bench.dropna().iloc[0]

    split = pd.Timestamp(C.IN_SAMPLE_END)
    segs = {"样本内": nav[nav.index <= split],
            "样本外": nav[nav.index > split],
            "全期": nav}
    summary = pd.DataFrame({k: M.summarize(v) for k, v in segs.items() if len(v) > 2})

    bsegs = {"样本内": bench_nav[bench_nav.index <= split],
             "样本外": bench_nav[bench_nav.index > split],
             "全期": bench_nav}
    bench_summary = pd.DataFrame({k: M.summarize(v) for k, v in bsegs.items() if len(v) > 2})

    yearly = pd.DataFrame({"策略": M.yearly(nav), "沪深300ETF": M.yearly(bench_nav.dropna())})
    yearly.index.name = "年份"

    # ---------- 参数扫描 ----------
    scan = []
    for w in [10, 20, 60]:
        for k in [1, 2, 3, 5]:
            r, _ = bt.run(panel, mom_window=w, top_k=k)
            n = r["nav"]
            oos = n[n.index > split]
            scan.append({"动量窗口": w, "持仓数": k,
                         "年化": M.ann_return(n), "夏普": M.sharpe(n),
                         "最大回撤": M.max_dd(n), "样本外年化": M.ann_return(oos)})
            print(f"  window={w:<3} top={k:<2} 年化 {scan[-1]['年化']:.2%} "
                  f"夏普 {scan[-1]['夏普']:.2f} 样本外 {scan[-1]['样本外年化']:.2%}", flush=True)
    scan_df = pd.DataFrame(scan).set_index(["动量窗口", "持仓数"])

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(nav.index, nav, label="ETF 轮动", linewidth=1.7, color="#c0392b")
    ax.plot(bench_nav.index, bench_nav, label="沪深300ETF", linewidth=1.2,
            color="#7f8c8d", alpha=0.9)
    ax.axvline(split, color="#2980b9", linestyle="--", linewidth=1)
    ax.set_title("ETF 轮动 vs 沪深300ETF（含成本）")
    ax.legend()
    ax.grid(alpha=0.25)

    dd = nav / nav.cummax() - 1
    axes[1].fill_between(dd.index, dd * 100, 0, color="#c0392b", alpha=0.35)
    axes[1].set_title("回撤（%）")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORT_DIR, "nav.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(yearly))
    w = 0.38
    ax.bar(x - w / 2, yearly["策略"] * 100, w, label="策略", color="#c0392b")
    ax.bar(x + w / 2, yearly["沪深300ETF"] * 100, w, label="沪深300ETF", color="#7f8c8d")
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("分年度收益（%）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORT_DIR, "yearly.png"), dpi=140)
    plt.close(fig)

    # ---------- 报告 ----------
    lines = ["# ETF 轮动策略\n",
             f"- 股票池：{panel['symbol'].nunique()} 只宽基 + 行业 ETF",
             f"- 信号：月末按「{C.MOM_WINDOW} 日动量 / 波动率」排序，取前 {C.TOP_K} 只等权",
             f"- 空仓规则：最强动量 < 0 时持现金" if C.CASH_WHEN_NEGATIVE else "- 始终满仓",
             f"- 成交：月末信号，次日开盘",
             f"- 成本：佣金 {C.COMMISSION*1e4:.1f}bp + 滑点 {C.SLIPPAGE*1e4:.1f}bp，ETF 免印花税\n",
             "## 绩效\n", md_table(summary),
             "\n## 沪深300ETF 基准\n", md_table(bench_summary),
             "\n## 分年度\n", md_table(yearly, index_name="年份", pct_mode="all"),
             "\n## 参数敏感性\n", md_table(scan_df, index_name="窗口 / 持仓数"),
             "\n## 图\n", "![净值](nav.png)\n", "![年度](yearly.png)\n"]
    with open(os.path.join(C.REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + md_table(summary))
    print(f"\n[report] {C.REPORT_DIR}/")


if __name__ == "__main__":
    main()
