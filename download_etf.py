"""ETF 轮动项目：数据下载。

只取一组主流宽基 + 行业 ETF（约 30 只），覆盖 A 股主要风格。
数据源：东方财富前复权日线。

用法：
    python download_etf.py
"""
import os
import time

import pandas as pd
import requests

from config_etf import DATA_DIR, END_DATE, ETF_LIST, START_DATE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def secid(symbol: str) -> str:
    return ("1." if symbol.startswith(("5", "6")) else "0.") + symbol


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101, "fqt": 1,
        "beg": start.replace("-", ""), "end": end.replace("-", ""),
        "lmt": 100000,
    }
    last = None
    for k in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            data = (r.json().get("data") or {}).get("klines") or []
            if data:
                break
        except Exception as e:                       # noqa: BLE001
            last = e
            data = []
        time.sleep(0.5 * (2 ** k))
    if not data:
        if last:
            raise last
        return pd.DataFrame()

    recs = []
    for line in data:
        p = line.split(",")
        recs.append({
            "date": p[0], "open": float(p[1]), "close": float(p[2]),
            "high": float(p[3]), "low": float(p[4]),
            "volume": float(p[5]), "amount": float(p[6]) / 1e4,   # 万元
        })
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    return df.sort_values("date").reset_index(drop=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames, bad = [], []
    for i, (sym, name) in enumerate(ETF_LIST, 1):
        try:
            d = fetch_one(sym, START_DATE, END_DATE)
            if len(d):
                frames.append(d)
                print(f"[{i}/{len(ETF_LIST)}] {sym} {name:<14} {len(d):4} 行 "
                      f"{d['date'].min().date()} ~ {d['date'].max().date()}", flush=True)
            else:
                bad.append((sym, name))
        except Exception as e:                       # noqa: BLE001
            bad.append((sym, name))
            print(f"[{i}/{len(ETF_LIST)}] {sym} {name} 失败 {type(e).__name__}", flush=True)
        time.sleep(0.2)

    if not frames:
        raise SystemExit("没有拿到数据")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    out = os.path.join(DATA_DIR, "etf.csv.gz")
    panel.to_csv(out, index=False, compression="gzip")
    print(f"[panel] {panel['symbol'].nunique()} 只 / {len(panel)} 行 -> {out}")
    if bad:
        print(f"[warn] 失败 {len(bad)} 只: {bad}")


if __name__ == "__main__":
    main()
