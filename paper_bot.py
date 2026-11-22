#!/usr/bin/env python3
"""
Paper trading bot v1 - many signals, each tested alone, then combined (confluence).
FAKE MONEY ONLY. No API keys, no orders. Uses Binance public futures data.

Setup:   pip install pandas numpy requests
Usage:   python paper_bot.py backtest   # test every signal on the last ~15 days
         python paper_bot.py live       # paper-trade live, checks every 15m candle close
         python paper_bot.py once       # one check, then exit (used by GitHub Actions)
         python paper_bot.py report     # stats from your live paper trades
"""
import sys, time, json, os
import numpy as np, pandas as pd, requests

SYMBOLS = ["XAUUSDT", "SOLUSDT"]          # add "BTCUSDT" later if you want
INTERVAL = "15m"
MIN_SCORE = 3        # how many signals must agree before a trade
SL_ATR = 1.5         # stop loss = 1.5 x ATR
RR = 2.0             # take profit = 2R
FEE = 0.0005         # 0.05% per side (Binance taker), counted in results
STATE, LOG = "paper_state.json", "paper_trades.csv"


# ---------- data ----------
YAHOO = {"XAUUSDT": "GC=F"}      # gold fallback when Binance blocks the server (e.g. GitHub Actions)


def fetch_yahoo(sym):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{YAHOO[sym]}",
                     params=dict(interval="15m", range="5d"),
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(dict(t=np.array(res["timestamp"], dtype=float) * 1000,
                           open=q["open"], high=q["high"], low=q["low"],
                           close=q["close"], volume=q["volume"]))
    df = df.dropna(subset=["open", "high", "low", "close"]).fillna({"volume": 0.0})
    return df.iloc[:-1].reset_index(drop=True)          # drop the unfinished candle


def fetch(sym, limit=1500):
    urls = ["https://fapi.binance.com/fapi/v1/klines",          # futures (preferred)
            "https://data-api.binance.vision/api/v3/klines"]     # spot mirror (fallback)
    last = None
    for u in urls:
        try:
            r = requests.get(u, params=dict(symbol=sym, interval=INTERVAL, limit=limit), timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["t", "open", "high", "low", "close", "volume"]
            df = df.astype(float)
            return df.iloc[:-1].reset_index(drop=True)      # drop the unfinished candle
        except Exception as e:
            last = e
    if sym in YAHOO:
        try:
            print(f"[{sym}] Binance blocked, using Yahoo {YAHOO[sym]} instead")
            return fetch_yahoo(sym)
        except Exception as e:
            last = e
    raise RuntimeError(f"{sym}: could not fetch data ({last})")


def prep(df):
    c = df.close
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    df["bbu"], df["bbl"] = m + 2 * s, m - 2 * s
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vma"] = df.volume.rolling(20).mean()
    return df


# ---------- zone signals (FVG / order block): first retest of a fresh zone ----------
def retest(df, zones, maxage):
    out = np.zeros(len(df))
    c, l, h = df.close.values, df.low.values, df.high.values
    for form, lo, hi, d in zones:
        for i in range(form + 1, min(form + 1 + maxage, len(df))):
            if d == 1:
                if c[i] < lo: break                       # zone broken
                if l[i] <= hi:                            # price came back into zone
                    if c[i] > lo and out[i] == 0: out[i] = 1
                    break
            else:
                if c[i] > hi: break
                if h[i] >= lo:
                    if c[i] < hi and out[i] == 0: out[i] = -1
                    break
    return out


def fvg_zones(df):
    h, l, atr, z = df.high.values, df.low.values, df.atr.values, []
    for j in range(2, len(df)):
        if l[j] > h[j - 2] and l[j] - h[j - 2] > 0.3 * atr[j]:
            z.append((j, h[j - 2], l[j], 1))
        if h[j] < l[j - 2] and l[j - 2] - h[j] > 0.3 * atr[j]:
            z.append((j, h[j], l[j - 2], -1))
    return z


def ob_zones(df):
    o, c, h, l, atr, z = (df.open.values, df.close.values, df.high.values,
                          df.low.values, df.atr.values, [])
    for j in range(1, len(df) - 1):
        k = j + 1
        big = abs(c[k] - o[k]) > 1.5 * atr[k]
        if c[j] < o[j] and c[k] > o[k] and big and c[k] > h[j]:
            z.append((k, l[j], h[j], 1))                  # last red candle before up-impulse
        if c[j] > o[j] and c[k] < o[k] and big and c[k] < l[j]:
            z.append((k, l[j], h[j], -1))
    return z


# ---------- all signals: +1 buy, -1 sell, 0 nothing ----------
def signals(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    up_w, lo_w = h - np.maximum(o, c), np.minimum(o, c) - l
    S = {}
    S["ema_trend"] = np.where((df.ema20 > df.ema50) & (c > df.ema20), 1,
                     np.where((df.ema20 < df.ema50) & (c < df.ema20), -1, 0))
    S["rsi"] = np.where(df.rsi < 30, 1, np.where(df.rsi > 70, -1, 0))
    S["bollinger"] = np.where(c < df.bbl, 1, np.where(c > df.bbu, -1, 0))
    S["volume"] = np.where(df.volume > 2 * df.vma, np.sign(c - o), 0)
    po, pc = o.shift(), c.shift()
    S["engulfing"] = np.where((pc < po) & (c > o) & (c >= po) & (o <= pc), 1,
                     np.where((pc > po) & (c < o) & (c <= po) & (o >= pc), -1, 0))
    S["pin_bar"] = np.where((lo_w >= 2 * body) & (up_w <= 0.3 * rng), 1,
                   np.where((up_w >= 2 * body) & (lo_w <= 0.3 * rng), -1, 0))
    o2, c2, b2, b1 = o.shift(2), c.shift(2), body.shift(2), body.shift(1)
    mid2 = (o2 + c2) / 2
    star_ok = (b2 > 0.8 * df.atr) & (b1 < 0.3 * b2)
    S["star"] = np.where(star_ok & (c2 < o2) & (c > o) & (c > mid2), 1,
                np.where(star_ok & (c2 > o2) & (c < o) & (c < mid2), -1, 0))
    S["fvg"] = retest(df, fvg_zones(df), 20)
    S["order_block"] = retest(df, ob_zones(df), 30)
    return {k: np.nan_to_num(np.asarray(v, dtype=float)).astype(int) for k, v in S.items()}


# ---------- paper trade simulator ----------
def simulate(df, sig):
    trades, i, n = [], 60, len(df)
    hi, lo, cl, atr = df.high.values, df.low.values, df.close.values, df.atr.values
    while i < n - 1:
        d = sig[i]
        if d == 0 or np.isnan(atr[i]):
            i += 1
            continue
        entry, dist = cl[i], SL_ATR * atr[i]
        sl, tp = entry - d * dist, entry + d * dist * RR
        res, j = None, i + 1
        for j in range(i + 1, n):
            hit_sl = lo[j] <= sl if d == 1 else hi[j] >= sl
            hit_tp = hi[j] >= tp if d == 1 else lo[j] <= tp
            if hit_sl: res = -1.0; break                  # if both in one candle, assume SL first
            if hit_tp: res = RR; break
        if res is None:
            break
        trades.append(res - 2 * FEE * entry / dist)       # result in R, after fees
        i = j + 1
    return trades


def stats(r):
    if not len(r): return "   0 trades"
    r = np.array(r)
    return f"{len(r):4d} trades | win {100 * (r > 0).mean():4.1f}% | avg {r.mean():+.2f}R | total {r.sum():+.1f}R"


def backtest():
    for sym in SYMBOLS:
        df = prep(fetch(sym))
        S = signals(df)
        print(f"\n=== {sym} | {len(df)} candles of {INTERVAL} ===")
        for name, s in S.items():
            print(f"{name:12s}", stats(simulate(df, s)))
        score = sum(S.values())
        conf = np.where(abs(score) >= MIN_SCORE, np.sign(score), 0).astype(int)
        print(f"{'CONFLUENCE':12s}", stats(simulate(df, conf)), f"(needs {MIN_SCORE}+ signals)")
    print("\nPast results are NOT a promise. Small samples lie. Trust it only after weeks of live paper data.")


# ---------- live paper mode ----------
def log_trade(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if new: f.write("symbol,dir,entry_time,exit_time,entry,sl,tp,r,signals\n")
        f.write(",".join(str(x) for x in row) + "\n")


def live_once():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    last = state.setdefault("_last", {})      # last candle already processed, per symbol
    for sym in SYMBOLS:
        try:
            df = prep(fetch(sym, 400))
        except Exception as e:
            print(f"[{sym}] data error, skipping this round: {e}")
            continue
        S = signals(df)
        n = len(df)
        last_t = last.get(sym)
        # first run: latest candle only. After that: every candle we haven't seen yet
        todo = [n - 1] if last_t is None else [j for j in range(n) if df.t.iat[j] > last_t]
        for j in todo:
            tr = state.get(sym)
            if tr:                                        # manage the open paper trade
                if df.t.iat[j] <= tr["t"]:
                    continue
                d, hi, lo = tr["dir"], df.high.iat[j], df.low.iat[j]
                hit_sl = lo <= tr["sl"] if d == 1 else hi >= tr["sl"]
                hit_tp = hi >= tr["tp"] if d == 1 else lo <= tr["tp"]
                if hit_sl or hit_tp:
                    dist = abs(tr["entry"] - tr["sl"])
                    r = (-1.0 if hit_sl else RR) - 2 * FEE * tr["entry"] / dist
                    log_trade([sym, d, tr["t"], float(df.t.iat[j]), tr["entry"], tr["sl"], tr["tp"],
                               round(r, 3), "|".join(tr["signals"])])
                    print(f"[{sym}] CLOSED {'SL' if hit_sl else 'TP'} {r:+.2f}R")
                    state[sym] = None
            else:                                         # look for a new entry
                score = sum(v[j] for v in S.values())
                if abs(score) >= MIN_SCORE and not np.isnan(df.atr.iat[j]):
                    d = int(np.sign(score))
                    entry, dist = float(df.close.iat[j]), float(SL_ATR * df.atr.iat[j])
                    state[sym] = dict(dir=d, entry=entry, sl=entry - d * dist, tp=entry + d * dist * RR,
                                      t=float(df.t.iat[j]), signals=[k for k, v in S.items() if v[j] == d])
                    print(f"[{sym}] OPEN {'LONG' if d == 1 else 'SHORT'} @ {entry} "
                          f"SL {state[sym]['sl']:.4f} TP {state[sym]['tp']:.4f} | {state[sym]['signals']}")
        last[sym] = float(df.t.iat[n - 1])
        print(f"[{sym}] checked {len(todo)} new candle(s), latest score "
              f"{sum(v[n - 1] for v in S.values())}, {'trade open' if state.get(sym) else 'no open trade'}")
    json.dump(state, open(STATE, "w"))


def live():
    print("Paper mode running. Ctrl+C to stop.")
    while True:
        try: live_once()
        except Exception as e: print("error:", e)
        time.sleep(900 - (time.time() % 900) + 5)         # wait for next 15m candle close


def report():
    if not os.path.exists(LOG): return print("No closed paper trades yet.")
    df = pd.read_csv(LOG)
    print("ALL      ", stats(df.r.values))
    for sym, g in df.groupby("symbol"):
        print(f"{sym:9s}", stats(g.r.values))
    rows = [(s, r) for r, ss in zip(df.r, df.signals) for s in str(ss).split("|")]
    per = pd.DataFrame(rows, columns=["signal", "r"])
    print("\nTrades each signal took part in:")
    for sig, g in per.groupby("signal"):
        print(f"{sig:12s}", stats(g.r.values))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    {"backtest": backtest, "live": live, "once": live_once, "report": report}[cmd]()
