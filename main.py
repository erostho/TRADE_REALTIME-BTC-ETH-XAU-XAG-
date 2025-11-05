# -*- coding: utf-8 -*-
"""
bot_trade_pro_cache.py

Multi-symbol, cache-aware analyst bot for OKX/Binance (Futures/Swap):
- Timeframes: 1H/2H/4H + 1D (daily only on new daily close)
- Cache OHLCV in memory AND persist to disk (pickle) to survive restarts
- Batch-rotation across symbols to avoid API rate limits on Render
- Mid-interval updates (every N minutes) to assess if the current 1H bar is behaving as predicted
- Technicals: EMA(20/50), RSI(14), MACD(12,26,9), BBWidth, ATR%
- Pattern hints: shooting_star / hammer / engulfing
- Break→Retest detection (up & down)
- Action suggestions: keep/partial close/move SL/add position based on live price vs Entry/SL/TP
- Telegram reporting

Env Vars:
  EXCHANGE             = okx | binance  (default: okx)
  SYMBOLS              = BTC/USDT:USDT,ETH/USDT:USDT     (OKX swap format) / (Binance futures: BTC/USDT,ETH/USDT)
  POLL_SECONDS         = 60
  MID_UPDATE_MIN       = 30            (mid-bar check interval in minutes)
  TIMEFRAME_1H         = 1h
  TIMEFRAME_2H         = 2h
  TIMEFRAME_4H         = 4h
  TIMEFRAME_1D         = 1d
  BATCH_SIZE           = 2
  TELEGRAM_BOT_TOKEN   = <token>
  TELEGRAM_CHAT_ID     = <chat_id>
  OKX defaultType swap; Binance defaultType future.

Run:
  pip install ccxt pandas numpy python-dotenv requests
  python bot_trade_pro_cache.py
"""

import os
import time
import math
import pickle
import requests
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os, requests
import ccxt

load_dotenv()

# ===========================
# Config
# ===========================
EXCHANGE_NAME = os.getenv("EXCHANGE", "okx").lower()
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT").split(",")
SYMBOLS = [s.strip() for s in SYMBOLS if s.strip()]

TF_1H = os.getenv("TIMEFRAME_1H", "1h")
TF_2H = os.getenv("TIMEFRAME_2H", "2h")
TF_4H = os.getenv("TIMEFRAME_4H", "4h")
TF_1D = os.getenv("TIMEFRAME_1D", "1d")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
MID_UPDATE_MIN = int(os.getenv("MID_UPDATE_MIN", "30"))
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "2")))
TELE_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELE_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Technical Parameters
EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
BB_LEN = 20
BB_K = 2.0
ATR_LEN = 14
SR_TOL_PCT = 0.15 / 100.0
TP1_PCT, TP2_PCT = 0.6/100.0, 1.2/100.0
SL_PAD_PCT = 0.3/100.0

# Cache (RAM + disk)
CACHE_FILE = "ohlcv_cache.pkl"
SAVE_INTERVAL = 300  # seconds
cache_ohlcv = {}     # { (symbol, timeframe): {"last_ts": int, "df": DataFrame} }
last_cache_save = 0

# Track last closed ts per symbol/timeframe and last signals
last_closed_1h = {}
last_closed_2h = {}
last_closed_4h = {}
last_closed_1d = {}
last_mid_update = {}
last_signal = {}

def init_state_structs(symbols):
    global last_closed_1h, last_closed_2h, last_closed_4h, last_closed_1d, last_mid_update, last_signal
    last_closed_1h = {sym: None for sym in symbols}
    last_closed_2h = {sym: None for sym in symbols}
    last_closed_4h = {sym: None for sym in symbols}
    last_closed_1d = {sym: None for sym in symbols}
    last_mid_update = {sym: 0 for sym in symbols}
    last_signal = {sym: None for sym in symbols}

# ================= TELEGRAM SEND =================

def tg_send(text, parse_mode="Markdown"):
    if not TG_TOKEN or not TG_CHAT:
        print("⚠️ TELEGRAM env chưa có, bỏ qua gửi.")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    data = {
        "chat_id": TG_CHAT,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=data, timeout=15)
        print("📩 Telegram:", r.status_code, r.text[:120])
    except Exception as e:
        print("❌ Telegram error:", e)
# ===========================
# ================= TELEGRAM BATCH =================
BATCH_LINES = []

def stage_now():
    """Xác định giai đoạn hiện tại — đầu giờ hay giữa giờ"""
    now = datetime.now(timezone.utc).astimezone()
    m = now.minute
    if m == 0:
        return "close"
    if m == 30:
        return "mid"
    return "other"

def add_line(symbol, timeframe, summary):
    """Gom các dòng text cần gửi"""
    BATCH_LINES.append(f"• *{symbol}* ({timeframe}): {summary}")

def flush_batch():
    """Gửi gộp 1 tin duy nhất"""
    stg = stage_now()
    if stg == "close":
        header = "🕐 *TỔNG HỢP SAU KHI ĐÓNG NẾN 1H*"
    elif stg == "mid":
        header = "⏱️ *CẬP NHẬT GIỮA GIỜ (30’)*"
    else:
        print("⏸️ Không phải giờ gửi, bỏ qua Telegram.")
        return

    if not BATCH_LINES:
        print("⚠️ Không có nội dung để gửi.")
        return

    body = "\n".join(BATCH_LINES)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tg_send(f"{header}\n{body}\n_{now_str}_")
    BATCH_LINES.clear()
      
# Exchange init
# =========================
def make_exchange(name: str):
    import ccxt
    ex = ccxt.okx({
        'enableRateLimit': True,
        # ép kiểu mặc định là swap
        'options': { 'defaultType': 'swap', 'fetchMarkets': ['swap'] }
    })

    # cố gắng load theo options ở trên
    try:
        ex.load_markets()
        return ex
    except Exception as e:
        print(f"⚠️ load_markets() lỗi: {e} -> fallback chỉ lấy SWAP")

    # --- Fallback: tự gọi instruments SWAP và parse thủ công ---
    data = ex.publicGetPublicInstruments({'instType': 'SWAP'})
    rows = data.get('data', [])
    safe = []
    for r in rows:
        base = r.get('baseCcy')
        quote = r.get('quoteCcy')
        # bỏ qua INDEX/OPTION (thiếu base/quote)
        if not base or not quote:
            continue
        safe.append(ex.parse_market(r))
    # đăng ký lại markets chỉ gồm SWAP
    ex.set_markets(safe)
    return ex
ex = make_exchange(EXCHANGE_NAME)

# ===========================
# Utilities
# ===========================
def ts_to_str(ts_ms):
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def telegram_send(msg: str):
    if not TELE_TOKEN or not TELE_CHAT:
        print("[TELE] Skipped (no token/chat).")
        return
    try:
        url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage"
        payload = {"chat_id": TELE_CHAT, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"[TELE] Error: {e}")

def timeframe_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return 86400
    return 0

def fetch_ohlcv_df(symbol, timeframe, limit=300):
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol"])
    return df

def save_cache():
    global cache_ohlcv
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_ohlcv, f)
        print(f"[CACHE] Saved cache to {CACHE_FILE} ({len(cache_ohlcv)} items)")
    except Exception as e:
        print(f"[CACHE] Save error: {e}")

def load_cache():
    global cache_ohlcv
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache_ohlcv = pickle.load(f)
            print(f"[CACHE] Loaded {len(cache_ohlcv)} items from {CACHE_FILE}")
        except Exception as e:
            print(f"[CACHE] Load error: {e}")

def fetch_ohlcv_cached(symbol, timeframe, limit=300):
    key = (symbol, timeframe)
    if key not in cache_ohlcv:
        df = fetch_ohlcv_df(symbol, timeframe, limit)
        cache_ohlcv[key] = {"last_ts": int(df["ts"].iloc[-1]), "df": df}
        return df
    cached_last_ts = cache_ohlcv[key]["last_ts"]
    try:
        df_tail = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
        df_tail = pd.DataFrame(df_tail, columns=["ts","open","high","low","close","vol"])
        if int(df_tail["ts"].iloc[-1]) != cached_last_ts:
            df = fetch_ohlcv_df(symbol, timeframe, limit)
            cache_ohlcv[key] = {"last_ts": int(df["ts"].iloc[-1]), "df": df}
            return df
        else:
            return cache_ohlcv[key]["df"]
    except Exception as e:
        print(f"[CACHE][{symbol} {timeframe}] tail-check error, returning cached: {e}")
        return cache_ohlcv[key]["df"]

# ===========================
# Indicators
# ===========================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(length).mean()
    ma_down = down.rolling(length).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df, length=14):
    tr = true_range(df)
    return tr.rolling(length).mean()

def bbands(series, length=20, k=2.0):
    ma = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=0)
    upper = ma + k*std
    lower = ma - k*std
    width = (upper - lower) / ma * 100.0
    return upper, ma, lower, width

# ===========================
# Patterns & Detection
# ===========================
def last_swing_high(df, lookback=10):
    highs = df["high"].values
    idx = None
    for i in range(len(highs)-2, len(highs)-lookback-2, -1):
        if i-1 < 0: break
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            idx = i
            break
    return None if idx is None else (df["ts"].iloc[idx], df["high"].iloc[idx])

def last_swing_low(df, lookback=10):
    lows = df["low"].values
    idx = None
    for i in range(len(lows)-2, len(lows)-lookback-2, -1):
        if i-1 < 0: break
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            idx = i
            break
    return None if idx is None else (df["ts"].iloc[idx], df["low"].iloc[idx])

def detect_break_retest(df, side="down"):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    sh = last_swing_high(df, lookback=12)
    sl = last_swing_low(df, lookback=12)

    if side == "down" and sl is not None:
        sr = sl[1]
        broke = close[-2] < sr*(1 - SR_TOL_PCT) or low[-2] < sr*(1 - SR_TOL_PCT)
        retest = high[-1] <= sr*(1 + SR_TOL_PCT) and close[-1] <= sr*(1 + SR_TOL_PCT)
        if broke and retest:
            return {"type": "break_retest_down", "sr": sr}
    if side == "up" and sh is not None:
        sr = sh[1]
        broke = close[-2] > sr*(1 + SR_TOL_PCT) or high[-2] > sr*(1 + SR_TOL_PCT)
        retest = low[-1] >= sr*(1 - SR_TOL_PCT) and close[-1] >= sr*(1 - SR_TOL_PCT)
        if broke and retest:
            return {"type": "break_retest_up", "sr": sr}
    return None

def candle_signal(df):
    o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = (h - l) + 1e-12
    body_pct = body / rng

    if upper > 2*body and body_pct < 0.4 and lower < body*0.6 and c < h:
        return "shooting_star"
    if lower > 2*body and body_pct < 0.4 and upper < body*0.6 and c > l:
        return "hammer"

    o2, c2 = df["open"].iloc[-2], df["close"].iloc[-2]
    if (c < o) and (c2 > o2) and (o >= c2) and (c <= o2):
        return "bearish_engulfing"
    if (c > o) and (c2 < o2) and (o <= c2) and (c >= o2):
        return "bullish_engulfing"
    return None

# ===========================
# Analyzer
# ===========================
def analyze_one_tf(df, tf_name):
    df = df.copy()
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    macd_line, macd_signal, macd_hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    _, _, _, bb_w = bbands(df["close"], BB_LEN, BB_K)
    df["bb_width"] = bb_w
    atr_val = atr(df, ATR_LEN)
    df["atr_pct"] = (atr_val / df["close"]) * 100.0

    trend = "down" if df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1] else "up"
    sig_candle = candle_signal(df)

    sh = last_swing_high(df)
    sl = last_swing_low(df)

    comp = (df["bb_width"].iloc[-1] <= 1.0) and (df["atr_pct"].iloc[-1] <= 0.8)

    return {
        "tf": tf_name,
        "price": float(df["close"].iloc[-1]),
        "time": int(df["ts"].iloc[-1]),
        "trend": trend,
        "rsi": float(df["rsi"].iloc[-1]),
        "macd": float(df["macd"].iloc[-1]),
        "macd_signal": float(df["macd_signal"].iloc[-1]),
        "macd_hist": float(df["macd_hist"].iloc[-1]),
        "ema_fast": float(df["ema_fast"].iloc[-1]),
        "ema_slow": float(df["ema_slow"].iloc[-1]),
        "bb_width": float(df["bb_width"].iloc[-1]),
        "atr_pct": float(df["atr_pct"].iloc[-1]),
        "compression": bool(comp),
        "candle": sig_candle,
        "swing_high": None if sh is None else float(sh[1]),
        "swing_low": None if sl is None else float(sl[1]),
        "brk_retest_down": detect_break_retest(df, "down"),
        "brk_retest_up": detect_break_retest(df, "up"),
    }

def pretty_price(p):
    if p >= 1000:
        return f"{p:,.2f}"
    return f"{p:,.2f}"

def build_reco(side, price):
    if side == "sell":
        tp1 = price * (1 - TP1_PCT)
        tp2 = price * (1 - TP2_PCT)
        sl = price * (1 + SL_PAD_PCT)
    else:
        tp1 = price * (1 + TP1_PCT)
        tp2 = price * (1 + TP2_PCT)
        sl = price * (1 - SL_PAD_PCT)
    return {"sl": sl, "tp1": tp1, "tp2": tp2}

def make_report(sym, a1, a2, a4, ad):
    price = a1["price"]
    tstr = ts_to_str(a1["time"])
    lines = []
    lines.append(f"🕒 <b>{sym}</b> • cập nhật {tstr}")
    lines.append(f"Giá hiện tại: <b>{pretty_price(price)}</b>")

    lines.append("\n<b>• Xu hướng & tín hiệu đa khung</b>")
    lines.append(f"1H: trend <b>{a1['trend']}</b>, RSI={a1['rsi']:.1f}, MACD={a1['macd']:.3f}/{a1['macd_signal']:.3f}, "
                 f"BBW={a1['bb_width']:.2f}%"
                 + (f", nến={a1['candle']}" if a1['candle'] else ""))
    lines.append(f"2H: trend <b>{a2['trend']}</b>, RSI={a2['rsi']:.1f}, MACD={a2['macd']:.3f}/{a2['macd_signal']:.3f}, "
                 f"BBW={a2['bb_width']:.2f}%"
                 + (f", nến={a2['candle']}" if a2['candle'] else ""))
    lines.append(f"4H: trend <b>{a4['trend']}</b>, RSI={a4['rsi']:.1f}, MACD={a4['macd']:.3f}/{a4['macd_signal']:.3f}, "
                 f"BBW={a4['bb_width']:.2f}%"
                 + (f", nến={a4['candle']}" if a4['candle'] else ""))
    lines.append(f"1D: trend <b>{ad['trend']}</b>, RSI={ad['rsi']:.1f}, MACD={ad['macd']:.3f}/{ad['macd_signal']:.3f}, "
                 f"BBW={ad['bb_width']:.2f}%"
                 + (f", nến={ad['candle']}" if ad['candle'] else ""))

    if a1["compression"]:
        lines.append("🔧 1H đang <b>nén giá</b> → sắp có cú bung mạnh.")

    bull_cond = (a1["rsi"] > 52 and a1["macd"] > a1["macd_signal"])
    bear_cond = (a1["rsi"] < 48 and a1["macd"] < a1["macd_signal"])

    br_down = a1["brk_retest_down"] or a2["brk_retest_down"] or a4["brk_retest_down"]
    br_up = a1["brk_retest_up"] or a2["brk_retest_up"] or a4["brk_retest_up"]

    reco_text = ""
    decided_side = None

    if br_down and a1["trend"] == "down":
        reco = build_reco("sell", price)
        lines.append(f"\n⚔️ <b>Break→Retest giảm</b> phát hiện quanh {pretty_price(br_down['sr'])} (ưu tiên SELL).")
        reco_text = (f"➡️ SELL {pretty_price(price)} | SL {pretty_price(reco['sl'])} | "
                     f"TP1 {pretty_price(reco['tp1'])} • TP2 {pretty_price(reco['tp2'])}")
        decided_side = "sell"
    elif br_up and a1["trend"] == "up":
        reco = build_reco("buy", price)
        lines.append(f"\n⚔️ <b>Break→Retest tăng</b> phát hiện quanh {pretty_price(br_up['sr'])} (ưu tiên BUY).")
        reco_text = (f"➡️ BUY {pretty_price(price)} | SL {pretty_price(reco['sl'])} | "
                     f"TP1 {pretty_price(reco['tp1'])} • TP2 {pretty_price(reco['tp2'])}")
        decided_side = "buy"
    else:
        if a1["trend"] == "down" and a4["trend"] == "down":
            reco = build_reco("sell", price)
            lines.append("\n📉 Tổng thể <b>bearish</b>; ưu tiên canh <b>SELL khi hồi yếu</b>.")
            reco_text = (f"➡️ SELL {pretty_price(price)} | SL {pretty_price(reco['sl'])} | "
                         f"TP1 {pretty_price(reco['tp1'])} • TP2 {pretty_price(reco['tp2'])}")
            decided_side = "sell"
        elif a1["trend"] == "up" and a4["trend"] == "up":
            reco = build_reco("buy", price)
            lines.append("\n📈 Tổng thể <b>bullish</b>; ưu tiên canh <b>BUY khi điều chỉnh nông</b>.")
            reco_text = (f"➡️ BUY {pretty_price(price)} | SL {pretty_price(reco['sl'])} | "
                         f"TP1 {pretty_price(reco['tp1'])} • TP2 {pretty_price(reco['tp2'])}")
            decided_side = "buy"
        else:
            lines.append("\n⏸ Đa khung chưa đồng pha, chờ tín hiệu rõ hơn.")

    if bear_cond and a1["trend"] == "down":
        lines.append("📉 <b>RSI & MACD đồng thuận giảm</b> → tín hiệu SELL tin cậy.")
    if bull_cond and a1["trend"] == "up":
        lines.append("📈 <b>RSI & MACD đồng thuận tăng</b> → tín hiệu BUY tin cậy.")

    if a1["candle"] in ("shooting_star", "bearish_engulfing") and a1["trend"] == "down":
        lines.append("🧨 1H vừa có nến đảo chiều giảm → xác suất đạp tiếp cao.")
    if a1["candle"] in ("hammer", "bullish_engulfing") and a1["trend"] == "up":
        lines.append("🧨 1H vừa có nến đảo chiều tăng → xác suất bật tiếp cao.")

    if reco_text:
        lines.append(reco_text)

    return "\n".join(lines), decided_side, (reco if reco_text else None)

# ===========================
# Mid-interval management
# ===========================
# ==============================
def mid_update(sym, df1):
    now = time.time()
    if now - last_mid_update.get(sym, 0) < MID_UPDATE_MIN * 60:
        return
    last_mid_update[sym] = now

    price_now = None
    try:
        t = ex.fetch_ticker(sym)
        price_now = float(t["last"])
    except Exception:
        price_now = float(df1["close"].iloc[-1])

    df_temp = df1.copy()
    df_temp.iloc[-1, df_temp.columns.get_loc("close")] = price_now
    a1_live = analyze_one_tf(df_temp, TF_1H)

    msg_lines = [f"⏱️ <b>Giữa kỳ {sym}</b>",
                 f"Giá hiện tại: {pretty_price(price_now)}",
                 f"Trend tạm: {a1_live['trend']}, RSI={a1_live['rsi']:.1f}, MACD={a1_live['macd']:.3f}/{a1_live['macd_signal']:.3f}"]

    sig = last_signal.get(sym)
    advice = None
    if sig:
        side = sig["side"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp1 = sig["tp1"]
        tp2 = sig["tp2"]

        if side == "sell":
            if price_now <= tp1:
                advice = f"🎯 Gần/qua TP1 → chốt 50% & dời SL về {pretty_price(entry)}"
            elif entry * 1.000 <= price_now <= entry * 1.003 and price_now < sl:
                advice = f"📈 Hồi gần Entry → có thể thêm SELL nhỏ (SL {pretty_price(sl)})"
            elif price_now >= sl * 0.995:
                advice = f"⚠️ Sát SL → cân nhắc thoát bớt giảm rủi ro"
            if a1_live['macd'] < a1_live['macd_signal'] and a1_live['rsi'] < 45:
                advice = (advice or "") + "\n📉 Momentum yếu → giữ SELL/đẩy SL gần."
            if a1_live['macd'] > a1_live['macd_signal'] and a1_live['rsi'] > 55:
                advice = (advice or "") + "\n⚠️ Momentum phục hồi → tránh add SELL."
        else:
            if price_now >= tp1:
                advice = f"🎯 Gần/qua TP1 → chốt 50% & dời SL về {pretty_price(entry)}"
            elif entry * 0.997 <= price_now <= entry * 0.999 and price_now > sl:
                advice = f"📉 Hồi gần Entry → có thể thêm BUY nhỏ (SL {pretty_price(sl)})"
            elif price_now <= sl * 1.005:
                advice = f"⚠️ Sát SL → cân nhắc thoát bớt giảm rủi ro"
            if a1_live['macd'] > a1_live['macd_signal'] and a1_live['rsi'] > 55:
                advice = (advice or "") + "\n📈 Momentum tốt → có thể giữ/đẩy SL."
            if a1_live['macd'] < a1_live['macd_signal'] and a1_live['rsi'] < 45:
                advice = (advice or "") + "\n⚠️ Momentum suy yếu → tránh add BUY."

    else:
        advice = "🕓 Chưa có tín hiệu gốc để quản lý."

    if advice:
        msg_lines.append(advice)

    add_line(sym, "Giữa kỳ", text)
    print("\n".join(msg_lines))

      
# =================
# Telegram batching
# =================
TG_BATCH = []

def stage_now():
    """Xác định đang ở đầu giờ (H1 close) hay giữa giờ (H1 mid)."""
    # phút 0 => CLOSE nến 1H; phút 30 => MID của nến 1H
    m = datetime.now(timezone.utc).minute
    return "H1 CLOSE" if m == 0 else ("H1 MID" if m == 30 else "RUN")

def add_line(symbol, tf, text):
    TG_BATCH.append(f"• {symbol} [{tf}] {text}")

def flush_batch():
    if not TG_BATCH:
        return
    # Tiêu đề cho bản cập nhật giữa kỳ
    title = f"[{stage_now()}] CẬP NHẬT GIỮA KỲ THỊ TRƯỜNG"
    # Gộp các symbol bằng đường kẻ rõ ràng
    body = title + "\n" + "\n──────\n".join(TG_BATCH)
    telegram_send(body)
    TG_BATCH.clear()
# ===========================
# Main Loop
# ===========================
def run_loop():
    global last_cache_save
    init_state_structs(SYMBOLS)
    print(f"[START] {EXCHANGE_NAME.upper()} • Multi-symbol: {', '.join(SYMBOLS)} • TF {TF_1H}/{TF_2H}/{TF_4H}/1D")
    batch_index = 0

    while True:
        try:
            start = batch_index * BATCH_SIZE
            end = start + BATCH_SIZE
            batch = SYMBOLS[start:end]
            if not batch:
                batch_index = 0
                continue

            for sym in batch:
                try:
                    df1 = fetch_ohlcv_cached(sym, TF_1H, limit=300)

                    new_1h = (last_closed_1h[sym] is None or int(df1["ts"].iloc[-1]) != last_closed_1h[sym])
                    if new_1h:
                        df2 = fetch_ohlcv_cached(sym, TF_2H, limit=300)
                        df4 = fetch_ohlcv_cached(sym, TF_4H, limit=300)
                        dfd = fetch_ohlcv_cached(sym, TF_1D, limit=200)

                        a1 = analyze_one_tf(df1, TF_1H)
                        a2 = analyze_one_tf(df2, TF_2H)
                        a4 = analyze_one_tf(df4, TF_4H)
                        ad = analyze_one_tf(dfd, TF_1D)

                        report, decided_side, reco = make_report(sym, a1, a2, a4, ad)
                        print("="*72)
                        print(report)

                        # ✅ gom vào batch, hiển thị theo TF chính bạn đang phân tích (1H)
                        add_line(sym, "1H", report)

                        if decided_side and reco:
                            last_signal[sym] = {
                                "side": decided_side,
                                "entry": a1["price"],
                                "sl": reco["sl"],
                                "tp1": reco["tp1"],
                                "tp2": reco["tp2"],
                                "time": a1["time"],
                            }

                        last_closed_1h[sym] = int(df1["ts"].iloc[-1])

                    mid_update(sym, df1)

                except Exception as e:
                    print(f"[ERR][{sym}] {e}")
            # ✅ gửi 1 tin duy nhất cho batch vừa xử lý
            flush_batch()

            if time.time() - last_cache_save > SAVE_INTERVAL:
                save_cache()
                last_cache_save = time.time()

            total_batches = math.ceil(len(SYMBOLS)/BATCH_SIZE)
            batch_index = (batch_index + 1) % max(1, total_batches)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"[LOOP] Error: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    load_cache()
    run_loop()
