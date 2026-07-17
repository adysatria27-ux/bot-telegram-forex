"""
================================================================================
 Modul Analisis Teknikal Multi-Timeframe untuk Bot Telegram XAU/USD
================================================================================
Sumber data : Twelve Data (https://twelvedata.com)
Indikator   : RSI(14), SMA(20), Bollinger Bands(20,2), ATR(14)
Output      : Sinyal BUY/SELL/NEUTRAL + confidence score + SL/TP dinamis (ATR)

CARA INTEGRASI KE BOT YANG SUDAH ADA:
--------------------------------------------------------------------------------
1. Install dependency tambahan (masukkan ke requirements.txt):
     aiohttp
     pandas
     numpy

2. Set environment variable di Railway (Settings > Variables):
     TWELVE_DATA_API_KEY = <api_key_anda>
   JANGAN hardcode API key langsung di file ini.

3. Di file bot utama Anda:
     from xauusd_analysis import get_market_data, button

     application.add_handler(CallbackQueryHandler(button, pattern="^analyze_xauusd$"))

     # Contoh tombol untuk memicu analisis:
     keyboard = InlineKeyboardMarkup(
         [[InlineKeyboardButton("📊 Analisa XAU/USD", callback_data="analyze_xauusd")]]
     )
     await update.message.reply_text("Pilih menu:", reply_markup=keyboard)

CATATAN RATE LIMIT:
--------------------------------------------------------------------------------
Twelve Data plan gratis: 8 request/menit, 800 request/hari.
Setiap kali analisis dijalankan, modul ini melakukan 4 request (1 per timeframe).
Artinya maksimal ~2x analisis/menit di plan gratis. Jika bot dipakai banyak user
sekaligus untuk scalping real-time, pertimbangkan upgrade plan Twelve Data.
================================================================================
"""

import os
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional

import aiohttp
import pandas as pd
import numpy as np

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# ── Konfigurasi ──────────────────────────────────────────────────────────────
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
BASE_URL = "https://api.twelvedata.com/time_series"

DEFAULT_SYMBOL = "XAU/USD"
TIMEFRAMES = ["5min", "15min", "30min", "1h"]
TIMEFRAME_LABELS = {"5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h"}
# Bobot: timeframe lebih besar dianggap lebih menentukan arah tren utama
TIMEFRAME_WEIGHTS = {"5min": 1.0, "15min": 1.5, "30min": 2.0, "1h": 2.5}

OUTPUT_SIZE = 100          # jumlah candle historis yang diambil per timeframe
REQUEST_TIMEOUT_SEC = 10
MAX_RETRIES = 2

RSI_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV = 2
ATR_PERIOD = 14

# Multiplier ATR untuk SL/TP dinamis (bisa disesuaikan sesuai gaya scalping)
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

# Timeframe acuan untuk hitung ATR->SL/TP (cocok untuk scalping)
ENTRY_TIMEFRAME_FOR_ATR = "15min"

# Ambang skor gabungan (-1..+1) untuk menentukan arah sinyal
SIGNAL_THRESHOLD = 0.15


@dataclass
class TimeframeResult:
    timeframe: str
    close: float
    rsi: float
    sma: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    atr: float
    score: float  # -1 (bearish kuat) .. +1 (bullish kuat)


# ══════════════════════════════════════════════════════════════════════════
# 1. FETCH DATA (async, paralel per timeframe, dengan retry & error handling)
# ══════════════════════════════════════════════════════════════════════════
async def _fetch_ohlc(
    session: aiohttp.ClientSession, symbol: str, interval: str
) -> Optional[pd.DataFrame]:
    """Ambil data OHLC dari Twelve Data untuk satu timeframe. Return None jika gagal."""
    if not TWELVE_DATA_API_KEY:
        logger.error("TWELVE_DATA_API_KEY belum diset di environment variable.")
        return None

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVE_DATA_API_KEY,
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(BASE_URL, params=params, timeout=timeout) as resp:
                if resp.status == 429:
                    logger.warning("Rate limit Twelve Data tercapai untuk TF %s.", interval)
                    return None
                if resp.status != 200:
                    logger.warning("HTTP %d saat fetch TF %s.", resp.status, interval)
                    return None

                data = await resp.json(content_type=None)

                if isinstance(data, dict) and data.get("status") == "error":
                    logger.error("Twelve Data error [%s]: %s", interval, data.get("message"))
                    return None

                values = data.get("values") if isinstance(data, dict) else None
                if not values:
                    logger.warning("Data kosong dari Twelve Data untuk TF %s.", interval)
                    return None

                df = pd.DataFrame(values)
                required_cols = {"datetime", "open", "high", "low", "close"}
                if not required_cols.issubset(df.columns):
                    logger.error("Kolom tidak lengkap pada respons TF %s.", interval)
                    return None

                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
                df["datetime"] = pd.to_datetime(df["datetime"])
                # Twelve Data mengembalikan data terbaru lebih dulu -> urutkan kronologis
                df = df.sort_values("datetime").reset_index(drop=True)
                return df

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(
                "Percobaan %d/%d gagal fetch TF %s: %s", attempt, MAX_RETRIES, interval, e
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5 * attempt)  # backoff sederhana
        except Exception as e:
            logger.exception("Error tak terduga saat fetch TF %s: %s", interval, e)
            return None

    return None


# ══════════════════════════════════════════════════════════════════════════
# 2. INDIKATOR TEKNIKAL (dihitung manual dengan pandas, tanpa TA-Lib)
# ══════════════════════════════════════════════════════════════════════════
def _calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # netral jika belum cukup data


def _calculate_bollinger(close: pd.Series, period: int = BB_PERIOD, num_std: float = BB_STD_DEV):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _score_timeframe(df: pd.DataFrame, timeframe: str) -> Optional[TimeframeResult]:
    """Hitung semua indikator + skor konfluensi (-1..+1) untuk satu timeframe."""
    min_rows_needed = max(RSI_PERIOD, SMA_PERIOD, ATR_PERIOD) + 5
    if len(df) < min_rows_needed:
        logger.warning("Data TF %s terlalu sedikit (%d baris).", timeframe, len(df))
        return None

    close = df["close"]
    rsi_series = _calculate_rsi(close)
    sma_series = close.rolling(SMA_PERIOD).mean()
    bb_upper, bb_mid, bb_lower = _calculate_bollinger(close)
    atr_series = _calculate_atr(df)

    last_close = close.iloc[-1]
    last_rsi = rsi_series.iloc[-1]
    last_sma = sma_series.iloc[-1]
    last_bb_upper = bb_upper.iloc[-1]
    last_bb_lower = bb_lower.iloc[-1]
    last_bb_mid = bb_mid.iloc[-1]
    last_atr = atr_series.iloc[-1]

    if any(pd.isna(x) for x in [last_rsi, last_sma, last_bb_upper, last_bb_lower, last_atr]):
        logger.warning("Nilai indikator NaN pada TF %s, dilewati.", timeframe)
        return None

    # -- Skor RSI: oversold => bullish, overbought => bearish
    if last_rsi <= 30:
        rsi_score = 1.0
    elif last_rsi >= 70:
        rsi_score = -1.0
    else:
        rsi_score = max(-0.5, min(0.5, (50 - last_rsi) / 20))

    # -- Skor SMA: posisi harga vs SMA20 (trend following)
    sma_score = 1.0 if last_close > last_sma else -1.0

    # -- Skor Bollinger Bands: dekat lower band => bullish (potensi bounce),
    #    dekat upper band => bearish
    bb_range = last_bb_upper - last_bb_lower
    if bb_range > 0:
        position_in_band = (last_close - last_bb_lower) / bb_range  # ~0..1
        bb_score = max(-1.0, min(1.0, 1 - 2 * position_in_band))
    else:
        bb_score = 0.0

    total_score = (rsi_score + sma_score + bb_score) / 3

    return TimeframeResult(
        timeframe=timeframe,
        close=last_close,
        rsi=last_rsi,
        sma=last_sma,
        bb_upper=last_bb_upper,
        bb_lower=last_bb_lower,
        bb_mid=last_bb_mid,
        atr=last_atr,
        score=total_score,
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. FUNGSI UTAMA: get_market_data
# ══════════════════════════════════════════════════════════════════════════
async def get_market_data(symbol: str = DEFAULT_SYMBOL) -> dict:
    """
    Ambil & analisis data multi-timeframe (5m, 15m, 30m, 1h) untuk `symbol`.

    Return dict:
        {
            "symbol": str,
            "timeframes": {tf: TimeframeResult, ...},
            "combined_score": float (-1..+1),
            "confidence_pct": float (0..100),
            "direction": "BUY" | "SELL" | "NEUTRAL",
            "entry_price": float,
            "atr_reference_tf": str,
            "atr_value": float,
            "sl": float | None,
            "tp": float | None,
        }

    Raises RuntimeError jika SEMUA timeframe gagal diambil/dihitung
    (misalnya API key salah, limit habis, atau koneksi mati total).
    """
    try:
        async with aiohttp.ClientSession() as session:
            fetch_tasks = [_fetch_ohlc(session, symbol, tf) for tf in TIMEFRAMES]
            dfs = await asyncio.gather(*fetch_tasks)
    except Exception as e:
        logger.exception("Gagal membuat sesi HTTP / fetch data: %s", e)
        raise RuntimeError("Tidak dapat menghubungi Twelve Data.") from e

    tf_results: dict = {}
    for tf, df in zip(TIMEFRAMES, dfs):
        if df is None:
            continue
        res = _score_timeframe(df, tf)
        if res is not None:
            tf_results[tf] = res

    if not tf_results:
        raise RuntimeError(
            "Gagal mengambil/menghitung data dari semua timeframe. "
            "Cek API key, limit request, atau koneksi jaringan."
        )

    # ── Confidence score gabungan (weighted berdasarkan timeframe) ──
    total_weight = sum(TIMEFRAME_WEIGHTS[tf] for tf in tf_results)
    weighted_score = sum(res.score * TIMEFRAME_WEIGHTS[tf] for tf, res in tf_results.items())
    combined_score = weighted_score / total_weight  # -1..+1
    confidence_pct = round(abs(combined_score) * 100, 1)

    if combined_score > SIGNAL_THRESHOLD:
        direction = "BUY"
    elif combined_score < -SIGNAL_THRESHOLD:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    # ── SL/TP dinamis berbasis ATR ──
    atr_ref_tf = ENTRY_TIMEFRAME_FOR_ATR if ENTRY_TIMEFRAME_FOR_ATR in tf_results else next(iter(tf_results))
    ref = tf_results[atr_ref_tf]
    entry_price = ref.close
    atr_value = ref.atr

    sl = tp = None
    if direction == "BUY":
        sl = entry_price - ATR_SL_MULTIPLIER * atr_value
        tp = entry_price + ATR_TP_MULTIPLIER * atr_value
    elif direction == "SELL":
        sl = entry_price + ATR_SL_MULTIPLIER * atr_value
        tp = entry_price - ATR_TP_MULTIPLIER * atr_value

    return {
        "symbol": symbol,
        "timeframes": tf_results,
        "combined_score": combined_score,
        "confidence_pct": confidence_pct,
        "direction": direction,
        "entry_price": entry_price,
        "atr_reference_tf": atr_ref_tf,
        "atr_value": atr_value,
        "sl": sl,
        "tp": tp,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. FORMAT PESAN TELEGRAM
# ══════════════════════════════════════════════════════════════════════════
def generate_signal_message(analysis: dict) -> str:
    """Ubah hasil get_market_data() menjadi teks Markdown siap kirim ke Telegram."""
    symbol = analysis["symbol"]
    direction = analysis["direction"]
    confidence = analysis["confidence_pct"]
    emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}[direction]

    lines = [
        f"*📊 Analisis Multi-Timeframe — {symbol}*",
        "",
        f"{emoji} *Sinyal: {direction}*   |   Confidence: *{confidence}%*",
        "",
        "*Detail per Timeframe:*",
    ]

    for tf in TIMEFRAMES:
        res = analysis["timeframes"].get(tf)
        label = TIMEFRAME_LABELS.get(tf, tf)
        if res is None:
            lines.append(f"• {label}: _data tidak tersedia_")
            continue
        bias = "Bullish" if res.score > SIGNAL_THRESHOLD else "Bearish" if res.score < -SIGNAL_THRESHOLD else "Netral"
        bb_zone = "lower" if res.close <= res.bb_mid else "upper"
        trend_arrow = "↑" if res.close > res.sma else "↓"
        lines.append(
            f"• *{label}*: {bias} (RSI {res.rsi:.1f} | Close vs SMA20 {trend_arrow} | BB zona {bb_zone})"
        )

    lines.append("")
    if direction != "NEUTRAL":
        lines.append(f"*Entry:* {analysis['entry_price']:.2f}")
        lines.append(f"*Stop Loss:* {analysis['sl']:.2f}")
        lines.append(f"*Take Profit:* {analysis['tp']:.2f}")
        lines.append(
            f"_SL/TP dihitung dari ATR({ATR_PERIOD}) di TF {TIMEFRAME_LABELS.get(analysis['atr_reference_tf'])} "
            f"(SL {ATR_SL_MULTIPLIER}x ATR, TP {ATR_TP_MULTIPLIER}x ATR)_"
        )
    else:
        lines.append("_Sinyal belum cukup kuat / konfluensi timeframe belum searah. Tunggu konfirmasi._")

    lines.append("")
    lines.append("⚠️ _Bukan saran finansial. Selalu gunakan manajemen risiko & position sizing sendiri._")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 5. FUNGSI UTAMA: button (CallbackQueryHandler)
# ══════════════════════════════════════════════════════════════════════════
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler untuk tombol inline "Analisa XAU/USD".
    Daftarkan dengan:
        CallbackQueryHandler(button, pattern="^analyze_xauusd$")
    """
    query = update.callback_query
    await query.answer()  # wajib, agar tombol tidak "loading" terus di client Telegram

    if query.data != "analyze_xauusd":
        return

    refresh_keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔄 Refresh Analisa", callback_data="analyze_xauusd")]]
    )

    try:
        await query.edit_message_text(
            "⏳ Mengambil & menganalisis data 4 timeframe (5m/15m/30m/1h)..."
        )
    except Exception:
        # Bukan error fatal—lanjut saja meski pesan "loading" gagal diedit
        logger.debug("Gagal mengedit pesan loading, melanjutkan proses.")

    try:
        analysis = await get_market_data(DEFAULT_SYMBOL)
        message = generate_signal_message(analysis)
        await query.edit_message_text(message, parse_mode="Markdown", reply_markup=refresh_keyboard)

    except RuntimeError as e:
        logger.error("RuntimeError saat analisis pasar: %s", e)
        await query.edit_message_text(
            "❌ Gagal mengambil data pasar dari Twelve Data.\n"
            "Kemungkinan penyebab: limit API tercapai, API key tidak valid, atau koneksi bermasalah.\n"
            "Silakan coba lagi beberapa saat lagi.",
            reply_markup=refresh_keyboard,
        )
    except Exception as e:
        # Tangkap semua error tak terduga supaya bot TIDAK CRASH
        logger.exception("Error tak terduga di button handler")
        await query.edit_message_text(
            f"❌ Terjadi kesalahan tak terduga saat analisis ({type(e).__name__}). "
            "Tim developer sudah bisa cek log untuk detailnya.",
            reply_markup=refresh_keyboard,
        )
# Tambahkan ini di paling bawah tes_bot.py untuk menghidupkan bot
if __name__ == '__main__':
    from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler
    
    # Fungsi start sederhana
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [[InlineKeyboardButton("📊 Analisa XAU/USD", callback_data="analyze_xauusd")]]
        await update.message.reply_text("Pilih menu:", reply_markup=InlineKeyboardMarkup(keyboard))

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_xauusd$"))
    
    print("Bot berjalan...")
    app.run_polling()
