"""
technical_analysis.py
======================
Modul analisis teknikal multi-timeframe untuk bot Telegram XAU/USD.

Berisi:
- fetch_ohlc()          -> ambil data candle dari Twelve Data API
- calculate_rsi()       -> RSI (14)
- calculate_sma()       -> SMA (20)
- calculate_bollinger_bands()
- calculate_atr()       -> Average True Range
- analyze_timeframe()   -> analisis 1 timeframe
- get_market_data()     -> analisis multi-timeframe + confluence score + SL/TP
- button()              -> handler tombol Telegram (CallbackQueryHandler)

CATATAN KEAMANAN:
- JANGAN hardcode API key di sini. Set via environment variable di Railway:
  Project -> Variables -> TWELVE_DATA_API_KEY
- Karena key yang Anda kirim di chat sebelumnya sudah terekspos, segera
  regenerate key tersebut di dashboard Twelve Data.

Dependensi tambahan yang perlu ditambahkan ke requirements.txt:
    httpx>=0.27
    numpy>=1.26
"""

import os
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
BASE_URL = "https://api.twelvedata.com"

SYMBOL = "XAU/USD"

# Timeframe yang dianalisis, beserta bobot konfluensi.
# Timeframe lebih besar diberi bobot lebih tinggi karena lebih "reliable".
TIMEFRAMES: Dict[str, float] = {
    "5min": 1.0,
    "15min": 1.5,
    "30min": 2.0,
    "1h": 2.5,
}

RSI_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV = 2
ATR_PERIOD = 14

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

ATR_SL_MULTIPLIER = 1.5   # jarak SL = ATR * multiplier
ATR_TP_MULTIPLIER = 3.0   # jarak TP = ATR * multiplier (RR ~ 1:2)

HTTP_TIMEOUT = 10.0
MIN_CANDLES_REQUIRED = 30  # minimal candle agar indikator valid


# ----------------------------------------------------------------------------
# 1. PENGAMBILAN DATA (Twelve Data API)
# ----------------------------------------------------------------------------

async def fetch_ohlc(
    symbol: str,
    interval: str,
    outputsize: int = 100,
) -> Optional[List[Dict[str, Any]]]:
    """
    Ambil data candlestick (OHLC) dari Twelve Data.
    Mengembalikan list candle terurut dari LAMA -> BARU, atau None jika gagal.
    """
    if not TWELVE_DATA_API_KEY:
        logger.error("TWELVE_DATA_API_KEY tidak ditemukan di environment variable.")
        return None

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(f"{BASE_URL}/time_series", params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") == "error":
            logger.error(f"Twelve Data error [{interval}]: {data.get('message')}")
            return None

        values = data.get("values")
        if not values:
            logger.error(f"Data kosong dari Twelve Data untuk interval {interval}.")
            return None

        # Twelve Data mengembalikan data BARU -> LAMA, kita balik urutannya
        values = list(reversed(values))

        # Konversi field numerik dari string ke float
        parsed = []
        for v in values:
            parsed.append({
                "datetime": v["datetime"],
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
            })
        return parsed

    except httpx.TimeoutException:
        logger.error(f"Timeout saat mengambil data Twelve Data ({interval}).")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error Twelve Data ({interval}): {e.response.status_code}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Gagal parsing data Twelve Data ({interval}): {e}")
        return None
    except Exception as e:
        logger.exception(f"Error tak terduga saat fetch_ohlc ({interval}): {e}")
        return None


# ----------------------------------------------------------------------------
# 2. INDIKATOR TEKNIKAL
# ----------------------------------------------------------------------------

def calculate_sma(closes: np.ndarray, period: int = SMA_PERIOD) -> Optional[float]:
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def calculate_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def calculate_bollinger_bands(
    closes: np.ndarray, period: int = BB_PERIOD, std_dev: float = BB_STD_DEV
) -> Optional[Dict[str, float]]:
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window))

    upper = middle + std_dev * std
    lower = middle - std_dev * std
    # Lebar band dinormalisasi terhadap harga, untuk mengukur volatilitas relatif
    bandwidth = (upper - lower) / middle if middle != 0 else 0.0

    return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth}


def calculate_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD
) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    prev_closes = closes[:-1]
    cur_highs = highs[1:]
    cur_lows = lows[1:]

    tr1 = cur_highs - cur_lows
    tr2 = np.abs(cur_highs - prev_closes)
    tr3 = np.abs(cur_lows - prev_closes)
    true_ranges = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = float(np.mean(true_ranges[-period:]))
    return atr


# ----------------------------------------------------------------------------
# 3. ANALISIS PER TIMEFRAME
# ----------------------------------------------------------------------------

def analyze_timeframe(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Hitung semua indikator untuk satu timeframe dan tentukan bias
    (BULLISH / BEARISH / NEUTRAL) beserta skor mentah (-2..+2).
    """
    if len(candles) < MIN_CANDLES_REQUIRED:
        return None

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    last_price = float(closes[-1])

    rsi = calculate_rsi(closes)
    sma = calculate_sma(closes)
    bb = calculate_bollinger_bands(closes)
    atr = calculate_atr(highs, lows, closes)

    if rsi is None or sma is None or bb is None or atr is None:
        return None

    score = 0.0
    notes = []

    # --- Kontribusi SMA (trend) ---
    if last_price > sma:
        score += 1
        notes.append("Harga > SMA20 (trend naik)")
    elif last_price < sma:
        score -= 1
        notes.append("Harga < SMA20 (trend turun)")

    # --- Kontribusi RSI (momentum) ---
    if rsi <= RSI_OVERSOLD:
        score += 1
        notes.append(f"RSI oversold ({rsi:.1f})")
    elif rsi >= RSI_OVERBOUGHT:
        score -= 1
        notes.append(f"RSI overbought ({rsi:.1f})")

    # --- Kontribusi Bollinger Bands (volatilitas & posisi harga) ---
    if last_price <= bb["lower"]:
        score += 1
        notes.append("Harga menyentuh Lower Band")
    elif last_price >= bb["upper"]:
        score -= 1
        notes.append("Harga menyentuh Upper Band")

    if score > 0:
        bias = "BULLISH"
    elif score < 0:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "price": last_price,
        "rsi": rsi,
        "sma": sma,
        "bb": bb,
        "atr": atr,
        "score": score,
        "bias": bias,
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# 4. ANALISIS MULTI-TIMEFRAME + CONFLUENCE SCORE + SL/TP
# ----------------------------------------------------------------------------

async def get_market_data(symbol: str = SYMBOL) -> Dict[str, Any]:
    """
    Analisis multi-timeframe (5m/15m/30m/1h) dengan RSI, SMA, Bollinger Bands,
    hitung confluence score, dan tentukan SL/TP dinamis berbasis ATR.

    Mengembalikan dict siap-pakai untuk diformat ke pesan Telegram.
    Tidak pernah melempar exception ke caller — semua error ditangani
    dan dikembalikan sebagai {"success": False, "error": "..."}.
    """
    result: Dict[str, Any] = {
        "success": False,
        "symbol": symbol,
        "timeframes": {},
        "confluence_score": 0.0,
        "confidence_pct": 0.0,
        "signal": "NO SIGNAL",
        "entry": None,
        "sl": None,
        "tp": None,
        "error": None,
    }

    try:
        weighted_score = 0.0
        max_possible = sum(TIMEFRAMES.values()) * 3  # 3 indikator, skor maks ±1 tiap indikator
        valid_tf_count = 0
        last_price = None
        primary_atr = None  # ATR dari timeframe terkecil (5m) untuk SL/TP scalping

        for interval, weight in TIMEFRAMES.items():
            candles = await fetch_ohlc(symbol, interval, outputsize=100)
            if candles is None:
                logger.warning(f"Lewati timeframe {interval}: data tidak tersedia.")
                continue

            analysis = analyze_timeframe(candles)
            if analysis is None:
                logger.warning(f"Lewati timeframe {interval}: candle tidak cukup untuk indikator.")
                continue

            result["timeframes"][interval] = analysis
            weighted_score += analysis["score"] * weight
            valid_tf_count += 1
            last_price = analysis["price"]

            if interval == "5min":
                primary_atr = analysis["atr"]

        if valid_tf_count == 0:
            result["error"] = "Gagal mengambil data dari semua timeframe. Coba lagi beberapa saat."
            return result

        # Jika 5m tidak tersedia, fallback ke ATR timeframe manapun yang berhasil
        if primary_atr is None:
            primary_atr = next(
                (tf["atr"] for tf in result["timeframes"].values()), None
            )

        result["confluence_score"] = round(weighted_score, 2)
        confidence_pct = round((abs(weighted_score) / max_possible) * 100, 1) if max_possible else 0.0
        result["confidence_pct"] = confidence_pct

        # Tentukan sinyal akhir berdasarkan arah & kekuatan skor
        if weighted_score > 0:
            signal = "BUY" if confidence_pct >= 40 else "BUY (lemah)"
        elif weighted_score < 0:
            signal = "SELL" if confidence_pct >= 40 else "SELL (lemah)"
        else:
            signal = "NO SIGNAL / SIDEWAYS"

        result["signal"] = signal
        result["entry"] = last_price

        # --- SL/TP dinamis berbasis ATR ---
        if last_price is not None and primary_atr:
            if signal.startswith("BUY"):
                result["sl"] = round(last_price - primary_atr * ATR_SL_MULTIPLIER, 2)
                result["tp"] = round(last_price + primary_atr * ATR_TP_MULTIPLIER, 2)
            elif signal.startswith("SELL"):
                result["sl"] = round(last_price + primary_atr * ATR_SL_MULTIPLIER, 2)
                result["tp"] = round(last_price - primary_atr * ATR_TP_MULTIPLIER, 2)

        result["primary_atr"] = round(primary_atr, 2) if primary_atr else None
        result["success"] = True
        return result

    except Exception as e:
        logger.exception(f"Error tak terduga di get_market_data: {e}")
        result["error"] = "Terjadi kesalahan internal saat menganalisis pasar."
        return result


# ----------------------------------------------------------------------------
# 5. FORMAT PESAN
# ----------------------------------------------------------------------------

def format_analysis_message(data: Dict[str, Any]) -> str:
    if not data.get("success"):
        return f"⚠️ Analisis gagal: {data.get('error', 'Unknown error')}"

    lines = [
        f"📊 *Analisis {data['symbol']} — Multi-Timeframe*",
        "",
    ]

    for interval, tf in data["timeframes"].items():
        bias_emoji = "🟢" if tf["bias"] == "BULLISH" else "🔴" if tf["bias"] == "BEARISH" else "⚪"
        lines.append(
            f"{bias_emoji} *{interval}* — {tf['bias']}\n"
            f"   Price: {tf['price']:.2f} | RSI: {tf['rsi']:.1f} | SMA20: {tf['sma']:.2f}\n"
            f"   BB: [{tf['bb']['lower']:.2f} — {tf['bb']['upper']:.2f}] | ATR: {tf['atr']:.2f}"
        )

    lines.append("")
    lines.append(f"🎯 *Sinyal:* {data['signal']}")
    lines.append(f"📈 *Confluence Score:* {data['confluence_score']} ({data['confidence_pct']}% keyakinan)")

    if data.get("entry") is not None:
        lines.append("")
        lines.append(f"💰 Entry: {data['entry']:.2f}")
        if data.get("sl") is not None:
            lines.append(f"🛑 SL (ATR x{ATR_SL_MULTIPLIER}): {data['sl']:.2f}")
        if data.get("tp") is not None:
            lines.append(f"✅ TP (ATR x{ATR_TP_MULTIPLIER}): {data['tp']:.2f}")

    lines.append("")
    lines.append("_Bukan nasihat finansial. Selalu gunakan money management._")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 6. HANDLER TOMBOL TELEGRAM
# ----------------------------------------------------------------------------

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler untuk CallbackQueryHandler. Dipanggil saat user menekan tombol
    "Analisis XAU/USD" di inline keyboard.
    """
    query = update.callback_query
    await query.answer()  # wajib, agar tombol tidak "loading" terus di UI

    if query.data != "analyze_xauusd":
        return

    try:
        await query.edit_message_text("⏳ Mengambil data pasar & menghitung indikator...")
    except Exception:
        # Jika edit gagal (misal pesan sama/terlalu cepat), abaikan, lanjut proses
        pass

    try:
        data = await get_market_data(SYMBOL)
        message = format_analysis_message(data)

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Refresh Analisis", callback_data="analyze_xauusd")]
        ])

        await query.edit_message_text(
            text=message,
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

    except Exception as e:
        logger.exception(f"Error di handler button(): {e}")
        try:
            await query.edit_message_text(
                "⚠️ Terjadi kesalahan saat memproses analisis. Silakan coba lagi."
            )
        except Exception:
            # Kalau edit_message_text pun gagal, jangan sampai bot crash
            logger.error("Gagal mengirim pesan error ke user.")
