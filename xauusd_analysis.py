"""
================================================================================
Modul Analisis Teknikal Multi-Timeframe untuk Bot Telegram Forex
================================================================================

Sumber data:
    Twelve Data

Tujuan:
    Menghasilkan analisis BUY, SELL, atau HOLD menggunakan konfluensi
    beberapa timeframe dan indikator teknikal.

Indikator saat ini:
    - RSI 14
    - SMA 20
    - Bollinger Bands 20,2
    - ATR 14
    - Momentum harga
    - Multi-timeframe weighting

Catatan penting:
    Confidence Score pada versi ini adalah skor kualitas konfluensi internal.
    Nilai tersebut belum menjadi probabilitas kemenangan sampai dilakukan
    backtest dan kalibrasi statistik.

Kompatibilitas:
    - Fungsi get_market_data() tetap tersedia.
    - Fungsi generate_signal_message() tetap tersedia.
    - Fungsi button() tetap tersedia.
    - Field lama direction, sl, tp, dan atr_value tetap dipertahankan.
    - Cache hasil analisis dan OHLC menghemat request Twelve Data.
================================================================================
"""

import os
import copy
import time
import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
import numpy as np
import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes


logger = logging.getLogger(__name__)


# =============================================================================
# 1. KONFIGURASI DATA DAN TIMEFRAME
# =============================================================================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
BASE_URL = "https://api.twelvedata.com/time_series"

DEFAULT_SYMBOL = "XAU/USD"

TIMEFRAMES = ["5min", "15min", "30min", "1h"]

TIMEFRAME_LABELS = {
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
}

# Timeframe besar diberi pengaruh lebih tinggi.
TIMEFRAME_WEIGHTS = {
    "5min": 1.0,
    "15min": 1.5,
    "30min": 2.0,
    "1h": 2.5,
}

HIGHER_TIMEFRAMES = ("30min", "1h")

OUTPUT_SIZE = 120
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

# Candle terakhir dianggap masih berpotensi berjalan.
# Harga candle terakhir tetap digunakan sebagai harga referensi,
# tetapi perhitungan indikator menggunakan candle sebelumnya.
EXCLUDE_LATEST_CANDLE_FROM_INDICATORS = True

# Cache dibuat singkat agar data tetap segar sekaligus menghemat kuota API.
ANALYSIS_CACHE_TTL_SECONDS = 30
OHLC_CACHE_TTL_SECONDS = {
    "5min": 45,
    "15min": 90,
    "30min": 180,
    "1h": 300,
}
MAX_CONCURRENT_API_REQUESTS = 4


# =============================================================================
# 2. KONFIGURASI INDIKATOR DAN SINYAL
# =============================================================================
RSI_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV = 2.0
ATR_PERIOD = 14
MOMENTUM_LOOKBACK = 3

ENTRY_TIMEFRAME_FOR_ATR = "15min"

ATR_SL_MULTIPLIER = 1.5
ATR_TP1_MULTIPLIER = 1.5
ATR_TP2_MULTIPLIER = 2.5
ATR_TP3_MULTIPLIER = 4.0

# Sistem dibuat konservatif karena akurasi lebih penting daripada jumlah sinyal.
SIGNAL_SCORE_THRESHOLD = 0.25
TIMEFRAME_BIAS_THRESHOLD = 0.15
MIN_TIMEFRAMES_FOR_SIGNAL = 3
MIN_COVERAGE_RATIO = 0.65
MIN_DIRECTIONAL_AGREEMENT = 0.65
MIN_CONFIDENCE_FOR_SIGNAL = 60.0


@dataclass
class TimeframeResult:
    """
    Hasil analisis untuk satu timeframe.

    Field lama tetap dipertahankan agar kompatibel:
    timeframe, close, rsi, sma, bb_upper, bb_lower, bb_mid, atr, score.
    """

    timeframe: str
    close: float
    rsi: float
    sma: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    atr: float
    score: float
    trend_score: float
    rsi_score: float
    bb_score: float
    momentum_score: float
    analyzed_candle_time: str
    data_points: int


@dataclass
class CacheEntry:
    """Menyimpan nilai cache beserta waktu pembuatan dan kedaluwarsa."""

    value: Any
    created_at: float
    expires_at: float


_OHLC_CACHE: dict[tuple[str, str], CacheEntry] = {}
_ANALYSIS_CACHE: dict[str, CacheEntry] = {}
_ANALYSIS_LOCKS: dict[str, asyncio.Lock] = {}
_API_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _normalize_symbol(symbol: str) -> str:
    """Menormalkan symbol agar key cache konsisten."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Symbol tidak boleh kosong.")
    return normalized


def _get_analysis_lock(symbol: str) -> asyncio.Lock:
    """Mengambil lock per symbol untuk mencegah request identik ganda."""
    lock = _ANALYSIS_LOCKS.get(symbol)
    if lock is None:
        lock = asyncio.Lock()
        _ANALYSIS_LOCKS[symbol] = lock
    return lock


def _get_api_semaphore() -> asyncio.Semaphore:
    """Membatasi jumlah request Twelve Data yang berjalan bersamaan."""
    global _API_SEMAPHORE

    if _API_SEMAPHORE is None:
        _API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_API_REQUESTS)

    return _API_SEMAPHORE


def _get_valid_cache_entry(
    cache: dict[Any, CacheEntry],
    key: Any,
) -> Optional[CacheEntry]:
    """Mengambil entry cache yang masih berlaku."""
    entry = cache.get(key)
    if entry is None:
        return None

    if time.monotonic() >= entry.expires_at:
        cache.pop(key, None)
        return None

    return entry


def _get_cached_ohlc(
    symbol: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Mengambil salinan DataFrame OHLC dari cache."""
    entry = _get_valid_cache_entry(
        _OHLC_CACHE,
        (symbol, interval),
    )
    if entry is None:
        return None

    return entry.value.copy(deep=True)


def _set_cached_ohlc(
    symbol: str,
    interval: str,
    dataframe: pd.DataFrame,
) -> None:
    """Menyimpan DataFrame OHLC ke cache dengan TTL per timeframe."""
    ttl = OHLC_CACHE_TTL_SECONDS.get(interval, 60)
    now = time.monotonic()

    _OHLC_CACHE[(symbol, interval)] = CacheEntry(
        value=dataframe.copy(deep=True),
        created_at=now,
        expires_at=now + ttl,
    )


def _get_cached_analysis(symbol: str) -> Optional[dict]:
    """Mengambil salinan hasil analisis yang masih berlaku."""
    entry = _get_valid_cache_entry(_ANALYSIS_CACHE, symbol)
    if entry is None:
        return None

    result = copy.deepcopy(entry.value)
    result["cache_hit"] = True
    result["cache_age_seconds"] = round(
        max(0.0, time.monotonic() - entry.created_at),
        1,
    )
    return result


def _set_cached_analysis(symbol: str, analysis: dict) -> None:
    """Menyimpan hasil analisis lengkap ke cache."""
    now = time.monotonic()
    cached_value = copy.deepcopy(analysis)
    cached_value["cache_hit"] = False
    cached_value["cache_age_seconds"] = 0.0

    _ANALYSIS_CACHE[symbol] = CacheEntry(
        value=cached_value,
        created_at=now,
        expires_at=now + ANALYSIS_CACHE_TTL_SECONDS,
    )


def clear_caches(symbol: Optional[str] = None) -> None:
    """
    Menghapus cache secara manual.

    symbol=None menghapus seluruh cache. Fungsi ini berguna untuk test,
    maintenance, atau saat konfigurasi strategi berubah.
    """
    if symbol is None:
        _OHLC_CACHE.clear()
        _ANALYSIS_CACHE.clear()
        return

    normalized_symbol = _normalize_symbol(symbol)
    _ANALYSIS_CACHE.pop(normalized_symbol, None)

    for key in list(_OHLC_CACHE):
        if key[0] == normalized_symbol:
            _OHLC_CACHE.pop(key, None)


# =============================================================================
# 3. HELPER VALIDASI DAN FORMAT
# =============================================================================
def _safe_float(value: Any, field_name: str) -> float:
    """
    Mengubah nilai menjadi float dan memastikan hasilnya finite.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nilai {field_name} tidak valid.") from exc

    if not np.isfinite(number):
        raise ValueError(f"Nilai {field_name} bukan angka finite.")

    return number


def _decimal_places(symbol: str) -> int:
    """
    Menentukan jumlah desimal untuk format pesan.
    """
    if symbol == "XAU/USD":
        return 2
    return 5


def _timeframe_bias(score: float) -> str:
    """
    Mengubah skor timeframe menjadi label sederhana.
    """
    if score >= TIMEFRAME_BIAS_THRESHOLD:
        return "Bullish"
    if score <= -TIMEFRAME_BIAS_THRESHOLD:
        return "Bearish"
    return "Netral"


# =============================================================================
# 4. FETCH DATA
# =============================================================================
async def _fetch_ohlc(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """
    Mengambil OHLC untuk satu timeframe.

    Data cache digunakan terlebih dahulu. Jika cache tidak tersedia atau sudah
    kedaluwarsa, fungsi mengambil data dari Twelve Data dan menyimpannya kembali.
    """
    normalized_symbol = _normalize_symbol(symbol)

    cached_dataframe = _get_cached_ohlc(
        normalized_symbol,
        interval,
    )
    if cached_dataframe is not None:
        logger.info(
            "OHLC cache hit | symbol=%s | tf=%s",
            normalized_symbol,
            interval,
        )
        return cached_dataframe

    if not TWELVE_DATA_API_KEY:
        logger.error(
            "TWELVE_DATA_API_KEY belum diatur pada environment variable."
        )
        return None

    params = {
        "symbol": normalized_symbol,
        "interval": interval,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVE_DATA_API_KEY,
    }

    semaphore = _get_api_semaphore()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                async with session.get(BASE_URL, params=params) as response:
                    if response.status == 429:
                        logger.warning(
                            "Rate limit Twelve Data tercapai | symbol=%s | tf=%s",
                            normalized_symbol,
                            interval,
                        )
                        return None

                    if response.status >= 500:
                        logger.warning(
                            "Server Twelve Data error | status=%s | symbol=%s | "
                            "tf=%s | percobaan=%s/%s",
                            response.status,
                            normalized_symbol,
                            interval,
                            attempt,
                            MAX_RETRIES,
                        )

                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(
                                RETRY_BACKOFF_SECONDS * attempt
                            )
                            continue

                        return None

                    if response.status != 200:
                        logger.warning(
                            "HTTP Twelve Data tidak berhasil | status=%s | "
                            "symbol=%s | tf=%s",
                            response.status,
                            normalized_symbol,
                            interval,
                        )
                        return None

                    try:
                        data = await response.json(content_type=None)
                    except (aiohttp.ContentTypeError, ValueError):
                        logger.exception(
                            "Respons Twelve Data bukan JSON valid | "
                            "symbol=%s | tf=%s",
                            normalized_symbol,
                            interval,
                        )
                        return None

            if not isinstance(data, dict):
                logger.warning(
                    "Format respons tidak valid | symbol=%s | tf=%s",
                    normalized_symbol,
                    interval,
                )
                return None

            if data.get("status") == "error":
                logger.warning(
                    "Twelve Data error | symbol=%s | tf=%s | message=%s",
                    normalized_symbol,
                    interval,
                    data.get("message", "Tidak diketahui"),
                )
                return None

            values = data.get("values")
            if not isinstance(values, list) or not values:
                logger.warning(
                    "Data candle kosong | symbol=%s | tf=%s",
                    normalized_symbol,
                    interval,
                )
                return None

            dataframe = pd.DataFrame(values)

            required_columns = {
                "datetime",
                "open",
                "high",
                "low",
                "close",
            }
            missing_columns = required_columns.difference(dataframe.columns)

            if missing_columns:
                logger.warning(
                    "Kolom candle tidak lengkap | symbol=%s | tf=%s | "
                    "missing=%s",
                    normalized_symbol,
                    interval,
                    sorted(missing_columns),
                )
                return None

            dataframe["datetime"] = pd.to_datetime(
                dataframe["datetime"],
                errors="coerce",
            )

            numeric_columns = ["open", "high", "low", "close"]
            for column in numeric_columns:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

            dataframe = (
                dataframe.dropna(
                    subset=["datetime", *numeric_columns]
                )
                .sort_values("datetime")
                .drop_duplicates(
                    subset=["datetime"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

            if dataframe.empty:
                logger.warning(
                    "Data candle tidak valid setelah dibersihkan | "
                    "symbol=%s | tf=%s",
                    normalized_symbol,
                    interval,
                )
                return None

            _set_cached_ohlc(
                normalized_symbol,
                interval,
                dataframe,
            )

            return dataframe.copy(deep=True)

        except asyncio.TimeoutError:
            logger.warning(
                "Timeout fetch data | symbol=%s | tf=%s | "
                "percobaan=%s/%s",
                normalized_symbol,
                interval,
                attempt,
                MAX_RETRIES,
            )

        except aiohttp.ClientError as exc:
            logger.warning(
                "Koneksi Twelve Data gagal | symbol=%s | tf=%s | "
                "percobaan=%s/%s | error=%s",
                normalized_symbol,
                interval,
                attempt,
                MAX_RETRIES,
                exc,
            )

        except Exception:
            logger.exception(
                "Error tak terduga saat fetch data | symbol=%s | tf=%s",
                normalized_symbol,
                interval,
            )
            return None

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None


# =============================================================================
# 5. INDIKATOR TEKNIKAL
# =============================================================================
def _calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """
    Menghitung RSI Wilder.

    Perbaikan edge case:
    - Semua candle naik  -> RSI mendekati 100.
    - Semua candle turun -> RSI mendekati 0.
    - Harga datar        -> RSI 50.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    safe_average_loss = average_loss.where(
        average_loss != 0,
        np.nan,
    )

    relative_strength = average_gain / safe_average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    both_zero = (average_gain == 0) & (average_loss == 0)
    gain_only = (average_gain > 0) & (average_loss == 0)
    loss_only = (average_gain == 0) & (average_loss > 0)

    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask(gain_only, 100.0)
    rsi = rsi.mask(loss_only, 0.0)

    return rsi.fillna(50.0).clip(lower=0.0, upper=100.0)


def _calculate_bollinger(
    close: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD_DEV,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Menghitung Bollinger Bands.
    """
    middle = close.rolling(
        window=period,
        min_periods=period,
    ).mean()

    standard_deviation = close.rolling(
        window=period,
        min_periods=period,
    ).std(ddof=0)

    upper = middle + (num_std * standard_deviation)
    lower = middle - (num_std * standard_deviation)

    return upper, middle, lower


def _calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Menghitung ATR Wilder menggunakan True Range.
    """
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


# =============================================================================
# 6. ANALISIS SATU TIMEFRAME
# =============================================================================
def _score_timeframe(
    df: pd.DataFrame,
    timeframe: str,
) -> Optional[TimeframeResult]:
    """
    Menghitung indikator dan skor konfluensi untuk satu timeframe.

    Skor:
        -1.0 = bearish kuat
         0.0 = netral
        +1.0 = bullish kuat

    Sistem menggunakan pendekatan trend-following agar RSI dan Bollinger
    tidak otomatis melawan tren kuat.
    """
    minimum_rows = (
        max(
            RSI_PERIOD,
            SMA_PERIOD,
            BB_PERIOD,
            ATR_PERIOD,
        )
        + MOMENTUM_LOOKBACK
        + 2
    )

    if len(df) < minimum_rows:
        logger.warning(
            "Data timeframe terlalu sedikit | tf=%s | tersedia=%s | "
            "minimum=%s",
            timeframe,
            len(df),
            minimum_rows,
        )
        return None

    close = df["close"]

    rsi_series = _calculate_rsi(close)
    sma_series = close.rolling(
        window=SMA_PERIOD,
        min_periods=SMA_PERIOD,
    ).mean()

    bb_upper, bb_middle, bb_lower = _calculate_bollinger(close)
    atr_series = _calculate_atr(df)

    last_close = _safe_float(close.iloc[-1], "close")
    last_rsi = _safe_float(rsi_series.iloc[-1], "RSI")
    last_sma = _safe_float(sma_series.iloc[-1], "SMA")
    last_bb_upper = _safe_float(
        bb_upper.iloc[-1],
        "Bollinger upper",
    )
    last_bb_lower = _safe_float(
        bb_lower.iloc[-1],
        "Bollinger lower",
    )
    last_bb_middle = _safe_float(
        bb_middle.iloc[-1],
        "Bollinger middle",
    )
    last_atr = _safe_float(atr_series.iloc[-1], "ATR")

    if last_atr <= 0:
        logger.warning(
            "ATR tidak valid | tf=%s | atr=%s",
            timeframe,
            last_atr,
        )
        return None

    # -------------------------------------------------------------------------
    # A. Trend score: jarak harga terhadap SMA dinormalisasi menggunakan ATR.
    # -------------------------------------------------------------------------
    trend_distance = (last_close - last_sma) / last_atr
    trend_score = float(
        np.clip(trend_distance / 1.5, -1.0, 1.0)
    )

    # -------------------------------------------------------------------------
    # B. RSI score: RSI di atas 50 mendukung bullish, di bawah 50 bearish.
    # Overbought tidak otomatis dianggap SELL tanpa konfirmasi struktur.
    # -------------------------------------------------------------------------
    rsi_score = float(
        np.clip((last_rsi - 50.0) / 20.0, -1.0, 1.0)
    )

    # -------------------------------------------------------------------------
    # C. Bollinger score: posisi terhadap middle band mengikuti arah tren.
    # -------------------------------------------------------------------------
    half_band_width = max(
        (last_bb_upper - last_bb_lower) / 2.0,
        np.finfo(float).eps,
    )

    bb_score = float(
        np.clip(
            (last_close - last_bb_middle) / half_band_width,
            -1.0,
            1.0,
        )
    )

    # -------------------------------------------------------------------------
    # D. Momentum score: perubahan beberapa candle dinormalisasi ATR.
    # -------------------------------------------------------------------------
    momentum_start = _safe_float(
        close.iloc[-1 - MOMENTUM_LOOKBACK],
        "momentum start",
    )

    momentum_change = last_close - momentum_start
    momentum_score = float(
        np.clip(
            momentum_change
            / (last_atr * max(MOMENTUM_LOOKBACK, 1)),
            -1.0,
            1.0,
        )
    )

    # Bobot dibuat eksplisit agar mudah disesuaikan dan diuji.
    total_score = (
        (0.40 * trend_score)
        + (0.25 * rsi_score)
        + (0.20 * bb_score)
        + (0.15 * momentum_score)
    )

    total_score = float(np.clip(total_score, -1.0, 1.0))

    analyzed_candle_time = str(df["datetime"].iloc[-1])

    return TimeframeResult(
        timeframe=timeframe,
        close=last_close,
        rsi=last_rsi,
        sma=last_sma,
        bb_upper=last_bb_upper,
        bb_lower=last_bb_lower,
        bb_mid=last_bb_middle,
        atr=last_atr,
        score=total_score,
        trend_score=trend_score,
        rsi_score=rsi_score,
        bb_score=bb_score,
        momentum_score=momentum_score,
        analyzed_candle_time=analyzed_candle_time,
        data_points=len(df),
    )


# =============================================================================
# 7. HELPER AGREGASI MULTI-TIMEFRAME
# =============================================================================
def _calculate_higher_timeframe_trend(
    timeframe_results: dict[str, TimeframeResult],
) -> tuple[str, float]:
    """
    Menentukan tren utama menggunakan 30m dan 1h.

    Jika keduanya tidak tersedia, fungsi memakai semua timeframe yang tersedia.
    """
    selected_timeframes = [
        timeframe
        for timeframe in HIGHER_TIMEFRAMES
        if timeframe in timeframe_results
    ]

    if not selected_timeframes:
        selected_timeframes = list(timeframe_results.keys())

    total_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in selected_timeframes
    )

    if total_weight <= 0:
        return "Sideways", 0.0

    weighted_score = sum(
        timeframe_results[timeframe].score
        * TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in selected_timeframes
    )

    trend_score = weighted_score / total_weight

    if trend_score >= TIMEFRAME_BIAS_THRESHOLD:
        return "Bullish", trend_score

    if trend_score <= -TIMEFRAME_BIAS_THRESHOLD:
        return "Bearish", trend_score

    return "Sideways", trend_score


def _calculate_directional_agreement(
    timeframe_results: dict[str, TimeframeResult],
    candidate_signal: str,
) -> float:
    """
    Menghitung persentase bobot timeframe yang mendukung calon sinyal.
    """
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    available_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in timeframe_results
    )

    if available_weight <= 0:
        return 0.0

    if candidate_signal == "BUY":
        supporting_weight = sum(
            TIMEFRAME_WEIGHTS[timeframe]
            for timeframe, result in timeframe_results.items()
            if result.score >= TIMEFRAME_BIAS_THRESHOLD
        )
    else:
        supporting_weight = sum(
            TIMEFRAME_WEIGHTS[timeframe]
            for timeframe, result in timeframe_results.items()
            if result.score <= -TIMEFRAME_BIAS_THRESHOLD
        )

    return supporting_weight / available_weight


def _calculate_confidence_score(
    combined_score: float,
    directional_agreement: float,
    coverage_ratio: float,
) -> float:
    """
    Menghitung Confidence Score konfluensi internal.

    Komponen:
    - 50% kekuatan combined score
    - 30% keselarasan timeframe
    - 20% kelengkapan data timeframe

    Skor ini bukan probabilitas kemenangan.
    """
    normalized_strength = min(
        abs(combined_score) / 0.60,
        1.0,
    )

    confidence = 100.0 * (
        (0.50 * normalized_strength)
        + (0.30 * directional_agreement)
        + (0.20 * coverage_ratio)
    )

    return round(float(np.clip(confidence, 0.0, 100.0)), 1)


def _classify_signal_risk(
    signal: str,
    confidence_pct: float,
    directional_agreement: float,
    coverage_ratio: float,
) -> str:
    """
    Mengklasifikasikan risiko kualitas sinyal.

    Ini bukan pengukuran risiko akun atau position sizing.
    """
    if signal == "HOLD":
        return "Tinggi"

    if (
        confidence_pct >= 80.0
        and directional_agreement >= 0.80
        and coverage_ratio >= 0.90
    ):
        return "Rendah"

    if (
        confidence_pct >= MIN_CONFIDENCE_FOR_SIGNAL
        and directional_agreement >= MIN_DIRECTIONAL_AGREEMENT
    ):
        return "Sedang"

    return "Tinggi"


def _build_reasons(
    signal: str,
    timeframe_results: dict[str, TimeframeResult],
    combined_score: float,
    confidence_pct: float,
    coverage_ratio: float,
    directional_agreement: float,
    trend: str,
    missing_timeframes: list[str],
) -> list[str]:
    """
    Membuat alasan analisis yang mudah dipahami manusia.
    """
    reasons: list[str] = []

    if signal == "HOLD":
        if abs(combined_score) < SIGNAL_SCORE_THRESHOLD:
            reasons.append(
                "Kekuatan skor gabungan belum melewati ambang sinyal."
            )

        if confidence_pct < MIN_CONFIDENCE_FOR_SIGNAL:
            reasons.append(
                "Confidence Score belum memenuhi batas minimum."
            )

        if directional_agreement < MIN_DIRECTIONAL_AGREEMENT:
            reasons.append(
                "Arah antar-timeframe belum cukup selaras."
            )

        if coverage_ratio < MIN_COVERAGE_RATIO:
            reasons.append(
                "Data timeframe yang tersedia belum cukup lengkap."
            )

        if trend == "Sideways":
            reasons.append(
                "Timeframe besar belum menunjukkan tren yang jelas."
            )

        if not reasons:
            reasons.append(
                "Konfirmasi belum memenuhi seluruh aturan konservatif."
            )

    else:
        direction_word = "bullish" if signal == "BUY" else "bearish"

        if trend in {"Bullish", "Bearish"}:
            reasons.append(
                f"Tren timeframe besar teridentifikasi {trend.lower()}."
            )

        for timeframe in reversed(TIMEFRAMES):
            result = timeframe_results.get(timeframe)
            if result is None:
                continue

            supports_signal = (
                signal == "BUY"
                and result.score >= TIMEFRAME_BIAS_THRESHOLD
            ) or (
                signal == "SELL"
                and result.score <= -TIMEFRAME_BIAS_THRESHOLD
            )

            if not supports_signal:
                continue

            label = TIMEFRAME_LABELS.get(timeframe, timeframe)
            price_position = (
                "di atas"
                if result.close > result.sma
                else "di bawah"
            )

            reasons.append(
                f"{label} mendukung {direction_word}: harga {price_position} "
                f"SMA20 dan skor timeframe {result.score:+.2f}."
            )

            if len(reasons) >= 4:
                break

        if signal == "BUY":
            supporting_rsi = [
                result.rsi
                for result in timeframe_results.values()
                if result.score >= TIMEFRAME_BIAS_THRESHOLD
            ]
            if supporting_rsi and max(supporting_rsi) < 75:
                reasons.append(
                    "RSI timeframe pendukung belum berada pada ekstrem bullish."
                )
        else:
            supporting_rsi = [
                result.rsi
                for result in timeframe_results.values()
                if result.score <= -TIMEFRAME_BIAS_THRESHOLD
            ]
            if supporting_rsi and min(supporting_rsi) > 25:
                reasons.append(
                    "RSI timeframe pendukung belum berada pada ekstrem bearish."
                )

    if missing_timeframes:
        labels = [
            TIMEFRAME_LABELS.get(timeframe, timeframe)
            for timeframe in missing_timeframes
        ]
        reasons.append(
            "Peringatan: data tidak tersedia untuk timeframe "
            + ", ".join(labels)
            + "."
        )

    return reasons[:6]


# =============================================================================
# 8. FUNGSI UTAMA ANALISIS
# =============================================================================
async def _build_market_analysis(
    symbol: str = DEFAULT_SYMBOL,
) -> dict:
    """
    Mengambil dan menganalisis market secara multi-timeframe.

    Return utama:
        signal:
            BUY | SELL | HOLD

        direction:
            BUY | SELL | NEUTRAL
            Field kompatibilitas untuk kode lama.

        confidence_pct:
            Skor kualitas konfluensi internal 0-100.

        tp:
            Alias kompatibilitas untuk tp2.
    """
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY belum diatur pada environment variable."
        )

    timeout = aiohttp.ClientTimeout(
        total=REQUEST_TIMEOUT_SECONDS
    )

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            fetch_tasks = [
                _fetch_ohlc(session, symbol, timeframe)
                for timeframe in TIMEFRAMES
            ]

            dataframes = await asyncio.gather(*fetch_tasks)

    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Permintaan data pasar melewati batas waktu."
        ) from exc

    except aiohttp.ClientError as exc:
        raise RuntimeError(
            "Tidak dapat terhubung ke Twelve Data."
        ) from exc

    except Exception as exc:
        logger.exception(
            "Gagal membuat sesi atau mengambil data | symbol=%s",
            symbol,
        )
        raise RuntimeError(
            "Tidak dapat mengambil data pasar."
        ) from exc

    timeframe_results: dict[str, TimeframeResult] = {}
    latest_prices: dict[str, float] = {}
    latest_data_times: dict[str, str] = {}

    for timeframe, dataframe in zip(TIMEFRAMES, dataframes):
        if dataframe is None or dataframe.empty:
            continue

        latest_prices[timeframe] = _safe_float(
            dataframe["close"].iloc[-1],
            f"latest price {timeframe}",
        )
        latest_data_times[timeframe] = str(
            dataframe["datetime"].iloc[-1]
        )

        analysis_dataframe = dataframe

        if EXCLUDE_LATEST_CANDLE_FROM_INDICATORS:
            minimum_after_exclusion = (
                max(
                    RSI_PERIOD,
                    SMA_PERIOD,
                    BB_PERIOD,
                    ATR_PERIOD,
                )
                + MOMENTUM_LOOKBACK
                + 2
            )

            if len(dataframe) > minimum_after_exclusion:
                analysis_dataframe = dataframe.iloc[:-1].copy()

        result = _score_timeframe(
            analysis_dataframe,
            timeframe,
        )

        if result is not None:
            timeframe_results[timeframe] = result

    if not timeframe_results:
        raise RuntimeError(
            "Gagal mengambil atau menghitung data dari semua timeframe. "
            "Periksa API key, limit request, koneksi, atau symbol."
        )

    available_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in timeframe_results
    )
    total_possible_weight = sum(TIMEFRAME_WEIGHTS.values())

    combined_score = sum(
        result.score * TIMEFRAME_WEIGHTS[timeframe]
        for timeframe, result in timeframe_results.items()
    ) / available_weight

    coverage_ratio = available_weight / total_possible_weight

    if combined_score >= SIGNAL_SCORE_THRESHOLD:
        candidate_signal = "BUY"
    elif combined_score <= -SIGNAL_SCORE_THRESHOLD:
        candidate_signal = "SELL"
    else:
        candidate_signal = "HOLD"

    trend, higher_timeframe_score = (
        _calculate_higher_timeframe_trend(timeframe_results)
    )

    directional_agreement = _calculate_directional_agreement(
        timeframe_results,
        candidate_signal,
    )

    confidence_pct = _calculate_confidence_score(
        combined_score,
        directional_agreement,
        coverage_ratio,
    )

    enough_timeframes = (
        len(timeframe_results) >= MIN_TIMEFRAMES_FOR_SIGNAL
    )
    enough_coverage = coverage_ratio >= MIN_COVERAGE_RATIO
    enough_agreement = (
        directional_agreement >= MIN_DIRECTIONAL_AGREEMENT
    )
    enough_confidence = (
        confidence_pct >= MIN_CONFIDENCE_FOR_SIGNAL
    )

    trend_aligned = (
        candidate_signal == "BUY"
        and trend == "Bullish"
    ) or (
        candidate_signal == "SELL"
        and trend == "Bearish"
    )

    if (
        candidate_signal in {"BUY", "SELL"}
        and enough_timeframes
        and enough_coverage
        and enough_agreement
        and enough_confidence
        and trend_aligned
    ):
        signal = candidate_signal
    else:
        signal = "HOLD"

    # Field direction dipertahankan untuk kompatibilitas lama.
    direction = "NEUTRAL" if signal == "HOLD" else signal

    reference_timeframe = (
        ENTRY_TIMEFRAME_FOR_ATR
        if ENTRY_TIMEFRAME_FOR_ATR in timeframe_results
        else next(iter(timeframe_results))
    )

    reference_result = timeframe_results[reference_timeframe]
    entry_price = latest_prices.get(
        reference_timeframe,
        reference_result.close,
    )
    atr_value = reference_result.atr

    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None

    if signal == "BUY":
        stop_loss = (
            entry_price
            - (ATR_SL_MULTIPLIER * atr_value)
        )
        take_profit_1 = (
            entry_price
            + (ATR_TP1_MULTIPLIER * atr_value)
        )
        take_profit_2 = (
            entry_price
            + (ATR_TP2_MULTIPLIER * atr_value)
        )
        take_profit_3 = (
            entry_price
            + (ATR_TP3_MULTIPLIER * atr_value)
        )

    elif signal == "SELL":
        stop_loss = (
            entry_price
            + (ATR_SL_MULTIPLIER * atr_value)
        )
        take_profit_1 = (
            entry_price
            - (ATR_TP1_MULTIPLIER * atr_value)
        )
        take_profit_2 = (
            entry_price
            - (ATR_TP2_MULTIPLIER * atr_value)
        )
        take_profit_3 = (
            entry_price
            - (ATR_TP3_MULTIPLIER * atr_value)
        )

    missing_timeframes = [
        timeframe
        for timeframe in TIMEFRAMES
        if timeframe not in timeframe_results
    ]

    risk = _classify_signal_risk(
        signal,
        confidence_pct,
        directional_agreement,
        coverage_ratio,
    )

    reasons = _build_reasons(
        signal=signal,
        timeframe_results=timeframe_results,
        combined_score=combined_score,
        confidence_pct=confidence_pct,
        coverage_ratio=coverage_ratio,
        directional_agreement=directional_agreement,
        trend=trend,
        missing_timeframes=missing_timeframes,
    )

    logger.info(
        "Analisis multi-timeframe selesai | symbol=%s | signal=%s | "
        "score=%.3f | confidence=%.1f | agreement=%.2f | coverage=%.2f | "
        "trend=%s | available_tf=%s",
        symbol,
        signal,
        combined_score,
        confidence_pct,
        directional_agreement,
        coverage_ratio,
        trend,
        list(timeframe_results.keys()),
    )

    return {
        "symbol": symbol,
        "timeframes": timeframe_results,
        "combined_score": combined_score,
        "confidence_pct": confidence_pct,
        "directional_agreement_pct": round(
            directional_agreement * 100,
            1,
        ),
        "coverage_pct": round(coverage_ratio * 100, 1),
        "direction": direction,
        "signal": signal,
        "trend": trend,
        "higher_timeframe_score": higher_timeframe_score,
        "risk": risk,
        "reasons": reasons,
        "entry_price": entry_price,
        "atr_reference_tf": reference_timeframe,
        "atr_value": atr_value,
        "sl": stop_loss,
        "tp": take_profit_2,
        "tp1": take_profit_1,
        "tp2": take_profit_2,
        "tp3": take_profit_3,
        "missing_timeframes": missing_timeframes,
        "available_timeframes": list(timeframe_results.keys()),
        "latest_data_times": latest_data_times,
        "confidence_note": (
            "Confidence Score adalah skor konfluensi internal dan "
            "belum menjadi probabilitas kemenangan."
        ),
    }


async def get_market_data(
    symbol: str = DEFAULT_SYMBOL,
) -> dict:
    """
    Mengambil hasil analisis multi-timeframe dengan cache dan request coalescing.

    Permintaan identik untuk symbol yang sama tidak akan menjalankan analisis
    bersamaan. Hasil cache dikembalikan sebagai salinan agar pemanggil tidak
    dapat mengubah data yang tersimpan.
    """
    normalized_symbol = _normalize_symbol(symbol)

    cached_analysis = _get_cached_analysis(normalized_symbol)
    if cached_analysis is not None:
        logger.info(
            "Analysis cache hit | symbol=%s | age=%.1fs",
            normalized_symbol,
            cached_analysis.get("cache_age_seconds", 0.0),
        )
        return cached_analysis

    analysis_lock = _get_analysis_lock(normalized_symbol)

    async with analysis_lock:
        cached_analysis = _get_cached_analysis(normalized_symbol)
        if cached_analysis is not None:
            logger.info(
                "Analysis cache hit after lock | symbol=%s | age=%.1fs",
                normalized_symbol,
                cached_analysis.get("cache_age_seconds", 0.0),
            )
            return cached_analysis

        analysis = await _build_market_analysis(normalized_symbol)
        analysis["cache_hit"] = False
        analysis["cache_age_seconds"] = 0.0
        analysis["analysis_cache_ttl_seconds"] = (
            ANALYSIS_CACHE_TTL_SECONDS
        )

        _set_cached_analysis(normalized_symbol, analysis)

        return copy.deepcopy(analysis)


# =============================================================================
# 9. FORMAT PESAN TELEGRAM
# =============================================================================
def generate_signal_message(analysis: dict) -> str:
    """
    Mengubah hasil get_market_data() menjadi pesan Markdown Telegram.
    """
    symbol = analysis["symbol"]
    signal = analysis.get(
        "signal",
        analysis.get("direction", "HOLD"),
    )
    confidence = analysis["confidence_pct"]
    trend = analysis.get("trend", "Sideways")
    risk = analysis.get("risk", "Tinggi")

    emoji = {
        "BUY": "🟢",
        "SELL": "🔴",
        "HOLD": "🟡",
        "NEUTRAL": "🟡",
    }.get(signal, "⚪")

    decimals = _decimal_places(symbol)

    lines = [
        f"*📊 Analisis Multi-Timeframe — {symbol}*",
        "",
        f"{emoji} *Signal: {signal}*",
        f"*Confidence Score: {confidence:.1f}%*",
        f"*Trend: {trend}*",
        f"*Risk Sinyal: {risk}*",
        "",
        "*Alasan Analisis:*",
    ]

    reasons = analysis.get("reasons") or [
        "Belum ada alasan analisis yang tersedia."
    ]

    for reason in reasons:
        lines.append(f"• {reason}")

    lines.extend(
        [
            "",
            "*Detail per Timeframe:*",
        ]
    )

    for timeframe in TIMEFRAMES:
        result = analysis["timeframes"].get(timeframe)
        label = TIMEFRAME_LABELS.get(timeframe, timeframe)

        if result is None:
            lines.append(
                f"• *{label}*: data tidak tersedia"
            )
            continue

        bias = _timeframe_bias(result.score)

        price_position = (
            "di atas"
            if result.close > result.sma
            else "di bawah"
        )

        lines.append(
            f"• *{label}*: {bias} | "
            f"Score {result.score:+.2f} | "
            f"RSI {result.rsi:.1f} | "
            f"Harga {price_position} SMA20"
        )

    lines.append("")

    if signal in {"BUY", "SELL"}:
        entry = analysis["entry_price"]
        stop_loss = analysis["sl"]
        take_profit_1 = analysis["tp1"]
        take_profit_2 = analysis["tp2"]
        take_profit_3 = analysis["tp3"]

        lines.extend(
            [
                f"*Entry:* {entry:.{decimals}f}",
                f"*Stop Loss:* {stop_loss:.{decimals}f}",
                f"*Take Profit 1:* {take_profit_1:.{decimals}f}",
                f"*Take Profit 2:* {take_profit_2:.{decimals}f}",
                f"*Take Profit 3:* {take_profit_3:.{decimals}f}",
                "",
                (
                    f"_SL/TP menggunakan ATR({ATR_PERIOD}) timeframe "
                    f"{TIMEFRAME_LABELS.get(analysis['atr_reference_tf'], analysis['atr_reference_tf'])}._"
                ),
            ]
        )
    else:
        lines.append(
            "_Tidak ada entry karena konfirmasi belum memenuhi aturan konservatif._"
        )

    lines.extend(
        [
            "",
            (
                f"_Coverage data: {analysis.get('coverage_pct', 0):.1f}% | "
                f"Keselarasan timeframe: "
                f"{analysis.get('directional_agreement_pct', 0):.1f}%_"
            ),
            (
                "_Confidence Score adalah skor konfluensi internal, "
                "bukan probabilitas kemenangan._"
            ),
            "",
            (
                "⚠️ _Bukan saran finansial. Gunakan manajemen risiko "
                "dan position sizing sendiri._"
            ),
        ]
    )

    return "\n".join(lines)


# =============================================================================
# 10. CALLBACK HANDLER LEGACY
# =============================================================================
async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handler lama untuk tombol analyze_xauusd.

    Fungsi ini dipertahankan agar integrasi lama tetap dapat digunakan.
    Pada tahap berikutnya, tes_bot.py akan menjadi pemilik alur Telegram.
    """
    query = update.callback_query

    if query is None:
        return

    try:
        await query.answer()
    except Exception:
        logger.exception(
            "Gagal menjawab callback Telegram."
        )
        return

    if query.data != "analyze_xauusd":
        return

    refresh_keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔄 Refresh Analisa",
                    callback_data="analyze_xauusd",
                )
            ]
        ]
    )

    try:
        await query.edit_message_text(
            "⏳ Mengambil dan menganalisis data "
            "4 timeframe (5m/15m/30m/1h)..."
        )
    except Exception:
        logger.exception(
            "Gagal menampilkan pesan proses analisis."
        )
        return

    try:
        analysis = await get_market_data(DEFAULT_SYMBOL)
        message = generate_signal_message(analysis)

        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            reply_markup=refresh_keyboard,
        )

    except RuntimeError as exc:
        logger.warning(
            "Analisis pasar gagal | error=%s",
            exc,
        )

        try:
            await query.edit_message_text(
                f"❌ Gagal mengambil data pasar: {exc}",
                reply_markup=refresh_keyboard,
            )
        except Exception:
            logger.exception(
                "Gagal mengirim pesan RuntimeError."
            )

    except Exception:
        logger.exception(
            "Error tak terduga pada button handler."
        )

        try:
            await query.edit_message_text(
                "❌ Terjadi kesalahan tak terduga saat analisis. "
                "Silakan coba lagi beberapa saat.",
                reply_markup=refresh_keyboard,
            )
        except Exception:
            logger.exception(
                "Gagal mengirim pesan error tak terduga."
            )
