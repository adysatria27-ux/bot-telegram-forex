"""
================================================================================
Generic Multi-Asset Market Analysis Engine untuk Bot Telegram
================================================================================

Sumber data:
    Twelve Data

Tujuan:
    Menghasilkan analisis BUY, SELL, atau HOLD untuk forex, logam,
    crypto, dan indeks menggunakan satu pipeline analisis yang sama.

Indikator saat ini:
    - RSI 14
    - SMA 20
    - Bollinger Bands 20,2
    - ATR 14
    - Momentum harga
    - Support dan Resistance
    - Swing High dan Swing Low
    - HH, HL, LH, dan LL
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    - Sideways Detection
    - False Breakout Filter
    - Candlestick Pattern
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
MAX_OHLC_CACHE_ENTRIES = 128
MAX_ANALYSIS_CACHE_ENTRIES = 32


@dataclass(frozen=True)
class AssetConfig:
    """Konfigurasi satu aset tanpa menduplikasi pipeline analisis."""

    symbol: str
    provider_symbols: tuple[str, ...]
    asset_class: str
    decimals: int
    callback_data: str
    menu_label: str
    analysis_cache_ttl_seconds: int = ANALYSIS_CACHE_TTL_SECONDS
    aliases: tuple[str, ...] = ()
    legacy_menu_texts: tuple[str, ...] = ()


SUPPORTED_ASSETS: tuple[AssetConfig, ...] = (
    AssetConfig(
        symbol="XAU/USD",
        provider_symbols=("XAU/USD",),
        asset_class="metal",
        decimals=2,
        callback_data="analyze_xauusd",
        menu_label="XAU/USD",
        aliases=("XAUUSD", "GOLD"),
        legacy_menu_texts=("Cek Harga XAUUSD",),
    ),
    AssetConfig(
        symbol="XAG/USD",
        provider_symbols=("XAG/USD",),
        asset_class="metal",
        decimals=3,
        callback_data="analyze_xagusd",
        menu_label="XAG/USD",
        aliases=("XAGUSD", "SILVER"),
    ),
    AssetConfig(
        symbol="EUR/USD",
        provider_symbols=("EUR/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_eurusd",
        menu_label="EUR/USD",
        aliases=("EURUSD",),
        legacy_menu_texts=("Cek Harga EURUSD",),
    ),
    AssetConfig(
        symbol="GBP/USD",
        provider_symbols=("GBP/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_gbpusd",
        menu_label="GBP/USD",
        aliases=("GBPUSD",),
    ),
    AssetConfig(
        symbol="USD/JPY",
        provider_symbols=("USD/JPY",),
        asset_class="forex",
        decimals=3,
        callback_data="analyze_usdjpy",
        menu_label="USD/JPY",
        aliases=("USDJPY",),
    ),
    AssetConfig(
        symbol="AUD/USD",
        provider_symbols=("AUD/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_audusd",
        menu_label="AUD/USD",
        aliases=("AUDUSD",),
    ),
    AssetConfig(
        symbol="NZD/USD",
        provider_symbols=("NZD/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_nzdusd",
        menu_label="NZD/USD",
        aliases=("NZDUSD",),
    ),
    AssetConfig(
        symbol="USD/CAD",
        provider_symbols=("USD/CAD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_usdcad",
        menu_label="USD/CAD",
        aliases=("USDCAD",),
    ),
    AssetConfig(
        symbol="USD/CHF",
        provider_symbols=("USD/CHF",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_usdchf",
        menu_label="USD/CHF",
        aliases=("USDCHF",),
    ),
    AssetConfig(
        symbol="BTC/USD",
        provider_symbols=("BTC/USD",),
        asset_class="crypto",
        decimals=2,
        callback_data="analyze_btcusd",
        menu_label="BTC/USD",
        analysis_cache_ttl_seconds=20,
        aliases=("BTCUSD", "BITCOIN"),
    ),
    AssetConfig(
        symbol="ETH/USD",
        provider_symbols=("ETH/USD",),
        asset_class="crypto",
        decimals=2,
        callback_data="analyze_ethusd",
        menu_label="ETH/USD",
        analysis_cache_ttl_seconds=20,
        aliases=("ETHUSD", "ETHEREUM"),
    ),
    AssetConfig(
        symbol="NAS100",
        provider_symbols=("NDX", "NAS100"),
        asset_class="index",
        decimals=2,
        callback_data="analyze_nas100",
        menu_label="NAS100",
        aliases=("NDX", "NASDAQ100", "NASDAQ 100"),
    ),
    AssetConfig(
        symbol="US30",
        provider_symbols=("DJI", "DJIA", "US30"),
        asset_class="index",
        decimals=2,
        callback_data="analyze_us30",
        menu_label="US30",
        aliases=("DJI", "DJIA", "DOW", "DOW30"),
    ),
)


def _asset_key(value: str) -> str:
    """Membuat key pembanding yang toleran terhadap slash dan spasi."""
    return "".join(character for character in value.upper() if character.isalnum())


_ASSET_BY_SYMBOL: dict[str, AssetConfig] = {
    asset.symbol: asset for asset in SUPPORTED_ASSETS
}
_ASSET_ALIAS_LOOKUP: dict[str, str] = {}
_MENU_TEXT_LOOKUP: dict[str, str] = {}
_CALLBACK_SYMBOL_LOOKUP: dict[str, str] = {}

for _asset in SUPPORTED_ASSETS:
    for _alias in (_asset.symbol, _asset.menu_label, *_asset.aliases):
        _ASSET_ALIAS_LOOKUP[_asset_key(_alias)] = _asset.symbol

    _MENU_TEXT_LOOKUP[_asset.menu_label.casefold()] = _asset.symbol
    for _legacy_text in _asset.legacy_menu_texts:
        _MENU_TEXT_LOOKUP[_legacy_text.casefold()] = _asset.symbol

    if _asset.callback_data in _CALLBACK_SYMBOL_LOOKUP:
        raise RuntimeError(
            f"Callback Telegram duplikat: {_asset.callback_data}"
        )
    _CALLBACK_SYMBOL_LOOKUP[_asset.callback_data] = _asset.symbol


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

# Konfigurasi price action dan market structure.
SWING_WINDOW = 2
STRUCTURE_LOOKBACK = 80
LEVEL_CLUSTER_ATR_MULTIPLIER = 0.35
NEAR_LEVEL_ATR_MULTIPLIER = 0.60
BREAKOUT_ATR_BUFFER = 0.10
BREAKOUT_LOOKBACK = 4
FALSE_BREAKOUT_LOOKBACK = 3
SIDEWAYS_LOOKBACK = 20
SIDEWAYS_RANGE_ATR_THRESHOLD = 5.0
SIDEWAYS_EFFICIENCY_THRESHOLD = 0.30
SIDEWAYS_SLOPE_ATR_THRESHOLD = 0.15
MIN_STRUCTURE_CONFIRMATION = 0.18
MAX_SIDEWAYS_RATIO_FOR_SIGNAL = 0.60
MAX_FALSE_BREAKOUT_AGAINST_RATIO = 0.25


@dataclass(frozen=True)
class SwingPoint:
    """Swing high atau swing low yang sudah terkonfirmasi."""

    index: int
    price: float
    kind: str
    timestamp: str


@dataclass
class TimeframeResult:
    """
    Hasil analisis untuk satu timeframe.

    Seluruh field lama tetap dipertahankan. Field tambahan memiliki default
    agar kode eksternal yang masih membuat TimeframeResult versi lama tidak
    langsung rusak.
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
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    support_strength: int = 0
    resistance_strength: int = 0
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    swing_high_label: str = "N/A"
    swing_low_label: str = "N/A"
    market_structure: str = "Tidak cukup data"
    detected_trend: str = "Sideways"
    bos: Optional[str] = None
    choch: Optional[str] = None
    sideways: bool = False
    false_breakout: Optional[str] = None
    candlestick_patterns: tuple[str, ...] = ()
    structure_score: float = 0.0
    support_resistance_score: float = 0.0
    candlestick_score: float = 0.0
    sideways_efficiency: float = 0.0
    breakout_level: Optional[float] = None


@dataclass
class CacheEntry:
    """Menyimpan nilai cache beserta waktu pembuatan dan kedaluwarsa."""

    value: Any
    created_at: float
    expires_at: float


_OHLC_CACHE: dict[tuple[str, str], CacheEntry] = {}
_ANALYSIS_CACHE: dict[str, CacheEntry] = {}
_ANALYSIS_LOCKS: dict[str, asyncio.Lock] = {}
_PROVIDER_SYMBOL_CACHE: dict[str, str] = {}
_API_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _normalize_symbol(symbol: str) -> str:
    """Mengubah alias pengguna menjadi symbol canonical yang didukung."""
    if not isinstance(symbol, str):
        raise ValueError("Symbol harus berupa teks.")

    cleaned = symbol.strip()
    if not cleaned:
        raise ValueError("Symbol tidak boleh kosong.")

    canonical = _ASSET_ALIAS_LOOKUP.get(_asset_key(cleaned))
    if canonical is None:
        supported = ", ".join(asset.symbol for asset in SUPPORTED_ASSETS)
        raise ValueError(
            f"Symbol '{cleaned}' belum didukung. Pilihan: {supported}."
        )
    return canonical


def _normalize_provider_symbol(symbol: str) -> str:
    """Menormalkan ticker yang dikirim langsung ke data provider."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Provider symbol tidak boleh kosong.")
    return normalized


def get_asset_config(symbol: str) -> AssetConfig:
    """Mengambil konfigurasi aset menggunakan canonical symbol atau alias."""
    return _ASSET_BY_SYMBOL[_normalize_symbol(symbol)]


def get_supported_assets() -> tuple[AssetConfig, ...]:
    """Daftar aset terurut untuk menu dan integrasi eksternal."""
    return SUPPORTED_ASSETS


def get_supported_symbols() -> tuple[str, ...]:
    """Daftar canonical symbol yang diterima get_market_data()."""
    return tuple(asset.symbol for asset in SUPPORTED_ASSETS)


def resolve_symbol_from_menu_text(text: str) -> Optional[str]:
    """Menerjemahkan teks Reply Keyboard, termasuk menu versi lama."""
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    direct_match = _MENU_TEXT_LOOKUP.get(cleaned.casefold())
    if direct_match is not None:
        return direct_match

    try:
        return _normalize_symbol(cleaned)
    except ValueError:
        return None


def resolve_symbol_from_callback(callback_data: str) -> Optional[str]:
    """Menerjemahkan callback Telegram menjadi canonical symbol."""
    if not isinstance(callback_data, str):
        return None
    return _CALLBACK_SYMBOL_LOOKUP.get(callback_data)


def get_callback_data(symbol: str) -> str:
    """Mengambil callback Telegram untuk symbol yang didukung."""
    return get_asset_config(symbol).callback_data

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


def _prune_cache(
    cache: dict[Any, CacheEntry],
    max_entries: int,
) -> None:
    """Menghapus entry kedaluwarsa dan membatasi pertumbuhan memory."""
    now = time.monotonic()
    for key, entry in list(cache.items()):
        if now >= entry.expires_at:
            cache.pop(key, None)

    overflow = len(cache) - max_entries
    if overflow <= 0:
        return

    oldest_keys = sorted(
        cache,
        key=lambda key: cache[key].created_at,
    )[:overflow]
    for key in oldest_keys:
        cache.pop(key, None)


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
    _prune_cache(_OHLC_CACHE, MAX_OHLC_CACHE_ENTRIES)


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
    """Menyimpan hasil analisis lengkap dengan TTL per jenis aset."""
    canonical_symbol = _normalize_symbol(symbol)
    asset = _ASSET_BY_SYMBOL[canonical_symbol]
    now = time.monotonic()
    cached_value = copy.deepcopy(analysis)
    cached_value["cache_hit"] = False
    cached_value["cache_age_seconds"] = 0.0

    _ANALYSIS_CACHE[canonical_symbol] = CacheEntry(
        value=cached_value,
        created_at=now,
        expires_at=now + asset.analysis_cache_ttl_seconds,
    )
    _prune_cache(_ANALYSIS_CACHE, MAX_ANALYSIS_CACHE_ENTRIES)

def clear_caches(symbol: Optional[str] = None) -> None:
    """Menghapus cache seluruh aset atau satu canonical symbol."""
    if symbol is None:
        _OHLC_CACHE.clear()
        _ANALYSIS_CACHE.clear()
        _PROVIDER_SYMBOL_CACHE.clear()
        return

    canonical_symbol = _normalize_symbol(symbol)
    asset = _ASSET_BY_SYMBOL[canonical_symbol]
    _ANALYSIS_CACHE.pop(canonical_symbol, None)
    _PROVIDER_SYMBOL_CACHE.pop(canonical_symbol, None)

    provider_keys = {
        _normalize_provider_symbol(provider)
        for provider in asset.provider_symbols
    }
    for key in list(_OHLC_CACHE):
        if key[0] in provider_keys:
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
    """Jumlah desimal berdasarkan konfigurasi aset."""
    return get_asset_config(symbol).decimals

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
    normalized_symbol = _normalize_provider_symbol(symbol)

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


async def _resolve_provider_symbol(asset: AssetConfig) -> str:
    """
    Menentukan ticker provider sekali per process.

    Probe menggunakan timeframe 15 menit. Data probe otomatis masuk OHLC cache,
    sehingga total request pertama tetap maksimal empat ketika kandidat pertama
    valid. Fallback terutama disediakan untuk alias indeks.
    """
    cached_provider = _PROVIDER_SYMBOL_CACHE.get(asset.symbol)
    if cached_provider is not None:
        return cached_provider

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for provider_symbol in asset.provider_symbols:
            normalized_provider = _normalize_provider_symbol(provider_symbol)
            probe = await _fetch_ohlc(
                session,
                normalized_provider,
                ENTRY_TIMEFRAME_FOR_ATR,
            )
            if probe is not None and not probe.empty:
                _PROVIDER_SYMBOL_CACHE[asset.symbol] = normalized_provider
                logger.info(
                    "Provider symbol resolved | symbol=%s | provider=%s",
                    asset.symbol,
                    normalized_provider,
                )
                return normalized_provider

    candidates = ", ".join(asset.provider_symbols)
    raise RuntimeError(
        f"Data {asset.symbol} tidak tersedia dari Twelve Data "
        f"menggunakan ticker: {candidates}. Untuk indeks, periksa juga "
        "dukungan paket Twelve Data yang digunakan."
    )


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
# 6. PRICE ACTION, MARKET STRUCTURE, DAN CANDLESTICK
# =============================================================================
def _detect_swings(
    df: pd.DataFrame,
    window: int = SWING_WINDOW,
) -> list[SwingPoint]:
    """Mendeteksi swing high dan swing low yang sudah terkonfirmasi."""
    if window < 1 or len(df) < ((window * 2) + 1):
        return []

    swings: list[SwingPoint] = []
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    for index in range(window, len(df) - window):
        current_high = highs[index]
        current_low = lows[index]

        left_highs = highs[index - window:index]
        right_highs = highs[index + 1:index + window + 1]
        left_lows = lows[index - window:index]
        right_lows = lows[index + 1:index + window + 1]

        is_swing_high = (
            current_high > float(np.max(left_highs))
            and current_high >= float(np.max(right_highs))
        )
        is_swing_low = (
            current_low < float(np.min(left_lows))
            and current_low <= float(np.min(right_lows))
        )

        timestamp = str(df["datetime"].iloc[index])

        if is_swing_high:
            swings.append(
                SwingPoint(
                    index=index,
                    price=float(current_high),
                    kind="HIGH",
                    timestamp=timestamp,
                )
            )

        if is_swing_low:
            swings.append(
                SwingPoint(
                    index=index,
                    price=float(current_low),
                    kind="LOW",
                    timestamp=timestamp,
                )
            )

    return sorted(swings, key=lambda item: (item.index, item.kind))


def _compare_swing_values(
    latest: float,
    previous: float,
    higher_label: str,
    lower_label: str,
    tolerance: float,
) -> str:
    """Memberikan label HH/LH atau HL/LL dengan toleransi ATR."""
    if latest > previous + tolerance:
        return higher_label
    if latest < previous - tolerance:
        return lower_label
    return "EQ"


def _classify_market_structure(
    swings: list[SwingPoint],
    atr_value: float,
) -> dict:
    """Mengklasifikasikan HH, HL, LH, LL dan skor struktur."""
    swing_highs = [item for item in swings if item.kind == "HIGH"]
    swing_lows = [item for item in swings if item.kind == "LOW"]
    tolerance = max(atr_value * 0.05, np.finfo(float).eps)

    high_label = "N/A"
    low_label = "N/A"

    if len(swing_highs) >= 2:
        high_label = _compare_swing_values(
            swing_highs[-1].price,
            swing_highs[-2].price,
            "HH",
            "LH",
            tolerance,
        )

    if len(swing_lows) >= 2:
        low_label = _compare_swing_values(
            swing_lows[-1].price,
            swing_lows[-2].price,
            "HL",
            "LL",
            tolerance,
        )

    if high_label == "HH" and low_label == "HL":
        structure_score = 1.0
    elif high_label == "LH" and low_label == "LL":
        structure_score = -1.0
    else:
        components: list[float] = []
        if high_label == "HH":
            components.append(0.55)
        elif high_label == "LH":
            components.append(-0.55)

        if low_label == "HL":
            components.append(0.55)
        elif low_label == "LL":
            components.append(-0.55)

        structure_score = (
            float(np.mean(components)) if components else 0.0
        )

    if high_label == "N/A" and low_label == "N/A":
        structure_label = "Tidak cukup data"
    else:
        structure_label = f"{high_label}-{low_label}"

    return {
        "high_label": high_label,
        "low_label": low_label,
        "structure_label": structure_label,
        "structure_score": float(np.clip(structure_score, -1.0, 1.0)),
        "last_swing_high": swing_highs[-1].price if swing_highs else None,
        "last_swing_low": swing_lows[-1].price if swing_lows else None,
    }


def _cluster_swing_levels(
    points: list[SwingPoint],
    tolerance: float,
) -> list[dict]:
    """Menggabungkan swing berdekatan menjadi level support/resistance."""
    if not points:
        return []

    clusters: list[dict] = []

    for point in sorted(points, key=lambda item: item.price):
        matched_cluster: Optional[dict] = None

        for cluster in clusters:
            if abs(point.price - cluster["price"]) <= tolerance:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append(
                {
                    "price": point.price,
                    "strength": 1,
                    "last_index": point.index,
                }
            )
            continue

        strength = matched_cluster["strength"]
        matched_cluster["price"] = (
            (matched_cluster["price"] * strength) + point.price
        ) / (strength + 1)
        matched_cluster["strength"] = strength + 1
        matched_cluster["last_index"] = max(
            matched_cluster["last_index"],
            point.index,
        )

    return clusters


def _detect_support_resistance(
    swings: list[SwingPoint],
    current_close: float,
    atr_value: float,
    dataframe_length: int,
) -> tuple[Optional[float], Optional[float], int, int]:
    """Menentukan support dan resistance terdekat dari cluster swing."""
    recent_start = max(0, dataframe_length - STRUCTURE_LOOKBACK)
    recent_swings = [
        item for item in swings if item.index >= recent_start
    ]

    tolerance = max(
        atr_value * LEVEL_CLUSTER_ATR_MULTIPLIER,
        np.finfo(float).eps,
    )

    low_clusters = _cluster_swing_levels(
        [item for item in recent_swings if item.kind == "LOW"],
        tolerance,
    )
    high_clusters = _cluster_swing_levels(
        [item for item in recent_swings if item.kind == "HIGH"],
        tolerance,
    )

    support_candidates = [
        cluster
        for cluster in low_clusters
        if cluster["price"] <= current_close + tolerance
    ]
    resistance_candidates = [
        cluster
        for cluster in high_clusters
        if cluster["price"] >= current_close - tolerance
    ]

    if support_candidates:
        support_cluster = max(
            support_candidates,
            key=lambda item: (item["price"], item["strength"]),
        )
    elif low_clusters:
        support_cluster = min(
            low_clusters,
            key=lambda item: abs(item["price"] - current_close),
        )
    else:
        support_cluster = None

    if resistance_candidates:
        resistance_cluster = min(
            resistance_candidates,
            key=lambda item: (item["price"], -item["strength"]),
        )
    elif high_clusters:
        resistance_cluster = min(
            high_clusters,
            key=lambda item: abs(item["price"] - current_close),
        )
    else:
        resistance_cluster = None

    return (
        support_cluster["price"] if support_cluster else None,
        resistance_cluster["price"] if resistance_cluster else None,
        support_cluster["strength"] if support_cluster else 0,
        resistance_cluster["strength"] if resistance_cluster else 0,
    )


def _detect_break_event(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    atr_value: float,
    prior_structure_score: float,
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """Mendeteksi BOS atau CHoCH berdasarkan penutupan candle."""
    if len(df) < 2 or not swings:
        return None, None, None

    buffer = atr_value * BREAKOUT_ATR_BUFFER
    start_index = max(1, len(df) - BREAKOUT_LOOKBACK)
    events: list[tuple[int, str, float]] = []

    for index in range(start_index, len(df)):
        previous_close = float(df["close"].iloc[index - 1])
        current_close = float(df["close"].iloc[index])

        previous_highs = [
            item
            for item in swings
            if item.kind == "HIGH" and item.index < index
        ]
        previous_lows = [
            item
            for item in swings
            if item.kind == "LOW" and item.index < index
        ]

        if previous_highs:
            reference_high = max(
                previous_highs,
                key=lambda item: item.index,
            )
            if (
                previous_close <= reference_high.price + buffer
                and current_close > reference_high.price + buffer
            ):
                events.append((index, "BULLISH", reference_high.price))

        if previous_lows:
            reference_low = max(
                previous_lows,
                key=lambda item: item.index,
            )
            if (
                previous_close >= reference_low.price - buffer
                and current_close < reference_low.price - buffer
            ):
                events.append((index, "BEARISH", reference_low.price))

    if not events:
        return None, None, None

    _, direction, level = max(events, key=lambda item: item[0])

    if direction == "BULLISH" and prior_structure_score <= -0.35:
        return None, "BULLISH", level
    if direction == "BEARISH" and prior_structure_score >= 0.35:
        return None, "BEARISH", level

    return direction, None, level


def _detect_false_breakout(
    df: pd.DataFrame,
    support_level: Optional[float],
    resistance_level: Optional[float],
    atr_value: float,
) -> Optional[str]:
    """
    Mendeteksi breakout wick yang gagal ditutup di luar level.

    Return BULLISH berarti percobaan breakout ke atas gagal.
    Return BEARISH berarti percobaan breakout ke bawah gagal.
    """
    if df.empty:
        return None

    buffer = atr_value * BREAKOUT_ATR_BUFFER
    start_index = max(0, len(df) - FALSE_BREAKOUT_LOOKBACK)
    events: list[tuple[int, str]] = []

    for index in range(start_index, len(df)):
        row = df.iloc[index]
        candle_open = float(row["open"])
        candle_high = float(row["high"])
        candle_low = float(row["low"])
        candle_close = float(row["close"])

        if resistance_level is not None:
            failed_upside = (
                candle_high > resistance_level + buffer
                and candle_close < resistance_level
                and candle_open <= resistance_level + buffer
            )
            if failed_upside:
                events.append((index, "BULLISH"))

        if support_level is not None:
            failed_downside = (
                candle_low < support_level - buffer
                and candle_close > support_level
                and candle_open >= support_level - buffer
            )
            if failed_downside:
                events.append((index, "BEARISH"))

    if not events:
        return None

    return max(events, key=lambda item: item[0])[1]


def _candle_parts(row: pd.Series) -> dict:
    """Menghitung body dan wick candle secara aman."""
    candle_open = float(row["open"])
    candle_close = float(row["close"])
    candle_high = float(row["high"])
    candle_low = float(row["low"])
    candle_range = max(candle_high - candle_low, np.finfo(float).eps)
    body = abs(candle_close - candle_open)

    return {
        "open": candle_open,
        "close": candle_close,
        "high": candle_high,
        "low": candle_low,
        "range": candle_range,
        "body": body,
        "upper_wick": candle_high - max(candle_open, candle_close),
        "lower_wick": min(candle_open, candle_close) - candle_low,
        "bullish": candle_close > candle_open,
        "bearish": candle_close < candle_open,
    }


def _detect_candlestick_patterns(
    df: pd.DataFrame,
    atr_value: float,
) -> tuple[tuple[str, ...], float, bool]:
    """Mendeteksi pola candlestick pada candle terakhir yang sudah tutup."""
    if len(df) < 3:
        return (), 0.0, False

    last = _candle_parts(df.iloc[-1])
    previous = _candle_parts(df.iloc[-2])
    first = _candle_parts(df.iloc[-3])

    patterns: list[str] = []
    scores: list[float] = []

    lookback_index = max(0, len(df) - 7)
    earlier_close = float(df["close"].iloc[lookback_index])
    recent_context_close = float(df["close"].iloc[-2])
    downtrend_context = recent_context_close < earlier_close
    uptrend_context = recent_context_close > earlier_close

    doji = last["body"] <= (last["range"] * 0.10)
    if doji:
        patterns.append("Doji")

    bullish_engulfing = (
        last["bullish"]
        and previous["bearish"]
        and last["open"] <= previous["close"]
        and last["close"] >= previous["open"]
        and last["body"] >= previous["body"] * 0.90
    )
    bearish_engulfing = (
        last["bearish"]
        and previous["bullish"]
        and last["open"] >= previous["close"]
        and last["close"] <= previous["open"]
        and last["body"] >= previous["body"] * 0.90
    )

    if bullish_engulfing:
        patterns.append("Bullish Engulfing")
        scores.append(0.80)
    if bearish_engulfing:
        patterns.append("Bearish Engulfing")
        scores.append(-0.80)

    meaningful_body = max(last["body"], atr_value * 0.03)

    hammer = (
        downtrend_context
        and last["lower_wick"] >= meaningful_body * 2.0
        and last["lower_wick"] >= last["range"] * 0.50
        and last["upper_wick"] <= last["range"] * 0.20
    )
    shooting_star = (
        uptrend_context
        and last["upper_wick"] >= meaningful_body * 2.0
        and last["upper_wick"] >= last["range"] * 0.50
        and last["lower_wick"] <= last["range"] * 0.20
    )

    if hammer:
        patterns.append("Hammer")
        scores.append(0.65)
    if shooting_star:
        patterns.append("Shooting Star")
        scores.append(-0.65)

    bullish_pin_bar = (
        not hammer
        and last["lower_wick"] >= meaningful_body * 2.5
        and last["lower_wick"] >= last["range"] * 0.55
    )
    bearish_pin_bar = (
        not shooting_star
        and last["upper_wick"] >= meaningful_body * 2.5
        and last["upper_wick"] >= last["range"] * 0.55
    )

    if bullish_pin_bar:
        patterns.append("Bullish Pin Bar")
        scores.append(0.55)
    if bearish_pin_bar:
        patterns.append("Bearish Pin Bar")
        scores.append(-0.55)

    morning_star = (
        downtrend_context
        and first["bearish"]
        and first["body"] >= atr_value * 0.30
        and previous["body"] <= first["body"] * 0.55
        and last["bullish"]
        and last["body"] >= first["body"] * 0.40
        and last["close"] >= (first["open"] + first["close"]) / 2.0
    )
    evening_star = (
        uptrend_context
        and first["bullish"]
        and first["body"] >= atr_value * 0.30
        and previous["body"] <= first["body"] * 0.55
        and last["bearish"]
        and last["body"] >= first["body"] * 0.40
        and last["close"] <= (first["open"] + first["close"]) / 2.0
    )

    if morning_star:
        patterns.append("Morning Star")
        scores.append(1.0)
    if evening_star:
        patterns.append("Evening Star")
        scores.append(-1.0)

    positive_score = max([score for score in scores if score > 0], default=0.0)
    negative_score = min([score for score in scores if score < 0], default=0.0)
    pattern_score = float(
        np.clip(positive_score + negative_score, -1.0, 1.0)
    )

    return tuple(dict.fromkeys(patterns)), pattern_score, doji


def _detect_sideways_market(
    df: pd.DataFrame,
    atr_value: float,
    structure_score: float,
) -> tuple[bool, float]:
    """Mendeteksi market sideways dari range, efficiency, slope, dan struktur."""
    if len(df) < SIDEWAYS_LOOKBACK:
        return False, 1.0

    segment = df.iloc[-SIDEWAYS_LOOKBACK:]
    close = segment["close"].astype(float)
    market_range = float(segment["high"].max() - segment["low"].min())
    range_atr_ratio = market_range / max(atr_value, np.finfo(float).eps)

    travelled_distance = float(close.diff().abs().sum())
    net_distance = abs(float(close.iloc[-1] - close.iloc[0]))
    efficiency = (
        net_distance / travelled_distance
        if travelled_distance > 0
        else 0.0
    )

    slope_lookback = min(5, len(close) - 1)
    slope = abs(float(close.iloc[-1] - close.iloc[-1 - slope_lookback]))
    normalized_slope = slope / (
        max(atr_value, np.finfo(float).eps) * slope_lookback
    )

    conditions = [
        range_atr_ratio <= SIDEWAYS_RANGE_ATR_THRESHOLD,
        efficiency <= SIDEWAYS_EFFICIENCY_THRESHOLD,
        normalized_slope <= SIDEWAYS_SLOPE_ATR_THRESHOLD,
        abs(structure_score) < 0.55,
    ]

    return sum(conditions) >= 3, float(np.clip(efficiency, 0.0, 1.0))


def _derive_trend_label(
    sideways: bool,
    structure_score: float,
    trend_score: float,
    momentum_score: float,
) -> str:
    """Menentukan Bullish, Bearish, atau Sideways."""
    if sideways:
        return "Sideways"

    combined_direction = (
        (0.55 * structure_score)
        + (0.30 * trend_score)
        + (0.15 * momentum_score)
    )

    if combined_direction >= 0.20:
        return "Bullish"
    if combined_direction <= -0.20:
        return "Bearish"
    return "Sideways"


# =============================================================================
# 6. ANALISIS SATU TIMEFRAME
# =============================================================================
def _score_timeframe(
    df: pd.DataFrame,
    timeframe: str,
) -> Optional[TimeframeResult]:
    """Menghitung indikator, price action, dan skor konfluensi timeframe."""
    minimum_rows = (
        max(
            RSI_PERIOD,
            SMA_PERIOD,
            BB_PERIOD,
            ATR_PERIOD,
            SIDEWAYS_LOOKBACK,
        )
        + MOMENTUM_LOOKBACK
        + (SWING_WINDOW * 2)
        + 2
    )

    if len(df) < minimum_rows:
        logger.warning(
            "Data timeframe terlalu sedikit | tf=%s | tersedia=%s | minimum=%s",
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
    last_bb_upper = _safe_float(bb_upper.iloc[-1], "Bollinger upper")
    last_bb_lower = _safe_float(bb_lower.iloc[-1], "Bollinger lower")
    last_bb_middle = _safe_float(bb_middle.iloc[-1], "Bollinger middle")
    last_atr = _safe_float(atr_series.iloc[-1], "ATR")

    if last_atr <= 0:
        logger.warning("ATR tidak valid | tf=%s | atr=%s", timeframe, last_atr)
        return None

    trend_distance = (last_close - last_sma) / last_atr
    trend_score = float(np.clip(trend_distance / 1.5, -1.0, 1.0))
    rsi_score = float(np.clip((last_rsi - 50.0) / 20.0, -1.0, 1.0))

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

    momentum_start = _safe_float(
        close.iloc[-1 - MOMENTUM_LOOKBACK],
        "momentum start",
    )
    momentum_change = last_close - momentum_start
    momentum_score = float(
        np.clip(
            momentum_change / (last_atr * max(MOMENTUM_LOOKBACK, 1)),
            -1.0,
            1.0,
        )
    )

    swings = _detect_swings(df)
    structure = _classify_market_structure(swings, last_atr)

    support_level, resistance_level, support_strength, resistance_strength = (
        _detect_support_resistance(
            swings,
            last_close,
            last_atr,
            len(df),
        )
    )

    bos, choch, breakout_level = _detect_break_event(
        df,
        swings,
        last_atr,
        structure["structure_score"],
    )

    breakout_component = 0.0
    if bos == "BULLISH":
        breakout_component = 0.35
    elif bos == "BEARISH":
        breakout_component = -0.35
    elif choch == "BULLISH":
        breakout_component = 0.55
    elif choch == "BEARISH":
        breakout_component = -0.55

    structure_score = float(
        np.clip(
            (0.65 * structure["structure_score"]) + breakout_component,
            -1.0,
            1.0,
        )
    )

    false_breakout = _detect_false_breakout(
        df,
        support_level,
        resistance_level,
        last_atr,
    )

    candlestick_patterns, candlestick_score, doji_detected = (
        _detect_candlestick_patterns(df, last_atr)
    )

    support_resistance_score = 0.0

    if bos == "BULLISH":
        support_resistance_score += 0.80
    elif bos == "BEARISH":
        support_resistance_score -= 0.80

    if choch == "BULLISH":
        support_resistance_score += 1.0
    elif choch == "BEARISH":
        support_resistance_score -= 1.0

    if false_breakout == "BULLISH":
        support_resistance_score -= 0.70
    elif false_breakout == "BEARISH":
        support_resistance_score += 0.70

    near_distance = last_atr * NEAR_LEVEL_ATR_MULTIPLIER

    if (
        support_level is not None
        and abs(last_close - support_level) <= near_distance
    ):
        support_resistance_score += (
            0.30 if candlestick_score >= 0 else 0.12
        )

    if (
        resistance_level is not None
        and abs(last_close - resistance_level) <= near_distance
    ):
        support_resistance_score -= (
            0.30 if candlestick_score <= 0 else 0.12
        )

    support_resistance_score = float(
        np.clip(support_resistance_score, -1.0, 1.0)
    )

    sideways, sideways_efficiency = _detect_sideways_market(
        df,
        last_atr,
        structure_score,
    )

    detected_trend = _derive_trend_label(
        sideways,
        structure_score,
        trend_score,
        momentum_score,
    )

    total_score = (
        (0.24 * trend_score)
        + (0.14 * rsi_score)
        + (0.10 * bb_score)
        + (0.12 * momentum_score)
        + (0.24 * structure_score)
        + (0.09 * support_resistance_score)
        + (0.07 * candlestick_score)
    )

    if sideways:
        total_score *= 0.55

    if doji_detected and abs(candlestick_score) < 0.40:
        total_score *= 0.92

    total_score = float(np.clip(total_score, -1.0, 1.0))

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
        analyzed_candle_time=str(df["datetime"].iloc[-1]),
        data_points=len(df),
        support_level=support_level,
        resistance_level=resistance_level,
        support_strength=support_strength,
        resistance_strength=resistance_strength,
        last_swing_high=structure["last_swing_high"],
        last_swing_low=structure["last_swing_low"],
        swing_high_label=structure["high_label"],
        swing_low_label=structure["low_label"],
        market_structure=structure["structure_label"],
        detected_trend=detected_trend,
        bos=bos,
        choch=choch,
        sideways=sideways,
        false_breakout=false_breakout,
        candlestick_patterns=candlestick_patterns,
        structure_score=structure_score,
        support_resistance_score=support_resistance_score,
        candlestick_score=candlestick_score,
        sideways_efficiency=sideways_efficiency,
        breakout_level=breakout_level,
    )


# =============================================================================
# 7. HELPER AGREGASI MULTI-TIMEFRAME
# =============================================================================
def _calculate_higher_timeframe_trend(
    timeframe_results: dict[str, TimeframeResult],
) -> tuple[str, float]:
    """Menentukan tren utama menggunakan struktur 30m dan 1h."""
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

    bullish_weight = 0.0
    bearish_weight = 0.0
    weighted_score = 0.0

    for timeframe in selected_timeframes:
        result = timeframe_results[timeframe]
        weight = TIMEFRAME_WEIGHTS[timeframe]
        score_multiplier = 0.35 if result.sideways else 1.0
        weighted_score += result.score * weight * score_multiplier

        if result.detected_trend == "Bullish":
            bullish_weight += weight
        elif result.detected_trend == "Bearish":
            bearish_weight += weight

    trend_score = weighted_score / total_weight

    if (
        bullish_weight / total_weight >= 0.55
        and trend_score >= 0.10
    ):
        return "Bullish", trend_score

    if (
        bearish_weight / total_weight >= 0.55
        and trend_score <= -0.10
    ):
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


def _calculate_structure_confirmation(
    timeframe_results: dict[str, TimeframeResult],
    candidate_signal: str,
) -> float:
    """Mengukur konfirmasi struktur yang searah dengan calon sinyal."""
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    signal_sign = 1.0 if candidate_signal == "BUY" else -1.0
    target_direction = "BULLISH" if candidate_signal == "BUY" else "BEARISH"
    supporting_false_breakout = (
        "BEARISH" if candidate_signal == "BUY" else "BULLISH"
    )

    total_weight = 0.0
    confirmed_weight = 0.0

    for timeframe, result in timeframe_results.items():
        weight = TIMEFRAME_WEIGHTS[timeframe]
        confirmation = max(0.0, signal_sign * result.structure_score)

        if result.bos == target_direction:
            confirmation = max(confirmation, 0.75)
        if result.choch == target_direction:
            confirmation = max(confirmation, 0.90)
        if result.false_breakout == supporting_false_breakout:
            confirmation = max(confirmation, 0.55)
        if result.sideways:
            confirmation *= 0.50

        total_weight += weight
        confirmed_weight += confirmation * weight

    return confirmed_weight / total_weight if total_weight > 0 else 0.0


def _calculate_pattern_confirmation(
    timeframe_results: dict[str, TimeframeResult],
    candidate_signal: str,
) -> float:
    """Mengukur konfirmasi pola candlestick searah calon sinyal."""
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    signal_sign = 1.0 if candidate_signal == "BUY" else -1.0
    total_weight = 0.0
    confirmed_weight = 0.0

    for timeframe, result in timeframe_results.items():
        weight = TIMEFRAME_WEIGHTS[timeframe]
        confirmation = max(0.0, signal_sign * result.candlestick_score)
        total_weight += weight
        confirmed_weight += confirmation * weight

    return confirmed_weight / total_weight if total_weight > 0 else 0.0


def _calculate_sideways_ratio(
    timeframe_results: dict[str, TimeframeResult],
) -> float:
    """Menghitung bobot timeframe yang terdeteksi sideways."""
    total_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in timeframe_results
    )
    if total_weight <= 0:
        return 0.0

    sideways_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe, result in timeframe_results.items()
        if result.sideways
    )
    return sideways_weight / total_weight


def _calculate_false_breakout_against_ratio(
    timeframe_results: dict[str, TimeframeResult],
    candidate_signal: str,
) -> float:
    """Mengukur false breakout yang bertentangan dengan calon sinyal."""
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    against_direction = (
        "BULLISH" if candidate_signal == "BUY" else "BEARISH"
    )
    total_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in timeframe_results
    )
    if total_weight <= 0:
        return 0.0

    against_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe, result in timeframe_results.items()
        if result.false_breakout == against_direction
    )
    return against_weight / total_weight


def _calculate_confidence_score(
    combined_score: float,
    directional_agreement: float,
    coverage_ratio: float,
    structure_confirmation: float = 0.0,
    pattern_confirmation: float = 0.0,
    sideways_ratio: float = 0.0,
    false_breakout_against_ratio: float = 0.0,
) -> float:
    """
    Menghitung Confidence Score konfluensi internal.

    Struktur market dan candlestick dapat meningkatkan confidence jika searah.
    Sideways dan false breakout yang melawan arah mengurangi confidence.
    """
    normalized_strength = min(abs(combined_score) / 0.60, 1.0)

    confidence = 100.0 * (
        (0.38 * normalized_strength)
        + (0.22 * directional_agreement)
        + (0.15 * coverage_ratio)
        + (0.17 * structure_confirmation)
        + (0.08 * pattern_confirmation)
    )

    penalty_multiplier = (
        1.0
        - (0.22 * float(np.clip(sideways_ratio, 0.0, 1.0)))
        - (
            0.35
            * float(
                np.clip(false_breakout_against_ratio, 0.0, 1.0)
            )
        )
    )

    confidence *= max(0.35, penalty_multiplier)

    return round(float(np.clip(confidence, 0.0, 100.0)), 1)


def _classify_signal_risk(
    signal: str,
    confidence_pct: float,
    directional_agreement: float,
    coverage_ratio: float,
    sideways_ratio: float = 0.0,
    false_breakout_against_ratio: float = 0.0,
) -> str:
    """Mengklasifikasikan risiko kualitas sinyal."""
    if signal == "HOLD":
        return "Tinggi"

    if false_breakout_against_ratio > 0.0 or sideways_ratio > 0.50:
        return "Tinggi"

    if (
        confidence_pct >= 80.0
        and directional_agreement >= 0.80
        and coverage_ratio >= 0.90
        and sideways_ratio <= 0.25
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
    structure_confirmation: float = 0.0,
    pattern_confirmation: float = 0.0,
    sideways_ratio: float = 0.0,
    false_breakout_against_ratio: float = 0.0,
) -> list[str]:
    """Membuat alasan analisis yang mudah dipahami manusia."""
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
        if trend == "Sideways" or sideways_ratio > 0.50:
            reasons.append(
                "Market terdeteksi sideways atau tren timeframe besar belum jelas."
            )
        if false_breakout_against_ratio > 0.0:
            reasons.append(
                "False breakout terdeteksi melawan calon arah sinyal."
            )
        if structure_confirmation < MIN_STRUCTURE_CONFIRMATION:
            reasons.append(
                "HH/HL, LH/LL, BOS, atau CHoCH belum memberi konfirmasi kuat."
            )
        if not reasons:
            reasons.append(
                "Konfirmasi belum memenuhi seluruh aturan konservatif."
            )

    else:
        target_direction = "BULLISH" if signal == "BUY" else "BEARISH"
        direction_word = "bullish" if signal == "BUY" else "bearish"

        if trend in {"Bullish", "Bearish"}:
            reasons.append(
                f"Tren timeframe besar teridentifikasi {trend.lower()}."
            )

        for timeframe in reversed(TIMEFRAMES):
            result = timeframe_results.get(timeframe)
            if result is None:
                continue

            label = TIMEFRAME_LABELS.get(timeframe, timeframe)

            if result.choch == target_direction:
                reasons.append(
                    f"CHoCH {direction_word} terkonfirmasi pada {label}."
                )
            elif result.bos == target_direction:
                reasons.append(
                    f"BOS {direction_word} terkonfirmasi pada {label}."
                )

            structure_supports = (
                signal == "BUY" and result.structure_score > 0.25
            ) or (
                signal == "SELL" and result.structure_score < -0.25
            )
            if structure_supports:
                reasons.append(
                    f"Struktur {label} membentuk {result.market_structure}."
                )

            pattern_supports = (
                signal == "BUY" and result.candlestick_score > 0.25
            ) or (
                signal == "SELL" and result.candlestick_score < -0.25
            )
            if pattern_supports and result.candlestick_patterns:
                reasons.append(
                    f"Pola {result.candlestick_patterns[0]} mendukung pada {label}."
                )

            if len(reasons) >= 5:
                break

        if structure_confirmation >= 0.45:
            reasons.append(
                "Konfirmasi market structure searah dengan sinyal."
            )
        if pattern_confirmation >= 0.40:
            reasons.append(
                "Candlestick pattern meningkatkan konfluensi sinyal."
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


def _calculate_risk_reward(
    entry: Optional[float],
    stop_loss: Optional[float],
    target: Optional[float],
) -> Optional[float]:
    """Menghitung reward terhadap satu unit risk."""
    if entry is None or stop_loss is None or target is None:
        return None

    risk_distance = abs(entry - stop_loss)
    if risk_distance <= np.finfo(float).eps:
        return None

    reward_distance = abs(target - entry)
    return round(reward_distance / risk_distance, 2)


def _build_indicator_checklist(
    candidate_signal: str,
    signal: str,
    trend: str,
    confidence_pct: float,
    coverage_ratio: float,
    directional_agreement: float,
    structure_confirmation: float,
    pattern_confirmation: float,
    sideways_ratio: float,
    false_breakout_filter_passed: bool,
    timeframe_results: dict[str, TimeframeResult],
    reference_result: TimeframeResult,
) -> list[dict[str, str]]:
    """Membuat checklist indikator yang stabil untuk API dan Telegram."""
    checklist: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checklist.append(
            {"name": name, "status": status, "detail": detail}
        )

    if coverage_ratio >= 0.90:
        add("Data multi-timeframe", "PASS", f"Coverage {coverage_ratio * 100:.0f}%")
    elif coverage_ratio >= MIN_COVERAGE_RATIO:
        add("Data multi-timeframe", "WARN", f"Coverage {coverage_ratio * 100:.0f}%")
    else:
        add("Data multi-timeframe", "FAIL", f"Coverage {coverage_ratio * 100:.0f}%")

    target_trend = "Bullish" if candidate_signal == "BUY" else "Bearish"
    if candidate_signal == "HOLD" or trend == "Sideways":
        add("Trend timeframe besar", "WARN", trend)
    elif trend == target_trend:
        add("Trend timeframe besar", "PASS", trend)
    else:
        add("Trend timeframe besar", "FAIL", trend)

    if directional_agreement >= 0.75:
        add("Keselarasan timeframe", "PASS", f"{directional_agreement * 100:.0f}%")
    elif directional_agreement >= MIN_DIRECTIONAL_AGREEMENT:
        add("Keselarasan timeframe", "WARN", f"{directional_agreement * 100:.0f}%")
    else:
        add("Keselarasan timeframe", "FAIL", f"{directional_agreement * 100:.0f}%")

    if confidence_pct >= 75:
        add("Confidence", "PASS", f"{confidence_pct:.1f}%")
    elif confidence_pct >= MIN_CONFIDENCE_FOR_SIGNAL:
        add("Confidence", "WARN", f"{confidence_pct:.1f}%")
    else:
        add("Confidence", "FAIL", f"{confidence_pct:.1f}%")

    if structure_confirmation >= 0.45:
        add("Market structure", "PASS", f"{structure_confirmation * 100:.0f}%")
    elif structure_confirmation >= MIN_STRUCTURE_CONFIRMATION:
        add("Market structure", "WARN", f"{structure_confirmation * 100:.0f}%")
    else:
        add("Market structure", "FAIL", reference_result.market_structure)

    target_direction = "BULLISH" if candidate_signal == "BUY" else "BEARISH"
    structure_events = [
        event
        for result in timeframe_results.values()
        for event in (result.choch, result.bos)
        if event is not None
    ]
    if candidate_signal in {"BUY", "SELL"} and target_direction in structure_events:
        add("BOS / CHoCH", "PASS", f"Konfirmasi {target_direction.title()}")
    elif structure_events:
        add("BOS / CHoCH", "FAIL", "Event berlawanan atau belum selaras")
    else:
        add("BOS / CHoCH", "WARN", "Belum ada break valid")

    if candidate_signal == "BUY" and reference_result.support_level is not None:
        distance = abs(reference_result.close - reference_result.support_level)
        status = "PASS" if distance <= reference_result.atr * NEAR_LEVEL_ATR_MULTIPLIER else "WARN"
        add("Support / Resistance", status, "Support tersedia")
    elif candidate_signal == "SELL" and reference_result.resistance_level is not None:
        distance = abs(reference_result.resistance_level - reference_result.close)
        status = "PASS" if distance <= reference_result.atr * NEAR_LEVEL_ATR_MULTIPLIER else "WARN"
        add("Support / Resistance", status, "Resistance tersedia")
    else:
        add("Support / Resistance", "WARN", "Belum menjadi konfirmasi utama")

    if pattern_confirmation >= 0.35:
        add("Candlestick pattern", "PASS", f"{pattern_confirmation * 100:.0f}%")
    elif pattern_confirmation > 0:
        add("Candlestick pattern", "WARN", f"{pattern_confirmation * 100:.0f}%")
    elif reference_result.candlestick_patterns:
        add("Candlestick pattern", "WARN", ", ".join(reference_result.candlestick_patterns[:2]))
    else:
        add("Candlestick pattern", "WARN", "Tidak ada pola kuat")

    if sideways_ratio <= 0.25:
        add("Sideways filter", "PASS", f"{sideways_ratio * 100:.0f}% sideways")
    elif sideways_ratio <= MAX_SIDEWAYS_RATIO_FOR_SIGNAL:
        add("Sideways filter", "WARN", f"{sideways_ratio * 100:.0f}% sideways")
    else:
        add("Sideways filter", "FAIL", f"{sideways_ratio * 100:.0f}% sideways")

    add(
        "False breakout filter",
        "PASS" if false_breakout_filter_passed else "FAIL",
        "Lolos" if false_breakout_filter_passed else "Terdeteksi risiko false breakout",
    )

    if signal in {"BUY", "SELL"}:
        add("Keputusan final", "PASS", signal)
    else:
        add("Keputusan final", "WARN", "HOLD — tunggu konfirmasi")

    return checklist


# =============================================================================
# 8. FUNGSI UTAMA ANALISIS
# =============================================================================
async def _build_market_analysis(
    symbol: str = DEFAULT_SYMBOL,
) -> dict:
    """Mengambil dan menganalisis market secara multi-timeframe."""
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY belum diatur pada environment variable."
        )

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

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
        raise RuntimeError("Tidak dapat mengambil data pasar.") from exc

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
                    SIDEWAYS_LOOKBACK,
                )
                + MOMENTUM_LOOKBACK
                + (SWING_WINDOW * 2)
                + 2
            )
            if len(dataframe) > minimum_after_exclusion:
                analysis_dataframe = dataframe.iloc[:-1].copy()

        result = _score_timeframe(analysis_dataframe, timeframe)
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

    trend, higher_timeframe_score = _calculate_higher_timeframe_trend(
        timeframe_results
    )
    directional_agreement = _calculate_directional_agreement(
        timeframe_results,
        candidate_signal,
    )
    structure_confirmation = _calculate_structure_confirmation(
        timeframe_results,
        candidate_signal,
    )
    pattern_confirmation = _calculate_pattern_confirmation(
        timeframe_results,
        candidate_signal,
    )
    sideways_ratio = _calculate_sideways_ratio(timeframe_results)
    false_breakout_against_ratio = (
        _calculate_false_breakout_against_ratio(
            timeframe_results,
            candidate_signal,
        )
    )

    confidence_pct = _calculate_confidence_score(
        combined_score,
        directional_agreement,
        coverage_ratio,
        structure_confirmation,
        pattern_confirmation,
        sideways_ratio,
        false_breakout_against_ratio,
    )

    enough_timeframes = len(timeframe_results) >= MIN_TIMEFRAMES_FOR_SIGNAL
    enough_coverage = coverage_ratio >= MIN_COVERAGE_RATIO
    enough_agreement = (
        directional_agreement >= MIN_DIRECTIONAL_AGREEMENT
    )
    enough_confidence = confidence_pct >= MIN_CONFIDENCE_FOR_SIGNAL
    enough_structure = (
        structure_confirmation >= MIN_STRUCTURE_CONFIRMATION
        or abs(combined_score) >= 0.45
    )
    not_too_sideways = sideways_ratio <= MAX_SIDEWAYS_RATIO_FOR_SIGNAL
    false_breakout_filter_passed = (
        false_breakout_against_ratio
        <= MAX_FALSE_BREAKOUT_AGAINST_RATIO
    )

    higher_timeframe_false_breakout = any(
        (
            candidate_signal == "BUY"
            and timeframe_results[timeframe].false_breakout == "BULLISH"
        )
        or (
            candidate_signal == "SELL"
            and timeframe_results[timeframe].false_breakout == "BEARISH"
        )
        for timeframe in HIGHER_TIMEFRAMES
        if timeframe in timeframe_results
    )

    trend_aligned = (
        candidate_signal == "BUY" and trend == "Bullish"
    ) or (
        candidate_signal == "SELL" and trend == "Bearish"
    )

    if (
        candidate_signal in {"BUY", "SELL"}
        and enough_timeframes
        and enough_coverage
        and enough_agreement
        and enough_confidence
        and enough_structure
        and not_too_sideways
        and false_breakout_filter_passed
        and not higher_timeframe_false_breakout
        and trend_aligned
    ):
        signal = candidate_signal
    else:
        signal = "HOLD"

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
        stop_loss = entry_price - (ATR_SL_MULTIPLIER * atr_value)
        take_profit_1 = entry_price + (ATR_TP1_MULTIPLIER * atr_value)
        take_profit_2 = entry_price + (ATR_TP2_MULTIPLIER * atr_value)
        take_profit_3 = entry_price + (ATR_TP3_MULTIPLIER * atr_value)
    elif signal == "SELL":
        stop_loss = entry_price + (ATR_SL_MULTIPLIER * atr_value)
        take_profit_1 = entry_price - (ATR_TP1_MULTIPLIER * atr_value)
        take_profit_2 = entry_price - (ATR_TP2_MULTIPLIER * atr_value)
        take_profit_3 = entry_price - (ATR_TP3_MULTIPLIER * atr_value)

    risk_reward_tp1 = _calculate_risk_reward(
        entry_price if signal in {"BUY", "SELL"} else None,
        stop_loss,
        take_profit_1,
    )
    risk_reward_tp2 = _calculate_risk_reward(
        entry_price if signal in {"BUY", "SELL"} else None,
        stop_loss,
        take_profit_2,
    )
    risk_reward_tp3 = _calculate_risk_reward(
        entry_price if signal in {"BUY", "SELL"} else None,
        stop_loss,
        take_profit_3,
    )

    indicator_checklist = _build_indicator_checklist(
        candidate_signal=candidate_signal,
        signal=signal,
        trend=trend,
        confidence_pct=confidence_pct,
        coverage_ratio=coverage_ratio,
        directional_agreement=directional_agreement,
        structure_confirmation=structure_confirmation,
        pattern_confirmation=pattern_confirmation,
        sideways_ratio=sideways_ratio,
        false_breakout_filter_passed=(
            false_breakout_filter_passed
            and not higher_timeframe_false_breakout
        ),
        timeframe_results=timeframe_results,
        reference_result=reference_result,
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
        sideways_ratio,
        false_breakout_against_ratio,
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
        structure_confirmation=structure_confirmation,
        pattern_confirmation=pattern_confirmation,
        sideways_ratio=sideways_ratio,
        false_breakout_against_ratio=false_breakout_against_ratio,
    )

    logger.info(
        "Analisis selesai | symbol=%s | signal=%s | score=%.3f | "
        "confidence=%.1f | structure=%.2f | pattern=%.2f | "
        "sideways=%.2f | false_breakout=%.2f | trend=%s",
        symbol,
        signal,
        combined_score,
        confidence_pct,
        structure_confirmation,
        pattern_confirmation,
        sideways_ratio,
        false_breakout_against_ratio,
        trend,
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
        "entry": entry_price if signal in {"BUY", "SELL"} else None,
        "atr_reference_tf": reference_timeframe,
        "atr_value": atr_value,
        "sl": stop_loss,
        "tp": take_profit_2,
        "tp1": take_profit_1,
        "tp2": take_profit_2,
        "tp3": take_profit_3,
        "risk_reward": {
            "tp1": risk_reward_tp1,
            "tp2": risk_reward_tp2,
            "tp3": risk_reward_tp3,
        },
        "risk_reward_tp1": risk_reward_tp1,
        "risk_reward_tp2": risk_reward_tp2,
        "risk_reward_tp3": risk_reward_tp3,
        "rr": risk_reward_tp2,
        "indicator_checklist": indicator_checklist,
        "missing_timeframes": missing_timeframes,
        "available_timeframes": list(timeframe_results.keys()),
        "latest_data_times": latest_data_times,
        "support": reference_result.support_level,
        "resistance": reference_result.resistance_level,
        "market_structure": reference_result.market_structure,
        "bos": reference_result.bos,
        "choch": reference_result.choch,
        "sideways": reference_result.sideways,
        "false_breakout": reference_result.false_breakout,
        "candlestick_patterns": list(
            reference_result.candlestick_patterns
        ),
        "structure_confirmation_pct": round(
            structure_confirmation * 100,
            1,
        ),
        "pattern_confirmation_pct": round(
            pattern_confirmation * 100,
            1,
        ),
        "sideways_ratio_pct": round(sideways_ratio * 100, 1),
        "false_breakout_against_pct": round(
            false_breakout_against_ratio * 100,
            1,
        ),
        "false_breakout_filter_passed": (
            false_breakout_filter_passed
            and not higher_timeframe_false_breakout
        ),
        "confidence_note": (
            "Confidence Score adalah skor konfluensi internal dan "
            "belum menjadi probabilitas kemenangan."
        ),
    }


async def get_market_data(
    symbol: str = DEFAULT_SYMBOL,
) -> dict:
    """
    Generic public API untuk seluruh aset yang didukung.

    Contoh:
        await get_market_data("XAU/USD")
        await get_market_data("EURUSD")
        await get_market_data("NAS100")

    Hasil cache dikembalikan sebagai deep copy untuk mencegah mutasi silang.
    """
    canonical_symbol = _normalize_symbol(symbol)
    asset = _ASSET_BY_SYMBOL[canonical_symbol]

    cached_analysis = _get_cached_analysis(canonical_symbol)
    if cached_analysis is not None:
        logger.info(
            "Analysis cache hit | symbol=%s | age=%.1fs",
            canonical_symbol,
            cached_analysis.get("cache_age_seconds", 0.0),
        )
        cached_analysis["requested_symbol"] = symbol
        return cached_analysis

    analysis_lock = _get_analysis_lock(canonical_symbol)

    async with analysis_lock:
        cached_analysis = _get_cached_analysis(canonical_symbol)
        if cached_analysis is not None:
            logger.info(
                "Analysis cache hit after lock | symbol=%s | age=%.1fs",
                canonical_symbol,
                cached_analysis.get("cache_age_seconds", 0.0),
            )
            cached_analysis["requested_symbol"] = symbol
            return cached_analysis

        provider_symbol = await _resolve_provider_symbol(asset)
        analysis = await _build_market_analysis(provider_symbol)

        analysis["symbol"] = canonical_symbol
        analysis["provider_symbol"] = provider_symbol
        analysis["asset_class"] = asset.asset_class
        analysis["decimal_places"] = asset.decimals
        analysis["requested_symbol"] = symbol
        analysis["cache_hit"] = False
        analysis["cache_age_seconds"] = 0.0
        analysis["analysis_cache_ttl_seconds"] = (
            asset.analysis_cache_ttl_seconds
        )

        _set_cached_analysis(canonical_symbol, analysis)
        return copy.deepcopy(analysis)

# =============================================================================
# 9. FORMAT PESAN TELEGRAM
# =============================================================================
def generate_signal_message(analysis: dict) -> str:
    """Mengubah hasil generic engine menjadi Markdown Telegram kompatibel."""
    symbol = analysis["symbol"]
    signal = analysis.get("signal", analysis.get("direction", "HOLD"))
    confidence = float(analysis.get("confidence_pct", 0.0))
    trend = analysis.get("trend", "Sideways")
    risk = analysis.get("risk", "Tinggi")
    decimals = _decimal_places(symbol)

    emoji = {
        "BUY": "🟢",
        "SELL": "🔴",
        "HOLD": "🟡",
        "NEUTRAL": "🟡",
    }.get(signal, "⚪")

    lines = [
        f"*📊 Analisis Multi-Timeframe — {symbol}*",
        "",
        f"{emoji} *Signal: {signal}*",
        f"*Confidence: {confidence:.1f}%*",
        f"*Trend: {trend}*",
        f"*Risk: {risk}*",
        "",
        "*Reasons / Alasan Analisis:*",
    ]

    reasons = analysis.get("reasons") or [
        "Belum ada alasan analisis yang tersedia."
    ]
    for reason in reasons[:6]:
        lines.append(f"• {reason}")

    lines.extend(["", "*Indicator Checklist / Checklist Indikator:*"])
    status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    for item in analysis.get("indicator_checklist", [])[:10]:
        status = str(item.get("status", "WARN")).upper()
        icon = status_emoji.get(status, "⚠️")
        name = item.get("name", "Indikator")
        detail = item.get("detail", "")
        lines.append(f"{icon} {name}: {detail}")

    lines.extend(["", "*Detail per Timeframe:*"])
    for timeframe in TIMEFRAMES:
        result = analysis.get("timeframes", {}).get(timeframe)
        label = TIMEFRAME_LABELS.get(timeframe, timeframe)

        if result is None:
            lines.append(f"• *{label}*: data tidak tersedia")
            continue

        bias = _timeframe_bias(result.score)
        event_labels: list[str] = []
        if result.choch:
            event_labels.append(f"CHoCH {result.choch.title()}")
        elif result.bos:
            event_labels.append(f"BOS {result.bos.title()}")
        if result.sideways:
            event_labels.append("Sideways")
        if result.false_breakout:
            event_labels.append(f"False BO {result.false_breakout.title()}")

        event_text = f" | {', '.join(event_labels)}" if event_labels else ""
        lines.append(
            f"• *{label}*: {bias} | {result.score:+.2f} | "
            f"RSI {result.rsi:.1f} | {result.market_structure}{event_text}"
        )

    reference_timeframe = analysis.get(
        "atr_reference_tf",
        ENTRY_TIMEFRAME_FOR_ATR,
    )
    reference_result = analysis.get("timeframes", {}).get(reference_timeframe)

    if reference_result is not None:
        lines.extend(["", "*Struktur & Price Action:*"])
        if reference_result.support_level is not None:
            lines.append(
                f"• Support: {reference_result.support_level:.{decimals}f} "
                f"(strength {reference_result.support_strength})"
            )
        if reference_result.resistance_level is not None:
            lines.append(
                f"• Resistance: {reference_result.resistance_level:.{decimals}f} "
                f"(strength {reference_result.resistance_strength})"
            )
        patterns = list(reference_result.candlestick_patterns)
        lines.append(
            "• Pola candle: "
            + (", ".join(patterns[:3]) if patterns else "tidak ada pola kuat")
        )

    lines.append("")
    if signal in {"BUY", "SELL"}:
        entry = analysis.get("entry") or analysis.get("entry_price")
        stop_loss = analysis.get("sl")
        tp1 = analysis.get("tp1")
        tp2 = analysis.get("tp2")
        tp3 = analysis.get("tp3")
        rr1 = analysis.get("risk_reward_tp1")
        rr2 = analysis.get("risk_reward_tp2")

        lines.extend(
            [
                f"*Entry:* {entry:.{decimals}f}",
                f"*SL:* {stop_loss:.{decimals}f}",
                f"*TP1:* {tp1:.{decimals}f}",
                f"*TP2:* {tp2:.{decimals}f}",
                f"*TP3:* {tp3:.{decimals}f}",
                (
                    f"*Risk Reward:* TP1 1:{rr1:.2f} | TP2 1:{rr2:.2f}"
                    if rr1 is not None and rr2 is not None
                    else "*Risk Reward:* belum tersedia"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "*Entry:* —",
                "*SL:* —",
                "*TP1:* —",
                "*TP2:* —",
                "*Risk Reward:* —",
                "_Tidak ada entry karena konfirmasi belum memenuhi aturan konservatif._",
            ]
        )

    lines.extend(
        [
            "",
            (
                f"_Coverage: {analysis.get('coverage_pct', 0):.1f}% | "
                f"Alignment: {analysis.get('directional_agreement_pct', 0):.1f}% | "
                f"Structure: {analysis.get('structure_confirmation_pct', 0):.1f}%_"
            ),
            "_Confidence adalah skor konfluensi internal, bukan probabilitas kemenangan._",
            "",
            "⚠️ _Bukan saran finansial. Gunakan manajemen risiko sendiri._",
        ]
    )

    message = "\n".join(lines)
    if len(message) <= 4096:
        return message

    # Kompakkan tanpa memotong Markdown di tengah baris.
    compact_lines = [
        line
        for line in lines
        if not line.startswith("• Pola candle: tidak ada")
        and not line.startswith("_Coverage:")
    ]
    message = "\n".join(compact_lines)
    if len(message) <= 4096:
        return message

    safe_lines: list[str] = []
    total = 0
    for line in compact_lines:
        addition = len(line) + (1 if safe_lines else 0)
        if total + addition > 4050:
            break
        safe_lines.append(line)
        total += addition
    safe_lines.append("_Pesan dipadatkan karena batas Telegram._")
    return "\n".join(safe_lines)

# =============================================================================
# 10. CALLBACK HANDLER LEGACY
# =============================================================================
async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handler legacy untuk tombol analyze_xauusd.

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
