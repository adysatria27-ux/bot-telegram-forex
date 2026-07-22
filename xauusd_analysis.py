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
    - RSI 14 (level + divergence + label momentum menguat/melemah)
    - EMA 9/21 (posisi harga, jarak/spread, kemiringan/slope trend)
    - SMA 20
    - Bollinger Bands 20,2
    - ATR 14 (untuk SL/TP dinamis, bukan angka tetap)
    - Momentum harga
    - Support dan Resistance
    - Supply Zone dan Demand Zone
    - Level psikologis harga (per aset, lihat AssetConfig.psychological_increment)
    - Swing High dan Swing Low
    - HH, HL, LH, dan LL
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    - Sideways Detection
    - False Breakout Filter
    - Candlestick Pattern (Engulfing, Pin Bar, Hammer, Shooting Star,
      Morning/Evening Star, Inside Bar, Doji)
    - Multi-timeframe weighting (M5, M15, M30, H1, H4)

Keputusan desain -- Volume (P5):
    Modul ini SENGAJA tidak menggunakan volume sebagai faktor skor, baik di
    total_score (_score_timeframe) maupun confidence_pct
    (_calculate_confidence_score). Alasannya bukan kelalaian, tapi
    keterbatasan data: Twelve Data tidak menyediakan angka volume riil untuk
    pasangan forex maupun XAU/USD spot (pasar OTC/desentralisasi, beda
    dengan saham atau futures yang punya volume terpusat). Volume "tick
    count" dari sebagian broker bukan representasi volume pasar sesungguhnya
    dan bisa menyesatkan kalau dipaksakan jadi faktor confidence.
    Bobot yang tadinya dialokasikan untuk volume dialihkan secara
    proporsional ke Trend dan Market Structure -- dua faktor dengan data
    paling reliable dari OHLC. Jika di masa depan sumber data diganti ke
    yang menyediakan volume riil (misalnya data futures/CME), slot untuk
    volume_score bisa ditambahkan kembali tanpa mengubah arsitektur inti
    (lihat komentar peta bobot di _score_timeframe() dan
    _calculate_confidence_score()).

Catatan penting:
    Confidence Score pada versi ini adalah skor kualitas konfluensi internal.
    Nilai tersebut belum menjadi probabilitas kemenangan sampai dilakukan
    backtest dan kalibrasi statistik.

Kompatibilitas:
    - Fungsi get_market_data() tetap tersedia.
    - Fungsi generate_signal_message() tetap tersedia.
    - Field lama direction, sl, tp, dan atr_value tetap dipertahankan.
    - Cache hasil analisis dan OHLC menghemat request Twelve Data.
    - Fungsi get_market_candles() tersedia untuk outcome tracker.
    - Handler Telegram button() legacy sudah dihapus (dead code, tidak pernah
      dipanggil -- tes_bot.py sudah punya handler sendiri). Modul ini sekarang
      tidak lagi bergantung pada python-telegram-bot sama sekali.
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
    # P6 -- jarak antar level psikologis (angka bulat) yang wajar untuk
    # aset ini. Beda-beda per instrumen karena skala harga & kebiasaan
    # trader beda (gold per $10-50, forex major per 50 pip, dst). Tidak
    # bisa disimpulkan otomatis dari harga saja karena forex yang harganya
    # ~1.10 dan indeks yang harganya ~4000 sama-sama satu digit di depan
    # tapi granularitas pip/poin-nya jauh berbeda.
    psychological_increment: float = 0.0


SUPPORTED_ASSETS: tuple[AssetConfig, ...] = (
    AssetConfig(
        symbol="XAU/USD",
        provider_symbols=("XAU/USD",),
        asset_class="metal",
        decimals=2,
        callback_data="analyze_xauusd",
        psychological_increment=10.0,
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
        psychological_increment=0.5,
        menu_label="XAG/USD",
        aliases=("XAGUSD", "SILVER"),
    ),
    AssetConfig(
        symbol="EUR/USD",
        provider_symbols=("EUR/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_eurusd",
        psychological_increment=0.0050,
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
        psychological_increment=0.0050,
        menu_label="GBP/USD",
        aliases=("GBPUSD",),
    ),
    AssetConfig(
        symbol="USD/JPY",
        provider_symbols=("USD/JPY",),
        asset_class="forex",
        decimals=3,
        callback_data="analyze_usdjpy",
        psychological_increment=0.50,
        menu_label="USD/JPY",
        aliases=("USDJPY",),
    ),
    AssetConfig(
        symbol="AUD/USD",
        provider_symbols=("AUD/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_audusd",
        psychological_increment=0.0050,
        menu_label="AUD/USD",
        aliases=("AUDUSD",),
    ),
    AssetConfig(
        symbol="NZD/USD",
        provider_symbols=("NZD/USD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_nzdusd",
        psychological_increment=0.0050,
        menu_label="NZD/USD",
        aliases=("NZDUSD",),
    ),
    AssetConfig(
        symbol="USD/CAD",
        provider_symbols=("USD/CAD",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_usdcad",
        psychological_increment=0.0050,
        menu_label="USD/CAD",
        aliases=("USDCAD",),
    ),
    AssetConfig(
        symbol="USD/CHF",
        provider_symbols=("USD/CHF",),
        asset_class="forex",
        decimals=5,
        callback_data="analyze_usdchf",
        psychological_increment=0.0050,
        menu_label="USD/CHF",
        aliases=("USDCHF",),
    ),
    AssetConfig(
        symbol="BTC/USD",
        provider_symbols=("BTC/USD",),
        asset_class="crypto",
        decimals=2,
        callback_data="analyze_btcusd",
        psychological_increment=500.0,
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
        psychological_increment=50.0,
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
        psychological_increment=50.0,
        menu_label="NAS100",
        aliases=("NDX", "NASDAQ100", "NASDAQ 100"),
    ),
    AssetConfig(
        symbol="US30",
        provider_symbols=("DJI", "DJIA", "US30"),
        asset_class="index",
        decimals=2,
        callback_data="analyze_us30",
        psychological_increment=100.0,
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

# EMA trend engine (P3): SMA_PERIOD di atas tetap dipertahankan untuk field
# "sma" lama (kompatibilitas), tapi trend_score sekarang dihitung dari EMA,
# bukan lagi cuma jarak close ke SMA. EMA_FAST/EMA_SLOW dipilih pasangan
# umum (9/21) yang cukup responsif untuk timeframe 5m-1h tanpa terlalu
# berisik. EMA_SLOPE_LOOKBACK menentukan seberapa jauh ke belakang
# kemiringan EMA cepat diukur.
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
EMA_SLOPE_LOOKBACK = 5

# RSI divergence & label momentum (P4).
RSI_DIVERGENCE_MIN_RSI_DELTA = 3.0
MOMENTUM_LABEL_LOOKBACK = 5
MOMENTUM_LABEL_MIN_RSI_DELTA = 3.0

ENTRY_TIMEFRAME_FOR_ATR = "15min"

ATR_SL_MULTIPLIER = 1.5

# PENTING: TP tidak lagi memakai kelipatan ATR yang independen dari SL.
# Sebelumnya TP1=1.5x dan TP2=2.5x ATR menghasilkan RR tetap 1:1.0 dan 1:1.67,
# yang secara matematis TIDAK PERNAH bisa mencapai minimum 1:2 berapa pun
# kuatnya confidence. Sekarang TP diturunkan langsung dari jarak SL dikali
# target RR, sehingga RR minimum terjamin by design, bukan kebetulan.
RR_TARGET_TP1 = 2.0
RR_TARGET_TP2 = 3.0
RR_TARGET_TP3 = 4.0

# Risk Reward minimum yang wajib dipenuhi TP1 agar sinyal boleh keluar.
# Ditegakkan juga sebagai hard-gate di _build_market_analysis, bukan cuma
# konsekuensi tidak langsung dari multiplier di atas.
MIN_RISK_REWARD_RATIO = 2.0

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

# Supply/Demand zone (P6): base sempit langsung diikuti candle impulsif.
DEMAND_SUPPLY_LOOKBACK = 40
DEMAND_SUPPLY_BASE_MAX_RANGE_ATR = 0.60
DEMAND_SUPPLY_IMPULSE_MIN_BODY_ATR = 1.20

# REVISI GATING (Stage 3):
# Total bobot TIMEFRAME_WEIGHTS adalah 1.0+1.5+2.0+2.5 = 7.0. Ambang lama
# (0.25) setara 1.75 dari total bobot itu, padahal bobot 1h saja sudah 2.5
# dan bobot 30m saja sudah 2.0 -- keduanya SENDIRIAN sudah melebihi 1.75.
# Akibatnya satu timeframe besar saja yang mendeteksi "gagal breakout" bisa
# memveto total sinyal, walau tiga timeframe lain sepakat kuat ke arah
# sebaliknya. Ini bertentangan dengan prinsip multi-timeframe confluence
# yang jadi inti sistem ini. Ambang dinaikkan ke 0.50 supaya veto total
# baru berlaku kalau MAYORITAS bobot timeframe sepakat menunjukkan gagal
# breakout melawan arah sinyal (bukan cuma satu timeframe sendirian).
# Efeknya terhadap confidence_pct (penalti proporsional 0.30 x rasio ini)
# TIDAK diubah -- sinyal dengan sedikit false breakout tetap mengalami
# pengurangan confidence, cuma tidak lagi otomatis gugur total karenanya.
MAX_FALSE_BREAKOUT_AGAINST_RATIO = 0.50


def _env_flag(name: str, default: bool) -> bool:
    """Membaca environment variable boolean secara toleran."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Membaca environment variable float secara toleran."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Env %s bukan angka valid: %s", name, raw)
        return default


# =============================================================================
# P2 -- Filter overextension / late-entry.
# Hipotesis dari data /stats real: sinyal confidence sangat tinggi cenderung
# muncul saat semua indikator sudah searah sempurna = tren sudah matang =
# entry telat/mengejar harga, dan justru bucket confidence tertinggi yang
# paling banyak kalah (90-100 kumulatif 0/3 di beberapa snapshot). Filter ini
# mengukur seberapa 'meregang' harga (jarak ke EMA dalam ATR + RSI ekstrem
# searah sinyal) dan menguranginya dari confidence secara PROPORSIONAL --
# bukan veto keras, konsisten dengan arsitektur penalti yang sudah ada.
# Sengaja dibuat ringan & reversibel: bobotnya bisa disetel/dimatikan lewat
# environment variable tanpa ubah kode, supaya bisa dikalibrasi dari data.
OVEREXTENSION_PENALTY_ENABLED = _env_flag(
    "OVEREXTENSION_PENALTY_ENABLED", True
)
OVEREXTENSION_PENALTY_WEIGHT = _env_float(
    "OVEREXTENSION_PENALTY_WEIGHT", 0.15
)
# Jarak harga ke EMA lambat (satuan ATR): di bawah START masih sehat (0),
# naik linear sampai penuh (1) di FULL.
OVEREXTENSION_EMA_ATR_START = 1.5
OVEREXTENSION_EMA_ATR_FULL = 3.0
# RSI yang sudah ekstrem searah sinyal.
OVEREXTENSION_RSI_BUY = 70.0
OVEREXTENSION_RSI_SELL = 30.0
_EFFECTIVE_OVEREXTENSION_WEIGHT = (
    OVEREXTENSION_PENALTY_WEIGHT if OVEREXTENSION_PENALTY_ENABLED else 0.0
)


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
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_spread_score: float = 0.0
    ema_slope_score: float = 0.0
    rsi_divergence: Optional[str] = None
    momentum_label: str = "Netral"
    demand_zone: Optional[tuple[float, float]] = None
    supply_zone: Optional[tuple[float, float]] = None
    # Catatan: level psikologis TIDAK disimpan per-timeframe di sini karena
    # levelnya bergantung pada harga saat ini (entry_price), bukan per-TF.
    # Nilainya dihitung sekali di _build_market_analysis() dan disimpan di
    # dict hasil analisis (key "near_psychological_level").


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


def _nearest_psychological_level(
    price: float,
    increment: float,
) -> Optional[float]:
    """
    Mencari level psikologis (angka bulat) terdekat dari harga saat ini.

    increment 0 atau negatif berarti aset itu belum dikonfigurasi levelnya
    (return None, bukan error, supaya aset baru tetap aman kalau lupa diisi).
    """
    if increment <= 0:
        return None
    return round(price / increment) * increment


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


def _calculate_ema(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Menghitung Exponential Moving Average standar."""
    return close.ewm(
        span=period,
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


def _detect_supply_demand_zones(
    df: pd.DataFrame,
    atr_value: float,
) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """
    Mendeteksi zona demand (base sebelum rally) & supply (base sebelum drop).

    Pola yang dicari: satu candle "base" dengan range sempit (konsolidasi
    -- jejak akumulasi/distribusi), langsung diikuti candle "impulsif"
    dengan body besar yang membentuk breakout. Kalau candle impulsif itu
    bullish, base di belakangnya jadi demand zone (area beli yang mendorong
    rally). Kalau impulsif bearish, base-nya jadi supply zone. Zona yang
    dipakai adalah kemunculan PALING BARU dari tiap jenis dalam lookback.

    Ini pelengkap support/resistance dari cluster swing yang sudah ada,
    bukan pengganti -- makanya bobot bonusnya di _score_timeframe lebih
    kecil daripada bonus S/R utama.
    """
    lookback = min(DEMAND_SUPPLY_LOOKBACK, len(df) - 2)
    if lookback < 3:
        return None, None

    segment = df.iloc[-lookback:].reset_index(drop=True)
    demand_zone: Optional[tuple[float, float]] = None
    supply_zone: Optional[tuple[float, float]] = None

    for index in range(len(segment) - 1):
        base = segment.iloc[index]
        impulse = segment.iloc[index + 1]

        base_range = float(base["high"] - base["low"])
        impulse_body = abs(float(impulse["close"] - impulse["open"]))

        is_tight_base = base_range <= atr_value * DEMAND_SUPPLY_BASE_MAX_RANGE_ATR
        is_strong_impulse = (
            impulse_body >= atr_value * DEMAND_SUPPLY_IMPULSE_MIN_BODY_ATR
        )

        if not (is_tight_base and is_strong_impulse):
            continue

        zone = (float(base["low"]), float(base["high"]))
        if float(impulse["close"]) > float(impulse["open"]):
            demand_zone = zone
        else:
            supply_zone = zone

    return demand_zone, supply_zone


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


def _detect_rsi_divergence(
    swings: list[SwingPoint],
    rsi_series: pd.Series,
) -> Optional[str]:
    """
    Mendeteksi bullish/bearish divergence RSI (P4).

    Memakai swing high/low yang sama dengan yang dipakai deteksi struktur,
    supaya konsisten dan tidak menduplikasi logika pencarian swing.

    Bullish divergence: harga bikin Lower Low, tapi RSI di titik itu malah
    lebih tinggi dari RSI di Lower Low sebelumnya -- momentum turun sudah
    melemah walau harga masih membuat titik rendah baru.

    Bearish divergence: harga bikin Higher High, tapi RSI di titik itu malah
    lebih rendah dari RSI di Higher High sebelumnya -- momentum naik sudah
    melemah walau harga masih membuat titik tinggi baru.

    Kalau kedua jenis divergence lolos syarat, dipakai yang paling baru
    (index swing paling besar).
    """
    swing_lows = [item for item in swings if item.kind == "LOW"]
    swing_highs = [item for item in swings if item.kind == "HIGH"]

    candidates: list[tuple[int, str]] = []

    if len(swing_lows) >= 2:
        previous_low, latest_low = swing_lows[-2], swing_lows[-1]
        price_lower_low = latest_low.price < previous_low.price
        rsi_previous_low = float(rsi_series.iloc[previous_low.index])
        rsi_latest_low = float(rsi_series.iloc[latest_low.index])
        rsi_higher_low = (
            rsi_latest_low - rsi_previous_low
        ) >= RSI_DIVERGENCE_MIN_RSI_DELTA

        if price_lower_low and rsi_higher_low:
            candidates.append((latest_low.index, "BULLISH"))

    if len(swing_highs) >= 2:
        previous_high, latest_high = swing_highs[-2], swing_highs[-1]
        price_higher_high = latest_high.price > previous_high.price
        rsi_previous_high = float(rsi_series.iloc[previous_high.index])
        rsi_latest_high = float(rsi_series.iloc[latest_high.index])
        rsi_lower_high = (
            rsi_previous_high - rsi_latest_high
        ) >= RSI_DIVERGENCE_MIN_RSI_DELTA

        if price_higher_high and rsi_lower_high:
            candidates.append((latest_high.index, "BEARISH"))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def _classify_momentum_label(
    rsi_series: pd.Series,
    lookback: int = MOMENTUM_LABEL_LOOKBACK,
) -> str:
    """
    Melabeli momentum sebagai Menguat, Melemah, atau Netral.

    Berbeda dari rsi_score (yang menilai LEVEL RSI sekarang), ini menilai
    ARAH PERUBAHAN RSI beberapa candle terakhir -- RSI 65 yang naik dari 55
    itu "Menguat", sedangkan RSI 65 yang turun dari 75 itu "Melemah", walau
    level RSI-nya sama-sama masih di atas 50.
    """
    if len(rsi_series) <= lookback:
        return "Netral"

    current_rsi = float(rsi_series.iloc[-1])
    prior_rsi = float(rsi_series.iloc[-1 - lookback])
    delta = current_rsi - prior_rsi

    if delta >= MOMENTUM_LABEL_MIN_RSI_DELTA:
        return "Menguat"
    if delta <= -MOMENTUM_LABEL_MIN_RSI_DELTA:
        return "Melemah"
    return "Netral"


def _derive_trend_label(
    sideways: bool,
    structure_score: float,
    trend_score: float,
    momentum_score: float,
    bos: Optional[str] = None,
    choch: Optional[str] = None,
) -> str:
    """
    Menentukan Bullish, Bearish, atau Sideways.

    REVISI (Stage 4): Label "sideways" dihitung dari efisiensi pergerakan
    harga 20 candle TERAKHIR -- ukuran ini lamban, jadi begitu ada breakout
    baru, beberapa candle pertama sesudahnya masih bisa tercatat "sideways"
    walau strukturnya sudah jelas berubah arah (BOS/CHoCH baru saja
    terkonfirmasi). Sebelumnya sideways=True langsung memaksa "Sideways"
    tanpa syarat, sehingga BOS/CHoCH yang lebih baru dan lebih relevan jadi
    diabaikan sepenuhnya. Sekarang BOS/CHoCH yang baru terkonfirmasi dan
    SEARAH dengan combined_direction boleh membatalkan label sideways itu.
    """
    combined_direction = (
        (0.55 * structure_score)
        + (0.30 * trend_score)
        + (0.15 * momentum_score)
    )

    fresh_break_direction: Optional[str] = None
    if choch == "BULLISH" or bos == "BULLISH":
        fresh_break_direction = "Bullish"
    elif choch == "BEARISH" or bos == "BEARISH":
        fresh_break_direction = "Bearish"

    if sideways:
        if fresh_break_direction == "Bullish" and combined_direction > 0:
            return "Bullish"
        if fresh_break_direction == "Bearish" and combined_direction < 0:
            return "Bearish"
        return "Sideways"

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
            EMA_SLOW_PERIOD,
        )
        + MOMENTUM_LOOKBACK
        + EMA_SLOPE_LOOKBACK
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
    ema_fast_series = _calculate_ema(close, EMA_FAST_PERIOD)
    ema_slow_series = _calculate_ema(close, EMA_SLOW_PERIOD)
    bb_upper, bb_middle, bb_lower = _calculate_bollinger(close)
    atr_series = _calculate_atr(df)

    last_close = _safe_float(close.iloc[-1], "close")
    last_rsi = _safe_float(rsi_series.iloc[-1], "RSI")
    last_sma = _safe_float(sma_series.iloc[-1], "SMA")
    last_ema_fast = _safe_float(ema_fast_series.iloc[-1], "EMA cepat")
    last_ema_slow = _safe_float(ema_slow_series.iloc[-1], "EMA lambat")
    prior_ema_fast = _safe_float(
        ema_fast_series.iloc[-1 - EMA_SLOPE_LOOKBACK],
        "EMA cepat sebelumnya",
    )
    last_bb_upper = _safe_float(bb_upper.iloc[-1], "Bollinger upper")
    last_bb_lower = _safe_float(bb_lower.iloc[-1], "Bollinger lower")
    last_bb_middle = _safe_float(bb_middle.iloc[-1], "Bollinger middle")
    last_atr = _safe_float(atr_series.iloc[-1], "ATR")

    if last_atr <= 0:
        logger.warning("ATR tidak valid | tf=%s | atr=%s", timeframe, last_atr)
        return None

    # ------------------------------------------------------------------
    # P3 -- EMA trend engine. Sebelumnya trend_score cuma dari jarak close
    # ke SMA20 (satu ukuran, tidak melihat kemiringan atau crossing sama
    # sekali). Sekarang trend_score gabungan tiga ukuran EMA:
    #   1) posisi harga relatif EMA lambat (bobot dominan, mirip logika lama
    #      tapi pakai EMA yang lebih responsif daripada SMA)
    #   2) jarak antar EMA cepat & lambat (makin lebar makin kuat trennya --
    #      ini "kekuatan trend" dan otomatis mencakup info crossing, karena
    #      spread mendekati nol saat EMA baru saja crossing)
    #   3) kemiringan EMA cepat beberapa candle terakhir (arah & kecepatan
    #      pergerakan EMA itu sendiri, bukan cuma level statis)
    # ------------------------------------------------------------------
    ema_price_position_score = float(
        np.clip((last_close - last_ema_slow) / last_atr / 1.5, -1.0, 1.0)
    )
    ema_spread_score = float(
        np.clip((last_ema_fast - last_ema_slow) / last_atr, -1.0, 1.0)
    )
    ema_slope_score = float(
        np.clip(
            (last_ema_fast - prior_ema_fast)
            / (last_atr * max(EMA_SLOPE_LOOKBACK, 1)),
            -1.0,
            1.0,
        )
    )
    trend_score = float(
        np.clip(
            (0.45 * ema_price_position_score)
            + (0.35 * ema_spread_score)
            + (0.20 * ema_slope_score),
            -1.0,
            1.0,
        )
    )
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

    # ------------------------------------------------------------------
    # P4 -- RSI divergence & label momentum. Divergence dipakai untuk
    # menyesuaikan rsi_score (bukan komponen skor terpisah baru, supaya
    # bobot total_score yang sudah ada tidak perlu dirombak): divergence
    # searah menambah keyakinan, divergence berlawanan arah mengurangi.
    # ------------------------------------------------------------------
    rsi_divergence = _detect_rsi_divergence(swings, rsi_series)
    momentum_label = _classify_momentum_label(rsi_series)

    if rsi_divergence == "BULLISH":
        rsi_score = float(np.clip(rsi_score + 0.30, -1.0, 1.0))
    elif rsi_divergence == "BEARISH":
        rsi_score = float(np.clip(rsi_score - 0.30, -1.0, 1.0))

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

    # P6 -- Supply/Demand zone: bonus lebih kecil daripada S/R swing biasa
    # (0.20 vs 0.30) karena ini pola tambahan/konfirmasi, bukan pengganti
    # support/resistance utama yang sudah dihitung dari cluster swing.
    demand_zone, supply_zone = _detect_supply_demand_zones(df, last_atr)

    if demand_zone is not None and (
        demand_zone[0] - near_distance
        <= last_close
        <= demand_zone[1] + near_distance
    ):
        support_resistance_score += 0.20

    if supply_zone is not None and (
        supply_zone[0] - near_distance
        <= last_close
        <= supply_zone[1] + near_distance
    ):
        support_resistance_score -= 0.20

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
        bos=bos,
        choch=choch,
    )

    # =========================================================================
    # P8 -- Peta bobot total_score (skor arah per timeframe, skala -1..1).
    # Dipetakan ke kategori yang diminta di instruksi proyek supaya mudah
    # ditelusuri, walau angkanya tidak persis sama dengan usulan awal
    # (instruksi proyek mengizinkan pembobotan ditentukan sendiri):
    #   Trend (EMA)          24% -> trend_score
    #   Market Structure     24% -> structure_score (HH/HL/LH/LL + BOS/CHoCH)
    #   Momentum             12% -> momentum_score
    #                        +14% -> rsi_score (bagian dari "Momentum" secara
    #                                konsep, dipisah historically dari RSI 14
    #                                klasik; termasuk penyesuaian divergence)
    #   Support/Resistance    9% -> support_resistance_score (S/R + supply/
    #                                demand zone P6, BOS/CHoCH, false breakout)
    #   Candlestick           7% -> candlestick_score
    #   Bollinger Bands      10% -> bb_score (pelengkap volatilitas, di luar
    #                                7 kategori asli tapi dipertahankan karena
    #                                sudah ada sebelumnya dan berguna)
    #   ATR                   -   dipakai untuk SL/TP & normalisasi skor lain,
    #                                bukan skor arah tersendiri (ATR memang
    #                                bukan indikator arah, jadi wajar tidak
    #                                punya slot bobot sendiri di sini)
    #   Volume                0% -> SENGAJA tidak diberi bobot (lihat P5:
    #                                Twelve Data tidak menyediakan volume
    #                                riil untuk forex/XAU spot). Bobotnya
    #                                dialihkan proporsional ke Trend &
    #                                Structure, dua faktor dengan data paling
    #                                reliable.
    # Total = 100%. Lihat juga _calculate_confidence_score() untuk lapisan
    # bobot KEDUA (confidence, berbeda dari total_score ini).
    # =========================================================================
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
        ema_fast=last_ema_fast,
        ema_slow=last_ema_slow,
        ema_spread_score=ema_spread_score,
        ema_slope_score=ema_slope_score,
        rsi_divergence=rsi_divergence,
        momentum_label=momentum_label,
        demand_zone=demand_zone,
        supply_zone=supply_zone,
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


def _calculate_trend_alignment_score(
    trend: str,
    candidate_signal: str,
) -> float:
    """
    Skor keselarasan kontinu antara trend timeframe besar dan calon sinyal.

    Sebelumnya trend besar dipakai sebagai syarat mutlak (harus persis sama
    dengan calon sinyal, kalau tidak langsung HOLD). Itu membuat satu label
    "Sideways" di H1/H4 bisa membatalkan sinyal walau timeframe lain sangat
    kuat. Sekarang trend besar tetap berpengaruh besar (lihat pembobotan di
    _calculate_confidence_score), tapi secara proporsional:
        - Searah penuh  -> 1.0
        - Netral/Sideways -> 0.5 (tidak mendukung, tapi juga tidak menghukum)
        - Berlawanan    -> 0.0 (biarkan _calculate_counter_trend_penalty yang
          menghukum lebih lanjut kalau berlawanan secara kuat)
    """
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    target_trend = "Bullish" if candidate_signal == "BUY" else "Bearish"

    if trend == target_trend:
        return 1.0
    if trend == "Sideways":
        return 0.5
    return 0.0


def _calculate_counter_trend_penalty(
    trend: str,
    higher_timeframe_score: float,
    candidate_signal: str,
) -> float:
    """
    Penalti proporsional saat calon sinyal melawan trend besar yang kuat.

    Hanya aktif kalau trend besar benar-benar berlawanan arah (bukan sekadar
    Sideways) dan kekuatannya (higher_timeframe_score) signifikan. Nilainya
    0.0-1.0, dipakai mengurangi confidence secara proporsional, bukan
    memveto total seperti logika lama.
    """
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    opposite_trend = "Bearish" if candidate_signal == "BUY" else "Bullish"
    if trend != opposite_trend:
        return 0.0

    return float(np.clip(abs(higher_timeframe_score), 0.0, 1.0))


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


def _timeframe_overextension(
    result: "TimeframeResult",
    candidate_signal: str,
) -> float:
    """
    Skor 0..1 seberapa 'telat/meregang' harga untuk arah calon sinyal (P2).

    Dua proksi 'kejauhan', dipakai nilai TERBESAR (salah satu cukup):
      1) Jarak harga ke EMA lambat dalam satuan ATR, HANYA yang searah
         sinyal (harga jauh DI ATAS EMA untuk BUY, jauh DI BAWAH untuk
         SELL). Di bawah START ATR dianggap sehat (0), naik linear sampai
         penuh (1) di FULL ATR.
      2) RSI yang sudah ekstrem searah sinyal (>70 BUY, <30 SELL).

    Tidak pernah memveto sinyal; hanya jadi penalti proporsional di
    confidence (lihat _calculate_confidence_score).
    """
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    atr = result.atr
    if atr is None or atr <= 0:
        return 0.0

    ema_component = 0.0
    if result.ema_slow is not None:
        distance_atr = (result.close - result.ema_slow) / atr
        directional = (
            distance_atr if candidate_signal == "BUY" else -distance_atr
        )
        if directional > OVEREXTENSION_EMA_ATR_START:
            span = max(
                OVEREXTENSION_EMA_ATR_FULL - OVEREXTENSION_EMA_ATR_START,
                np.finfo(float).eps,
            )
            ema_component = (
                directional - OVEREXTENSION_EMA_ATR_START
            ) / span

    rsi_component = 0.0
    rsi_value = result.rsi
    if candidate_signal == "BUY" and rsi_value > OVEREXTENSION_RSI_BUY:
        rsi_component = (rsi_value - OVEREXTENSION_RSI_BUY) / max(
            100.0 - OVEREXTENSION_RSI_BUY, np.finfo(float).eps
        )
    elif candidate_signal == "SELL" and rsi_value < OVEREXTENSION_RSI_SELL:
        rsi_component = (OVEREXTENSION_RSI_SELL - rsi_value) / max(
            OVEREXTENSION_RSI_SELL, np.finfo(float).eps
        )

    return float(np.clip(max(ema_component, rsi_component), 0.0, 1.0))


def _calculate_overextension_ratio(
    timeframe_results: dict[str, "TimeframeResult"],
    candidate_signal: str,
) -> float:
    """Rata-rata tertimbang overextension seluruh timeframe (P2)."""
    if candidate_signal not in {"BUY", "SELL"}:
        return 0.0

    total_weight = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        for timeframe in timeframe_results
    )
    if total_weight <= 0:
        return 0.0

    weighted = sum(
        TIMEFRAME_WEIGHTS[timeframe]
        * _timeframe_overextension(result, candidate_signal)
        for timeframe, result in timeframe_results.items()
    )
    return float(np.clip(weighted / total_weight, 0.0, 1.0))


def _calculate_confidence_score(
    combined_score: float,
    directional_agreement: float,
    coverage_ratio: float,
    structure_confirmation: float = 0.0,
    pattern_confirmation: float = 0.0,
    trend_alignment_score: float = 0.0,
    sideways_ratio: float = 0.0,
    false_breakout_against_ratio: float = 0.0,
    counter_trend_penalty: float = 0.0,
    overextension_ratio: float = 0.0,
) -> float:
    """
    Menghitung Confidence Score konfluensi internal.

    Struktur market dan candlestick dapat meningkatkan confidence jika searah.
    Sideways, false breakout yang melawan arah, dan trend besar yang
    berlawanan kuat akan mengurangi confidence secara proporsional.

    Catatan desain (revisi):
    Trend timeframe besar sebelumnya adalah syarat mutlak terpisah (harus
    identik dengan calon sinyal, kalau tidak otomatis HOLD). Sekarang trend
    besar masuk sebagai komponen berbobot (trend_alignment_score) plus
    penalti proporsional saat benar-benar berlawanan (counter_trend_penalty).
    Sinyal yang sangat kuat di timeframe kecil-menengah tidak lagi otomatis
    gugur hanya karena H1/H4 sedang diklasifikasikan Sideways.

    =========================================================================
    P8 -- Peta bobot confidence_pct (lapisan bobot KEDUA, beda dari
    total_score di _score_timeframe()). total_score menilai ARAH per
    timeframe; confidence_pct menilai KUALITAS/KELAYAKAN sinyal gabungan
    secara keseluruhan -- karena itu wajar kalau bobot & komponennya beda.

    Komponen positif (menjumlah ke 100%):
      normalized_strength    28% -> seberapa kuat combined_score (gabungan
                                     seluruh timeframe) dibanding target 0.60.
                                     Proxy gabungan Trend + Momentum + SR +
                                     Candlestick, karena combined_score itu
                                     sendiri sudah hasil agregasi total_score
                                     per timeframe.
      directional_agreement  18% -> berapa persen timeframe yang searah
                                     (konfirmasi lintas-timeframe, bagian
                                     dari kategori Market Structure).
      coverage_ratio         12% -> berapa lengkap data timeframe yang
                                     berhasil diambil (kualitas data, bukan
                                     sinyal arah, tapi tetap penting supaya
                                     confidence tidak overclaim saat data
                                     timeframe banyak yang hilang).
      structure_confirmation 15% -> HH/HL/LH/LL, BOS, CHoCH (Market
                                     Structure).
      pattern_confirmation    7% -> candlestick pattern (Candlestick).
      trend_alignment_score  20% -> keselarasan trend H1/H4 (Trend, EMA).
      -------------------------------------------------------------
      Total                 100%

    Penalti proporsional (mengalikan confidence, bukan menjumlah/mengurangi
    dari 100 -- supaya tidak bisa menjadi negatif dan tetap proporsional):
      sideways_ratio               20% dari penalty_multiplier -> menghukum
                                    saat mayoritas timeframe sideways
                                    (Market Structure lemah).
      false_breakout_against_ratio 30% dari penalty_multiplier -> penalti
                                    terbesar karena false breakout yang
                                    melawan arah adalah sinyal bahaya paling
                                    konkret (S/R & Market Structure palsu).
      counter_trend_penalty        25% dari penalty_multiplier -> menghukum
                                    saat melawan trend besar H1/H4 (Trend).
      penalty_multiplier dibatasi minimum 0.30 (bukan 0) supaya sinyal yang
      tetap valid secara teknikal tidak langsung hangus ke confidence 0
      hanya karena satu faktor penalti besar; ini konsisten dengan filosofi
      "berani memberi sinyal kalau memang layak" (poin #10 instruksi).

    Volume: sama seperti total_score, confidence_pct juga sengaja TIDAK
    memberi bobot untuk volume (lihat P5 & catatan di _score_timeframe()).
    =========================================================================
    """
    normalized_strength = min(abs(combined_score) / 0.60, 1.0)

    confidence = 100.0 * (
        (0.28 * normalized_strength)
        + (0.18 * directional_agreement)
        + (0.12 * coverage_ratio)
        + (0.15 * structure_confirmation)
        + (0.07 * pattern_confirmation)
        + (0.20 * float(np.clip(trend_alignment_score, 0.0, 1.0)))
    )

    penalty_multiplier = (
        1.0
        - (0.20 * float(np.clip(sideways_ratio, 0.0, 1.0)))
        - (
            0.30
            * float(
                np.clip(false_breakout_against_ratio, 0.0, 1.0)
            )
        )
        - (0.25 * float(np.clip(counter_trend_penalty, 0.0, 1.0)))
        # P2 -- penalti overextension/late-entry (ringan & reversibel;
        # bobot & on/off lewat env). Tetap dibatasi floor 0.30 di bawah.
        - (
            _EFFECTIVE_OVEREXTENSION_WEIGHT
            * float(np.clip(overextension_ratio, 0.0, 1.0))
        )
    )

    confidence *= max(0.30, penalty_multiplier)

    return round(float(np.clip(confidence, 0.0, 100.0)), 1)


def _classify_signal_risk(
    signal: str,
    confidence_pct: float,
    directional_agreement: float,
    coverage_ratio: float,
    sideways_ratio: float = 0.0,
    false_breakout_against_ratio: float = 0.0,
    counter_trend_penalty: float = 0.0,
) -> str:
    """Mengklasifikasikan risiko kualitas sinyal."""
    if signal == "HOLD":
        return "Tinggi"

    if (
        false_breakout_against_ratio > 0.0
        or sideways_ratio > 0.50
        or counter_trend_penalty >= 0.35
    ):
        return "Tinggi"

    if (
        confidence_pct >= 80.0
        and directional_agreement >= 0.80
        and coverage_ratio >= 0.90
        and sideways_ratio <= 0.25
        and counter_trend_penalty == 0.0
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
    risk_reward_tp1: Optional[float] = None,
    counter_trend_penalty: float = 0.0,
    overextension_ratio: float = 0.0,
    risk_reward_ok: bool = True,
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
        if sideways_ratio > 0.50:
            reasons.append(
                "Sebagian besar timeframe masih terdeteksi sideways."
            )
        if false_breakout_against_ratio > 0.0:
            reasons.append(
                "False breakout terdeteksi melawan calon arah sinyal."
            )
        if structure_confirmation < MIN_STRUCTURE_CONFIRMATION:
            reasons.append(
                "HH/HL, LH/LL, BOS, atau CHoCH belum memberi konfirmasi kuat."
            )
        if not risk_reward_ok:
            rr_text = (
                f"1:{risk_reward_tp1:.2f}"
                if risk_reward_tp1 is not None
                else "tidak dapat dihitung"
            )
            reasons.append(
                "Risk Reward TP1 belum memenuhi minimum "
                f"1:{MIN_RISK_REWARD_RATIO:.1f} (saat ini {rr_text})."
            )
        if counter_trend_penalty >= 0.35:
            reasons.append(
                "Calon sinyal melawan trend kuat di timeframe besar (H1/H4)."
            )
        if overextension_ratio >= 0.50:
            reasons.append(
                "Harga sudah terlalu jauh dari EMA / RSI ekstrem "
                "(indikasi entry telat) -- menunggu harga lebih sehat."
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

            if result.rsi_divergence == target_direction:
                reasons.append(
                    f"{target_direction.title()} divergence RSI terdeteksi "
                    f"pada {label}, momentum {result.momentum_label.lower()}."
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
        if risk_reward_tp1 is not None:
            reasons.append(
                f"Risk Reward TP1 mencapai 1:{risk_reward_tp1:.2f} "
                f"(minimum 1:{MIN_RISK_REWARD_RATIO:.1f})."
            )
        if counter_trend_penalty > 0.0:
            reasons.append(
                "Catatan: sinyal ini melawan sebagian trend timeframe besar, "
                "gunakan manajemen risiko lebih ketat."
            )
        if overextension_ratio >= 0.40:
            reasons.append(
                "Catatan: harga sudah cukup jauh dari EMA (kemungkinan entry "
                "telat) -- pertimbangkan tunggu retrace atau perketat risiko."
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
    risk_reward_tp1: Optional[float] = None,
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

    if risk_reward_tp1 is not None:
        if risk_reward_tp1 >= MIN_RISK_REWARD_RATIO:
            add("Risk Reward (TP1)", "PASS", f"1:{risk_reward_tp1:.2f}")
        else:
            add(
                "Risk Reward (TP1)",
                "FAIL",
                f"1:{risk_reward_tp1:.2f} (min 1:{MIN_RISK_REWARD_RATIO:.1f})",
            )
    else:
        add("Risk Reward (TP1)", "WARN", "Belum dapat dihitung")

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
    psychological_increment: float = 0.0,
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
    trend_alignment_score = _calculate_trend_alignment_score(
        trend,
        candidate_signal,
    )
    counter_trend_penalty = _calculate_counter_trend_penalty(
        trend,
        higher_timeframe_score,
        candidate_signal,
    )
    overextension_ratio = _calculate_overextension_ratio(
        timeframe_results,
        candidate_signal,
    )

    confidence_pct = _calculate_confidence_score(
        combined_score,
        directional_agreement,
        coverage_ratio,
        structure_confirmation,
        pattern_confirmation,
        trend_alignment_score,
        sideways_ratio,
        false_breakout_against_ratio,
        counter_trend_penalty,
        overextension_ratio,
    )

    # ------------------------------------------------------------------
    # Entry/SL/TP dihitung dari SEKARANG (berdasarkan candidate_signal),
    # bukan setelah signal final ditentukan. Ini perlu agar Risk Reward
    # bisa ikut menjadi syarat kelayakan sinyal (poin 8), bukan sekadar
    # angka yang dihitung belakangan setelah keputusan sudah diambil.
    # ------------------------------------------------------------------
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

    # P6 -- level psikologis. Dihitung sekali di sini (bukan per timeframe
    # seperti supply/demand zone) karena butuh konfigurasi per-aset
    # (psychological_increment) yang baru tersedia di level fungsi ini,
    # bukan di _score_timeframe yang sengaja dibuat asset-agnostic. Ini
    # informasi pendukung untuk transparansi alasan (item 12), bukan
    # komponen skor baru -- supaya tidak menambah risiko di pipeline
    # scoring inti yang sudah divalidasi.
    nearest_psychological_level = _nearest_psychological_level(
        entry_price,
        psychological_increment,
    )
    near_psychological_level: Optional[float] = None
    if nearest_psychological_level is not None:
        psychological_distance = abs(entry_price - nearest_psychological_level)
        if psychological_distance <= atr_value * NEAR_LEVEL_ATR_MULTIPLIER:
            near_psychological_level = nearest_psychological_level

    prospective_sl: Optional[float] = None
    prospective_tp1: Optional[float] = None
    prospective_tp2: Optional[float] = None
    prospective_tp3: Optional[float] = None

    if candidate_signal == "BUY":
        prospective_sl = entry_price - (ATR_SL_MULTIPLIER * atr_value)
        risk_distance = entry_price - prospective_sl
        prospective_tp1 = entry_price + (risk_distance * RR_TARGET_TP1)
        prospective_tp2 = entry_price + (risk_distance * RR_TARGET_TP2)
        prospective_tp3 = entry_price + (risk_distance * RR_TARGET_TP3)
    elif candidate_signal == "SELL":
        prospective_sl = entry_price + (ATR_SL_MULTIPLIER * atr_value)
        risk_distance = prospective_sl - entry_price
        prospective_tp1 = entry_price - (risk_distance * RR_TARGET_TP1)
        prospective_tp2 = entry_price - (risk_distance * RR_TARGET_TP2)
        prospective_tp3 = entry_price - (risk_distance * RR_TARGET_TP3)

    risk_reward_tp1 = _calculate_risk_reward(
        entry_price if candidate_signal in {"BUY", "SELL"} else None,
        prospective_sl,
        prospective_tp1,
    )
    risk_reward_tp2 = _calculate_risk_reward(
        entry_price if candidate_signal in {"BUY", "SELL"} else None,
        prospective_sl,
        prospective_tp2,
    )
    risk_reward_tp3 = _calculate_risk_reward(
        entry_price if candidate_signal in {"BUY", "SELL"} else None,
        prospective_sl,
        prospective_tp3,
    )

    enough_risk_reward = (
        candidate_signal in {"BUY", "SELL"}
        and risk_reward_tp1 is not None
        and risk_reward_tp1 >= MIN_RISK_REWARD_RATIO
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

    # ------------------------------------------------------------------
    # REVISI GATING (Stage 2):
    # Dua syarat lama di sini SEBELUMNYA bersifat veto mutlak:
    #   - trend_aligned: trend H1/H4 harus PERSIS sama dengan calon sinyal.
    #     Satu label "Sideways" di H1 saja sudah membatalkan segalanya.
    #   - higher_timeframe_false_breakout: satu wick gagal breakout di
    #     H1/H4 langsung memveto, walau timeframe lain sangat kuat.
    # Keduanya sekarang sudah masuk sebagai komponen BERBOBOT dan penalti
    # PROPORSIONAL di dalam confidence_pct (trend_alignment_score,
    # counter_trend_penalty, dan false_breakout_against_ratio yang memang
    # sudah memberi bobot lebih besar ke H1/H4 lewat TIMEFRAME_WEIGHTS).
    # Hard-gate yang tersisa hanya untuk hal yang benar-benar tidak layak
    # ditawar: kualitas data, konfluensi minimum, dan Risk Reward.
    # ------------------------------------------------------------------
    if (
        candidate_signal in {"BUY", "SELL"}
        and enough_timeframes
        and enough_coverage
        and enough_agreement
        and enough_confidence
        and enough_structure
        and not_too_sideways
        and false_breakout_filter_passed
        and enough_risk_reward
    ):
        signal = candidate_signal
    else:
        signal = "HOLD"

    direction = "NEUTRAL" if signal == "HOLD" else signal

    stop_loss = prospective_sl if signal in {"BUY", "SELL"} else None
    take_profit_1 = prospective_tp1 if signal in {"BUY", "SELL"} else None
    take_profit_2 = prospective_tp2 if signal in {"BUY", "SELL"} else None
    take_profit_3 = prospective_tp3 if signal in {"BUY", "SELL"} else None

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
        false_breakout_filter_passed=false_breakout_filter_passed,
        timeframe_results=timeframe_results,
        reference_result=reference_result,
        risk_reward_tp1=risk_reward_tp1,
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
        counter_trend_penalty,
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
        risk_reward_tp1=risk_reward_tp1,
        counter_trend_penalty=counter_trend_penalty,
        overextension_ratio=overextension_ratio,
        risk_reward_ok=enough_risk_reward,
    )

    logger.info(
        "Analisis selesai | symbol=%s | signal=%s | score=%.3f | "
        "confidence=%.1f | structure=%.2f | pattern=%.2f | "
        "sideways=%.2f | false_breakout=%.2f | overext=%.2f | trend=%s",
        symbol,
        signal,
        combined_score,
        confidence_pct,
        structure_confirmation,
        pattern_confirmation,
        sideways_ratio,
        false_breakout_against_ratio,
        overextension_ratio,
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
        "overextension_ratio_pct": round(overextension_ratio * 100, 1),
        "false_breakout_filter_passed": false_breakout_filter_passed,
        "trend_alignment_score": round(trend_alignment_score, 2),
        "counter_trend_penalty": round(counter_trend_penalty, 2),
        "min_risk_reward_ratio": MIN_RISK_REWARD_RATIO,
        "risk_reward_ok": enough_risk_reward,
        "near_psychological_level": near_psychological_level,
        "demand_zone": reference_result.demand_zone,
        "supply_zone": reference_result.supply_zone,
        "confidence_note": (
            "Confidence Score adalah skor konfluensi internal dan "
            "belum menjadi probabilitas kemenangan."
        ),
    }



async def get_market_candles(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = "5min",
) -> list[dict[str, Any]]:
    """
    Mengambil candle kronologis melalui provider resolver dan cache yang sama.

    Fungsi ini disediakan untuk signal outcome tracker agar tidak membuat
    implementasi pengambilan data yang terpisah atau menambah request ketika
    OHLC masih tersedia di cache.

    Return:
        [
            {
                "datetime": "2026-07-18T10:00:00",
                "open": float,
                "high": float,
                "low": float,
                "close": float,
            },
            ...
        ]
    """
    canonical_symbol = _normalize_symbol(symbol)

    if interval not in TIMEFRAMES:
        supported = ", ".join(TIMEFRAMES)
        raise ValueError(
            f"Interval '{interval}' tidak didukung. Pilihan: {supported}."
        )

    asset = _ASSET_BY_SYMBOL[canonical_symbol]
    provider_symbol = await _resolve_provider_symbol(asset)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        dataframe = await _fetch_ohlc(
            session,
            provider_symbol,
            interval,
        )

    if dataframe is None or dataframe.empty:
        raise RuntimeError(
            f"Data candle {canonical_symbol} timeframe {interval} "
            "tidak tersedia."
        )

    candles: list[dict[str, Any]] = []
    for row in dataframe.itertuples(index=False):
        candle_time = getattr(row, "datetime")
        if hasattr(candle_time, "isoformat"):
            formatted_time = candle_time.isoformat()
        else:
            formatted_time = str(candle_time)

        candles.append(
            {
                "datetime": formatted_time,
                "open": float(getattr(row, "open")),
                "high": float(getattr(row, "high")),
                "low": float(getattr(row, "low")),
                "close": float(getattr(row, "close")),
            }
        )

    return candles

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
        analysis = await _build_market_analysis(
            provider_symbol,
            psychological_increment=asset.psychological_increment,
        )

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
        if result.rsi_divergence:
            event_labels.append(f"Divergence {result.rsi_divergence.title()}")

        event_text = f" | {', '.join(event_labels)}" if event_labels else ""
        lines.append(
            f"• *{label}*: {bias} | {result.score:+.2f} | "
            f"RSI {result.rsi:.1f} ({result.momentum_label})"
            f" | {result.market_structure}{event_text}"
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

        demand_zone = analysis.get("demand_zone")
        if demand_zone:
            lines.append(
                f"• Demand zone: {demand_zone[0]:.{decimals}f} - "
                f"{demand_zone[1]:.{decimals}f}"
            )
        supply_zone = analysis.get("supply_zone")
        if supply_zone:
            lines.append(
                f"• Supply zone: {supply_zone[0]:.{decimals}f} - "
                f"{supply_zone[1]:.{decimals}f}"
            )
        near_psychological_level = analysis.get("near_psychological_level")
        if near_psychological_level is not None:
            lines.append(
                "• Harga dekat area psikologis: "
                f"{near_psychological_level:.{decimals}f}"
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
