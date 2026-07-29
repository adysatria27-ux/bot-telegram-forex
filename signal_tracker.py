"""
================================================================================
Signal Logger dan Outcome Tracker
================================================================================

Tujuan:
    - Menyimpan seluruh hasil analisis BUY, SELL, dan HOLD.
    - Mencegah pencatatan ganda dari hasil cache/candle yang sama.
    - Melacak TP1, TP2, TP3, SL, dan sinyal kedaluwarsa.
    - Menyediakan statistik untuk kalibrasi confidence.

Penyimpanan:
    SQLite dari Python standard library, tanpa dependency tambahan.

Persistensi Railway:
    Agar database tidak hilang saat redeploy, pasang Railway Volume pada /data.
    Lokasi dapat diubah melalui environment variable SIGNAL_TRACKER_DB_PATH.

Aturan konservatif:
    Jika SL dan TP tersentuh pada candle yang sama, SL dianggap tersentuh dahulu.

Manajemen trade -- Breakeven setelah TP1:
    Setelah TP1 tersentuh, SL efektif dipindah ke harga entry. Trade yang
    sudah bergerak ke arah kita tidak lagi bisa berbalik menjadi kerugian
    penuh di SL awal; paling buruk ditutup impas (0R) dan dicatat sebagai
    BREAKEVEN_AFTER_TPn. Bisa dimatikan lewat env BREAKEVEN_AFTER_TP1=0.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


logger = logging.getLogger(__name__)

DATABASE_ENV_NAME = "SIGNAL_TRACKER_DB_PATH"
DEFAULT_VOLUME_DATABASE = Path("/data/signal_tracker.db")
FALLBACK_DATABASE = Path("signal_tracker.db").resolve()

DEFAULT_SIGNAL_EXPIRY_HOURS = 12
CRYPTO_SIGNAL_EXPIRY_HOURS = 24

# Masa berlaku sinyal dihitung dalam JUMLAH CANDLE, bukan jam tetap.
#
# Sebelumnya 12 jam dipakai untuk semua timeframe. Angka itu dirancang saat
# eksekusi di M15 (12 jam = 48 candle, wajar). Tapi konstanta yang sama ikut
# terbawa ke timeframe lain tanpa penyesuaian, dan di situ jadi merusak:
# di eksekusi H4, 12 jam hanya memberi TIGA candle kepada tiap trade.
# Backtest 185 sinyal EUR/USD menunjukkan 74% trade mati kedaluwarsa
# (102 EXPIRED + 28 EXPIRED_TP1 + 7 EXPIRED_TP2) dan TP3 tercapai cuma
# 1 kali -- bukan karena targetnya keliru, tapi karena tidak pernah ada
# waktu untuk mencapainya.
#
# Dengan basis candle, masa berlaku ikut menyesuaikan skala waktu secara
# otomatis. Nilai default sengaja dipilih agar PERSIS mereproduksi
# perilaku lama pada M15: 48 x 15 menit = 12 jam, dan crypto
# 96 x 15 menit = 24 jam. Jadi produksi tidak berubah sedikit pun sampai
# timeframe eksekusinya memang diubah.
SIGNAL_EXPIRY_CANDLES = 48
CRYPTO_SIGNAL_EXPIRY_CANDLES = 96
_REFERENCE_TIMEFRAME_FALLBACK = "15min"
MAX_RECENT_SIGNAL_LIMIT = 50

_SCHEMA_VERSION = 1
_DATABASE_PATH: Optional[Path] = None


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
        logger.warning(
            "Nilai environment variable %s='%s' tidak valid, pakai default %s.",
            name,
            raw,
            default,
        )
        return default


# Manajemen trade -- Breakeven setelah TP1.
# Setelah TP1 tersentuh, SL efektif dipindah ke harga entry sehingga trade
# yang sudah bergerak ke arah kita tidak lagi bisa berbalik menjadi kerugian
# penuh di SL awal; paling buruk ditutup impas (0R). Perilaku ini bisa
# dimatikan lewat environment variable BREAKEVEN_AFTER_TP1=0 bila ingin
# kembali ke perilaku lama (SL awal tetap sampai trade selesai).
BREAKEVEN_AFTER_TP1 = _env_flag("BREAKEVEN_AFTER_TP1", True)

# Lever B (rancangan geometri SL/TP) -- Breakeven SEBELUM TP1, begitu profit
# mengambang (MFE) mencapai kelipatan risk ini. Target utamanya kasus XAU
# "nyaris menang lalu dikembalikan penuh" (/diag 2026-07-24: beberapa XAU
# yang kena SL langsung sempat MFE/R >1.0, satu bahkan 0.93 ke TP1, sebelum
# berbalik penuh -- breakeven-setelah-TP1 tidak menolong karena TP1 di 2R
# belum sempat tersentuh). Default 1.0R dipilih dari simulasi memakai 12
# SL-langsung historis (Bapak): ambang 1.0R menyelamatkan 3 trade (+3R),
# ambang lebih rendah (0.5R) menyelamatkan lebih banyak (+6R) tapi lebih
# berisiko memotong calon pemenang yang masih dalam pullback wajar.
# === DIMATIKAN 2026-07-28 (default 1.0 -> 0.0). BUKTI, BUKAN DUGAAN. ===
# Kondisi pemicu yang ditulis di komentar versi sebelumnya ("kalau TP2/TP3
# hit-rate turun drastis dibanding sebelum ini aktif, naikkan ambang atau
# matikan") TERPENUHI secara ekstrem pada snapshot /stats 2026-07-28:
#
#   24 Jul (27 trade selesai): 9 TP3, 12 SL. Ekspektasi +0.78R per trade.
#   28 Jul (123 trade selesai): 9 TP3, 68 SL, 35 BE-dini. -0.28R per trade.
#   Artinya: dari 96 trade BARU sejak lever ini aktif, NOL yang mencapai
#   TP3, sementara 35 trade ditutup impas di 0R.
#
# Penyebabnya matematis, bukan soal kalibrasi angka:
#   risk 1R = 1.5 x ATR(M15); trigger BE = 1.0R = 1.5 ATR; TP1 = 2R = 3 ATR.
#   Supaya menang, harga harus maju 1.5 ATR, LALU TIDAK BOLEH pullback
#   1.5 ATR, lalu maju 1.5 ATR lagi. Pullback sebesar itu setelah maju
#   sebesar itu adalah perilaku M15 yang paling normal. Jadi lever ini
#   memotong seluruh ekor kanan distribusi (pemenang -> 0R) sementara ekor
#   kiri (-1R) dibiarkan utuh. Untuk sistem yang edge-nya ada di TP3 (+4R),
#   itu dijamin negatif berapa pun bagusnya kualitas entry.
#
# KESIMPULAN DESAIN: breakeven berbasis R-multiple tidak kompatibel dengan
# TP1 di 2R. Proteksi pre-TP1 yang benar harus berbasis STRUKTUR (trailing
# ke swing low/high terakhir), bukan jarak R tetap -- itu rencana P2, bukan
# tambalan sekarang.
#
# Breakeven SETELAH TP1 (BREAKEVEN_AFTER_TP1 di atas) TETAP AKTIF. Lever itu
# terbukti benar dan tidak pernah memotong pemenang, karena baru berlaku
# setelah target pertama tercapai.
#
# Reversibel: set env BREAKEVEN_AT_MFE_R=1.0 untuk mengembalikan perilaku
# lama tanpa ubah kode. Nilai <= 0 mematikan lever ini sepenuhnya.
BREAKEVEN_AT_MFE_R = _env_float("BREAKEVEN_AT_MFE_R", 0.0)


# === ANTI-DUPLIKAT POSISI (baru 2026-07-28) ==================================
# Masalah yang diserang: sejak Live Scanner aktif, `_scan_once()` memanggil
# `_track_analysis()` SETIAP siklus (default tiap 15 menit) SEBELUM cek
# cooldown alert. Cooldown 60 menit hanya menahan NOTIFIKASI Telegram, tidak
# menahan PENCATATAN trade. Sementara dedup fingerprint memakai
# reference_candle_time (waktu candle M15) yang selalu berubah tiap siklus,
# sehingga fingerprint tidak pernah bentrok.
#
# Akibatnya satu tren yang berjalan 5 jam melahirkan ~20 posisi BUY pada ide
# yang sama persis, di harga yang makin lama makin buruk. Saat tren berbalik,
# semuanya kena SL bersamaan. Ini merusak dua hal sekaligus:
#   1. STATISTIK -- 123 "trade" bukan 123 sampel independen, mungkin cuma
#      15-25 setup nyata yang dihitung berulang. Semua kesimpulan kalibrasi
#      dari angka itu terlalu percaya diri.
#   2. RISIKO NYATA -- kalau tiap alert dieksekusi, itu overexposure ~20x
#      pada satu ide. Satu pembalikan menghabiskan akun.
#
# Aturan baru: selama masih ada posisi OPEN untuk symbol + arah yang sama,
# sinyal berikutnya dengan arah itu TIDAK dicatat sebagai trade baru. Ini
# meniru perilaku trader nyata -- satu ide, satu posisi, sampai selesai.
#
# Yang SENGAJA tidak diblokir: arah berlawanan (BUY saat SELL masih OPEN)
# tetap dicatat, karena itu pembalikan bias yang sah, bukan duplikasi. HOLD
# juga tetap dicatat penuh supaya data kalibrasi tidak hilang.
#
# Reversibel: env ALLOW_DUPLICATE_OPEN=1 mengembalikan perilaku lama.
ALLOW_DUPLICATE_OPEN = _env_flag("ALLOW_DUPLICATE_OPEN", False)


# === TITIK NOL STATISTIK (baru 2026-07-28) ==================================
# Masalah: data historis (per 28 Jul: 514 analisis, 136 sinyal, 123 selesai)
# dihasilkan oleh rezim yang sudah terbukti rusak -- breakeven-dini memotong
# semua pemenang jadi 0R, dan scanner mencatat sinyal duplikat tiap siklus
# sehingga satu setup dihitung belasan kali. Angka apa pun yang dihitung dari
# campuran data lama + baru akan menyesatkan untuk waktu yang lama, karena
# data lama jumlahnya jauh lebih banyak dan akan mendominasi rata-rata.
#
# SENGAJA TIDAK MENGHAPUS DATA. Penghapusan tidak bisa dibatalkan, dan kalau
# hipotesis soal breakeven-dini ternyata keliru, histori itu satu-satunya
# pembanding yang kita punya. Jadi pendekatannya: beri tanda titik nol.
# Baris lama tetap ada di database, cuma tidak dihitung oleh laporan.
#
# Cara pakai: set env STATS_SINCE ke waktu redeploy, mis.
#   STATS_SINCE=2026-07-29
#   STATS_SINCE=2026-07-29T10:30:00
# Kosongkan (atau hapus env-nya) untuk melihat seluruh histori lagi --
# reset ini sepenuhnya reversibel, karena tidak ada yang dibuang.
#
# Efek keduanya, yang sama pentingnya: sinyal OPEN yang dibuka SEBELUM titik
# nol akan diarsipkan saat startup (lihat archive_signals_before_epoch).
# Tanpa itu, 13 posisi OPEN warisan rezim lama akan MEMBLOKIR sinyal baru
# lewat aturan anti-duplikat, dan hasil akhirnya tetap mencemari statistik.
STATS_SINCE_RAW = os.getenv("STATS_SINCE", "").strip()


def _stats_since() -> Optional[datetime]:
    """Membaca titik nol statistik dari env. None = hitung semua histori."""
    if not STATS_SINCE_RAW:
        return None

    parsed = _parse_datetime(STATS_SINCE_RAW)
    if parsed is None:
        logger.warning(
            "STATS_SINCE bukan tanggal yang valid, diabaikan: %s",
            STATS_SINCE_RAW,
        )
        return None

    return parsed


def _report_start_time(days: int) -> str:
    """
    Menentukan batas awal laporan.

    Diambil yang PALING BARU antara jendela hari yang diminta dan titik nol
    STATS_SINCE. Dengan begitu `/stats 30 hari` tidak bisa diam-diam menarik
    kembali data dari rezim lama yang sudah direset.
    """
    safe_days = max(1, min(int(days), 3650))
    window_start = _utc_now() - timedelta(days=safe_days)

    epoch = _stats_since()
    if epoch is not None and epoch > window_start:
        return _to_iso(epoch)

    return _to_iso(window_start)


def archive_signals_before_epoch() -> int:
    """
    Menutup sinyal yang masih OPEN dari sebelum titik nol.

    Dijalankan sekali saat startup. Sinyal warisan rezim lama ditutup dengan
    outcome ARCHIVED_PRE_RESET -- BUKAN sebagai menang maupun kalah, karena
    hasilnya memang tidak pernah diketahui dan tidak boleh dinilai.

    Alasan ini perlu: aturan anti-duplikat memblokir sinyal searah selama
    masih ada posisi OPEN. Posisi lama yang menggantung akan mengunci pair
    itu sampai kedaluwarsa, sehingga bot tampak "tidak pernah memberi sinyal"
    padahal analisisnya jalan.

    Aman dipanggil berulang kali (idempoten): setelah diarsipkan, statusnya
    bukan OPEN lagi sehingga tidak tersentuh pemanggilan berikutnya.
    """
    epoch = _stats_since()
    if epoch is None:
        return 0

    initialize_database()
    now_iso = _to_iso(_utc_now())

    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE analyses
            SET status = 'CLOSED',
                outcome = 'ARCHIVED_PRE_RESET',
                outcome_at = ?,
                updated_at = ?
            WHERE status = 'OPEN'
              AND created_at < ?
            """,
            (now_iso, now_iso, _to_iso(epoch)),
        )
        archived = cursor.rowcount or 0

    if archived:
        logger.info(
            "Sinyal lama diarsipkan karena reset statistik | jumlah=%s | "
            "titik_nol=%s",
            archived,
            _to_iso(epoch),
        )

    return archived


def _utc_now() -> datetime:
    """Menghasilkan waktu UTC yang timezone-aware."""
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    """Menyimpan datetime dalam ISO-8601 UTC."""
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> Optional[datetime]:
    """
    Membaca timestamp ISO/provider secara toleran.

    Timestamp provider yang tidak mempunyai timezone diperlakukan sebagai UTC
    hanya untuk kebutuhan pengurutan internal.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None

        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            supported_formats = (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M",
            )
            parsed = None
            for date_format in supported_formats:
                try:
                    parsed = datetime.strptime(text, date_format)
                    break
                except ValueError:
                    continue

            if parsed is None:
                return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _json_dumps(value: Any) -> str:
    """Serialisasi JSON aman untuk data metadata."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _safe_json_object(value: Any) -> dict[str, Any]:
    """Membaca kolom JSON secara toleran; isi rusak diperlakukan kosong."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    """Konversi float yang mengembalikan None untuk nilai tidak valid."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    if number in (float("inf"), float("-inf")):
        return None

    return number


def _confidence_bucket(confidence: Optional[float]) -> str:
    """
    Melabeli confidence ke bucket yang sama dengan get_performance_summary().

    Dipertahankan konsisten dengan CASE di query bucket supaya diagnostik dan
    /stats memakai batas yang identik.
    """
    if confidence is None:
        return "?"
    if confidence < 60:
        return "<60"
    if confidence < 70:
        return "60-69"
    if confidence < 80:
        return "70-79"
    if confidence < 90:
        return "80-89"
    return "90-100"


def _resolve_database_path() -> Path:
    """Menentukan lokasi database dengan dukungan Railway Volume."""
    configured = os.getenv(DATABASE_ENV_NAME, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    if DEFAULT_VOLUME_DATABASE.parent.exists():
        return DEFAULT_VOLUME_DATABASE

    return FALLBACK_DATABASE


def get_database_path() -> Path:
    """Mengembalikan database aktif."""
    global _DATABASE_PATH

    if _DATABASE_PATH is None:
        _DATABASE_PATH = _resolve_database_path()

    return _DATABASE_PATH


def _prepare_database_parent(path: Path) -> Path:
    """
    Membuat folder database.

    Jika /data tidak dapat ditulis, tracker memakai database lokal sebagai
    fallback agar bot tetap berjalan.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        test_file = path.parent / ".signal_tracker_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return path
    except OSError:
        fallback = FALLBACK_DATABASE
        fallback.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Lokasi database %s tidak dapat ditulis. "
            "Menggunakan fallback sementara %s.",
            path,
            fallback,
        )
        return fallback


def _connect() -> sqlite3.Connection:
    """Membuka koneksi SQLite baru untuk satu operasi."""
    global _DATABASE_PATH

    resolved = _prepare_database_parent(get_database_path())
    _DATABASE_PATH = resolved

    connection = sqlite3.connect(
        resolved,
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def initialize_database() -> Path:
    """Membuat schema database secara idempotent."""
    with _connect() as connection:
        connection.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                provider_symbol TEXT,
                asset_class TEXT,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                trend TEXT,
                risk TEXT,
                entry REAL,
                sl REAL,
                tp1 REAL,
                tp2 REAL,
                tp3 REAL,
                rr1 REAL,
                rr2 REAL,
                rr3 REAL,
                atr REAL,
                reference_timeframe TEXT,
                reference_candle_time TEXT,
                latest_checked_candle_time TEXT,
                support REAL,
                resistance REAL,
                market_structure TEXT,
                bos TEXT,
                choch TEXT,
                sideways INTEGER NOT NULL DEFAULT 0,
                false_breakout TEXT,
                patterns_json TEXT NOT NULL DEFAULT '[]',
                reasons_json TEXT NOT NULL DEFAULT '[]',
                checklist_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                outcome TEXT NOT NULL,
                outcome_at TEXT,
                outcome_price REAL,
                max_tp_hit INTEGER NOT NULL DEFAULT 0,
                mfe REAL NOT NULL DEFAULT 0,
                mae REAL NOT NULL DEFAULT 0,
                expires_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_symbol_status
            ON analyses(symbol, status);

            CREATE INDEX IF NOT EXISTS idx_analyses_created_at
            ON analyses(created_at);

            CREATE INDEX IF NOT EXISTS idx_analyses_confidence
            ON analyses(confidence);

            CREATE INDEX IF NOT EXISTS idx_analyses_outcome
            ON analyses(outcome);

            CREATE TABLE IF NOT EXISTS alert_subscribers (
                chat_id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            COMMIT;
            """
        )
        # Migrasi aditif: kolom baru ditambahkan hanya kalau belum ada.
        # Dipilih ALTER TABLE (bukan ubah CREATE TABLE) supaya database
        # produksi di volume Railway tidak perlu dibuat ulang dan seluruh
        # riwayat trade lama tetap utuh.
        existing_columns = {
            str(column["name"])
            for column in connection.execute(
                "PRAGMA table_info(analyses)"
            ).fetchall()
        }
        if "realized_r" not in existing_columns:
            connection.execute(
                "ALTER TABLE analyses ADD COLUMN realized_r REAL"
            )
            logger.info(
                "Kolom realized_r ditambahkan ke tabel analyses "
                "(migrasi partial close)."
            )

        connection.execute(
            f"PRAGMA user_version = {_SCHEMA_VERSION}"
        )

    database_path = get_database_path()
    logger.info("Signal tracker database siap | path=%s", database_path)
    return database_path


def _analysis_reference_time(analysis: dict[str, Any]) -> str:
    """Mengambil timestamp candle referensi untuk deduplikasi."""
    reference_timeframe = str(
        analysis.get("atr_reference_tf") or "15min"
    )
    latest_times = analysis.get("latest_data_times")

    if isinstance(latest_times, dict):
        reference = latest_times.get(reference_timeframe)
        if reference:
            return str(reference)

        available = [
            str(value)
            for value in latest_times.values()
            if value is not None
        ]
        if available:
            return max(available)

    return _to_iso(_utc_now())


def _fingerprint_analysis(analysis: dict[str, Any]) -> str:
    """Membuat ID stabil agar hasil cache tidak dicatat berulang."""
    components = {
        "symbol": str(analysis.get("symbol", "")).upper(),
        "signal": str(analysis.get("signal", "HOLD")).upper(),
        "reference_time": _analysis_reference_time(analysis),
        "entry": _safe_float(
            analysis.get("entry")
            if analysis.get("entry") is not None
            else analysis.get("entry_price")
        ),
        "sl": _safe_float(analysis.get("sl")),
        "tp1": _safe_float(analysis.get("tp1")),
        "tp2": _safe_float(analysis.get("tp2")),
        "tp3": _safe_float(analysis.get("tp3")),
    }
    encoded = _json_dumps(components).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timeframe_minutes(timeframe: Any) -> int:
    """
    Mengubah label timeframe ("15min", "1h", "4h") menjadi menit.

    Nilai tak dikenal dikembalikan sebagai 15 menit -- sama dengan
    perilaku lama -- supaya data lama atau field kosong tidak pernah
    membuat sinyal gagal tercatat.
    """
    text = str(timeframe or "").strip().lower()
    try:
        if text.endswith("min"):
            return max(1, int(text[:-3]))
        if text.endswith("h"):
            return max(1, int(text[:-1]) * 60)
        if text.endswith("day") or text.endswith("d"):
            return max(1, int(text.rstrip("day").rstrip("d") or 1) * 1440)
    except ValueError:
        pass
    return 15


def _env_int(name: str, default: int) -> int:
    """Membaca environment variable integer secara toleran."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("%s tidak valid: %s", name, raw)
        return default


def _signal_expiry_hours(
    asset_class: str,
    reference_timeframe: Any = None,
) -> int:
    """
    Masa berlaku sinyal, diskalakan terhadap timeframe eksekusi.

    Urutan prioritas:
      1. SIGNAL_MAX_AGE_HOURS -- override absolut dalam jam (perilaku lama,
         tetap dihormati supaya setelan yang sudah terpasang tidak berubah).
      2. Jumlah candle x panjang timeframe referensi.

    Pada M15 hasilnya identik dengan konstanta lama (12 jam / 24 jam crypto),
    jadi mengganti file ini TIDAK mengubah perilaku produksi saat ini.
    """
    configured = os.getenv("SIGNAL_MAX_AGE_HOURS", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            logger.warning(
                "SIGNAL_MAX_AGE_HOURS tidak valid: %s",
                configured,
            )

    is_crypto = str(asset_class).lower() == "crypto"
    candles = _env_int(
        "SIGNAL_EXPIRY_CANDLES",
        CRYPTO_SIGNAL_EXPIRY_CANDLES if is_crypto else SIGNAL_EXPIRY_CANDLES,
    )
    minutes = _timeframe_minutes(
        reference_timeframe or _REFERENCE_TIMEFRAME_FALLBACK
    )
    return max(1, round(candles * minutes / 60))


def record_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Menyimpan satu hasil analisis.

    BUY/SELL dicatat sebagai OPEN.
    HOLD dicatat sebagai NO_TRADE agar dapat dipakai untuk kalibrasi.
    """
    initialize_database()

    symbol = str(analysis.get("symbol", "")).strip().upper()
    if not symbol:
        raise ValueError("Analysis tidak mempunyai symbol.")

    signal = str(
        analysis.get("signal")
        or analysis.get("direction")
        or "HOLD"
    ).upper()

    if signal == "NEUTRAL":
        signal = "HOLD"

    if signal not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"Signal tidak didukung: {signal}")

    asset_class = str(analysis.get("asset_class", "unknown"))
    created_at = _utc_now()
    reference_candle_time = _analysis_reference_time(analysis)

    entry = _safe_float(
        analysis.get("entry")
        if analysis.get("entry") is not None
        else analysis.get("entry_price")
    )
    sl = _safe_float(analysis.get("sl"))
    tp1 = _safe_float(analysis.get("tp1"))
    tp2 = _safe_float(analysis.get("tp2"))
    tp3 = _safe_float(analysis.get("tp3"))

    if signal in {"BUY", "SELL"}:
        required_levels = {
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
        }
        missing = [
            name
            for name, value in required_levels.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "Level trade tidak lengkap: " + ", ".join(missing)
            )

        status = "OPEN"
        outcome = "PENDING"
        expires_at = created_at + timedelta(
            hours=_signal_expiry_hours(
                asset_class,
                analysis.get("atr_reference_tf"),
            )
        )
    else:
        status = "NO_TRADE"
        outcome = "HOLD"
        expires_at = None

    fingerprint = _fingerprint_analysis(analysis)
    now_iso = _to_iso(created_at)

    metadata = {
        "coverage_pct": analysis.get("coverage_pct"),
        "directional_agreement_pct": analysis.get(
            "directional_agreement_pct"
        ),
        "structure_confirmation_pct": analysis.get(
            "structure_confirmation_pct"
        ),
        "pattern_confirmation_pct": analysis.get(
            "pattern_confirmation_pct"
        ),
        "sideways_ratio_pct": analysis.get("sideways_ratio_pct"),
        "false_breakout_against_pct": analysis.get(
            "false_breakout_against_pct"
        ),
        # Diagnostik P2 -- overextension/late-entry per trade. Sebelumnya
        # nilai ini hanya muncul di log & pesan Telegram lalu hilang,
        # sehingga korelasi overextension vs outcome (mis. SL langsung)
        # tidak bisa dianalisis dari DB. Sekarang ikut disimpan supaya
        # /diag dan kalibrasi bobot ke depan bisa memakainya. Trade lama
        # (sebelum perubahan ini) tidak punya key ini -> diagnostik
        # menampilkannya sebagai "-".
        "overextension_ratio_pct": analysis.get("overextension_ratio_pct"),
        # Bobot partial close DISIMPAN PER TRADE, bukan dibaca dari konfigurasi
        # saat menghitung. Kalau bobot diubah lewat env di kemudian hari,
        # trade lama tetap dinilai dengan bobot yang berlaku saat dibuka --
        # supaya riwayat R tidak berubah retroaktif.
        "partial_weights": [
            float(item.get("weight") or 0.0)
            for item in (analysis.get("partial_plan") or [])
        ],
        # Gate pemblokir untuk analisis yang berakhir HOLD. Inilah yang
        # memungkinkan /diag menjawab "gate mana yang paling sering
        # menahan sinyal" dengan angka.
        "hold_blockers": analysis.get("hold_blockers") or [],
        "hold_blocker_primary": analysis.get("hold_blocker_primary"),
        "cache_hit": analysis.get("cache_hit", False),
        "cache_age_seconds": analysis.get(
            "cache_age_seconds",
            0.0,
        ),
    }

    values = (
        fingerprint,
        now_iso,
        now_iso,
        symbol,
        analysis.get("provider_symbol"),
        asset_class,
        signal,
        float(analysis.get("confidence_pct", 0.0) or 0.0),
        analysis.get("trend"),
        analysis.get("risk"),
        entry,
        sl,
        tp1,
        tp2,
        tp3,
        _safe_float(analysis.get("risk_reward_tp1")),
        _safe_float(analysis.get("risk_reward_tp2")),
        _safe_float(analysis.get("risk_reward_tp3")),
        _safe_float(analysis.get("atr_value")),
        analysis.get("atr_reference_tf"),
        reference_candle_time,
        None,
        _safe_float(analysis.get("support")),
        _safe_float(analysis.get("resistance")),
        analysis.get("market_structure"),
        analysis.get("bos"),
        analysis.get("choch"),
        1 if analysis.get("sideways") else 0,
        analysis.get("false_breakout"),
        _json_dumps(analysis.get("candlestick_patterns", [])),
        _json_dumps(analysis.get("reasons", [])),
        _json_dumps(analysis.get("indicator_checklist", [])),
        status,
        outcome,
        _to_iso(expires_at) if expires_at is not None else None,
        _json_dumps(metadata),
    )

    with _connect() as connection:
        # Anti-duplikat posisi: satu ide, satu posisi. Dicek di dalam
        # koneksi yang sama dengan INSERT supaya tidak ada celah balapan
        # antara scanner dan analisis manual yang berjalan bersamaan.
        if (
            signal in {"BUY", "SELL"}
            and not ALLOW_DUPLICATE_OPEN
        ):
            existing = connection.execute(
                """
                SELECT id, created_at, entry
                FROM analyses
                WHERE symbol = ?
                  AND signal = ?
                  AND status = 'OPEN'
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (symbol, signal),
            ).fetchone()

            if existing is not None:
                logger.info(
                    "Sinyal duplikat dilewati | symbol=%s | signal=%s | "
                    "posisi_aktif_id=%s | dibuka=%s | entry_lama=%s | "
                    "entry_baru=%s",
                    symbol,
                    signal,
                    existing["id"],
                    existing["created_at"],
                    existing["entry"],
                    entry,
                )
                return {
                    "id": None,
                    "fingerprint": None,
                    "symbol": symbol,
                    "signal": signal,
                    "status": "SKIPPED_DUPLICATE",
                    "outcome": "SKIPPED_DUPLICATE",
                    "max_tp_hit": 0,
                    "created_at": now_iso,
                    "inserted": False,
                    "skipped_reason": "duplicate_open",
                    "blocking_id": existing["id"],
                }

        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO analyses (
                fingerprint,
                created_at,
                updated_at,
                symbol,
                provider_symbol,
                asset_class,
                signal,
                confidence,
                trend,
                risk,
                entry,
                sl,
                tp1,
                tp2,
                tp3,
                rr1,
                rr2,
                rr3,
                atr,
                reference_timeframe,
                reference_candle_time,
                latest_checked_candle_time,
                support,
                resistance,
                market_structure,
                bos,
                choch,
                sideways,
                false_breakout,
                patterns_json,
                reasons_json,
                checklist_json,
                status,
                outcome,
                expires_at,
                metadata_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            values,
        )

        inserted = cursor.rowcount == 1

        row = connection.execute(
            """
            SELECT id, fingerprint, symbol, signal, status, outcome,
                   max_tp_hit, created_at
            FROM analyses
            WHERE fingerprint = ?
            """,
            (fingerprint,),
        ).fetchone()

    if row is None:
        raise RuntimeError("Gagal membaca hasil pencatatan signal.")

    result = dict(row)
    result["inserted"] = inserted

    logger.info(
        "Analysis logged | id=%s | symbol=%s | signal=%s | inserted=%s",
        result["id"],
        symbol,
        signal,
        inserted,
    )
    return result


def _normalize_candles(
    candles: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Membersihkan dan mengurutkan candle untuk outcome tracking."""
    normalized: list[dict[str, Any]] = []

    for candle in candles:
        if not isinstance(candle, dict):
            continue

        candle_time = _parse_datetime(
            candle.get("datetime") or candle.get("time")
        )
        high = _safe_float(candle.get("high"))
        low = _safe_float(candle.get("low"))
        close = _safe_float(candle.get("close"))

        if (
            candle_time is None
            or high is None
            or low is None
            or close is None
        ):
            continue

        if high < low:
            continue

        normalized.append(
            {
                "datetime": candle_time,
                "high": high,
                "low": low,
                "close": close,
            }
        )

    normalized.sort(key=lambda item: item["datetime"])

    deduplicated: dict[str, dict[str, Any]] = {}
    for candle in normalized:
        deduplicated[_to_iso(candle["datetime"])] = candle

    return list(deduplicated.values())


def _calculate_excursions(
    signal: str,
    entry: float,
    high: float,
    low: float,
) -> tuple[float, float]:
    """Menghitung Maximum Favorable/Adverse Excursion dalam harga."""
    if signal == "BUY":
        favorable = max(0.0, high - entry)
        adverse = max(0.0, entry - low)
    else:
        favorable = max(0.0, entry - low)
        adverse = max(0.0, high - entry)

    return favorable, adverse


def _tp_hit_for_candle(
    signal: str,
    candle: dict[str, Any],
    tp1: Optional[float],
    tp2: Optional[float],
    tp3: Optional[float],
) -> int:
    """Mengembalikan target tertinggi yang tersentuh candle."""
    high = float(candle["high"])
    low = float(candle["low"])

    if signal == "BUY":
        if tp3 is not None and high >= tp3:
            return 3
        if tp2 is not None and high >= tp2:
            return 2
        if tp1 is not None and high >= tp1:
            return 1
    else:
        if tp3 is not None and low <= tp3:
            return 3
        if tp2 is not None and low <= tp2:
            return 2
        if tp1 is not None and low <= tp1:
            return 1

    return 0


def _sl_hit_for_candle(
    signal: str,
    candle: dict[str, Any],
    sl: float,
) -> bool:
    """Memeriksa sentuhan SL."""
    if signal == "BUY":
        return float(candle["low"]) <= sl

    return float(candle["high"]) >= sl


def _closed_outcome_after_sl(max_tp_hit: int) -> str:
    """Nama outcome saat SL tercapai."""
    if max_tp_hit <= 0:
        return "SL"

    return f"SL_AFTER_TP{max_tp_hit}"


def _partial_weights_from_metadata(
    metadata: Any,
) -> tuple[float, float, float]:
    """
    Bobot partial close milik satu trade, dinormalisasi ke jumlah 1.0.

    Trade yang dibuka SEBELUM fitur partial close tidak punya key ini. Untuk
    trade seperti itu dikembalikan (0, 0, 1) = seluruh posisi ditutup di TP3,
    yaitu perilaku lama persis. Riwayat lama karena itu tidak berubah
    nilainya, dan perbandingan sebelum/sesudah tetap jujur.
    """
    fallback = (0.0, 0.0, 1.0)

    if not isinstance(metadata, dict):
        return fallback

    weights = metadata.get("partial_weights")
    if not isinstance(weights, (list, tuple)) or len(weights) < 3:
        return fallback

    try:
        values = [max(0.0, float(weight)) for weight in weights[:3]]
    except (TypeError, ValueError):
        return fallback

    total = sum(values)
    if total <= 0:
        return fallback

    return (values[0] / total, values[1] / total, values[2] / total)


def _realized_r(
    signal: str,
    entry: float,
    risk: float,
    rr_ladder: tuple[Optional[float], Optional[float], Optional[float]],
    weights: tuple[float, float, float],
    max_tp_hit: int,
    exit_price: Optional[float],
    breakeven: bool,
) -> Optional[float]:
    """
    Menghitung hasil trade dalam satuan R dengan memperhitungkan partial close.

    Ini perbaikan akuntansi, BUKAN perubahan aturan keluar. Mesin exit di
    update_open_signals tidak diubah sama sekali: SL tetap SL, breakeven
    tetap breakeven, TP3 tetap menutup trade. Yang berubah hanya cara hasil
    dinilai.

    Sebelumnya tracker hanya mengenal "TP3 penuh atau tidak sama sekali",
    sehingga trade yang menyentuh TP1 lalu berbalik tercatat 0R -- padahal
    di eksekusi nyata separuh posisi sudah dibukukan di TP1. Dari 40 trade
    28-29 Jul, 17 trade sudah bergerak untung dan hanya 2 yang tercatat
    menghasilkan.

    Perhitungan:
        bagian yang sudah keluar di TPn  -> bobot_n x RR_n
        sisa posisi                      -> R harga keluar terakhir
                                            (breakeven = 0, SL = -1,
                                             expired = selisih nyata)
    """
    if risk <= 0:
        return None

    ladder = [
        value if isinstance(value, (int, float)) else None
        for value in rr_ladder
    ]

    banked = 0.0
    closed_weight = 0.0
    for index in range(min(max(max_tp_hit, 0), 3)):
        level_rr = ladder[index]
        if level_rr is None:
            continue
        banked += weights[index] * float(level_rr)
        closed_weight += weights[index]

    remaining_weight = max(0.0, 1.0 - closed_weight)

    if remaining_weight <= 0:
        return round(banked, 4)

    if breakeven:
        remainder_r = 0.0
    elif exit_price is None:
        remainder_r = 0.0
    else:
        direction = 1.0 if signal == "BUY" else -1.0
        remainder_r = direction * (float(exit_price) - entry) / risk

    return round(banked + (remaining_weight * remainder_r), 4)


def _breakeven_outcome(max_tp_hit: int) -> str:
    """Nama outcome saat SL breakeven (di entry) tersentuh.

    Berbeda dari SL_AFTER_TP: ini BUKAN kerugian penuh, melainkan trade yang
    ditutup impas (0R) karena SL sudah dipindah ke entry. Dua rute breakeven
    mungkin aktif:
      - Setelah TP1 (max_tp_hit >= 1): breakeven lama (P1) -> outcome
        BREAKEVEN_AFTER_TPn.
      - SEBELUM TP1 (max_tp_hit == 0), lewat Lever B (BREAKEVEN_AT_MFE_R):
        profit mengambang sudah cukup jauh tapi TP1 belum tersentuh ->
        outcome BREAKEVEN_PRE_TP1. Ini SENGAJA dibedakan dari
        BREAKEVEN_AFTER_TP0 supaya /stats bisa memisahkan dua rezim risiko
        yang berbeda (setelah bank profit 2R vs baru 1R).
    """
    if max_tp_hit <= 0:
        return "BREAKEVEN_PRE_TP1"

    return f"BREAKEVEN_AFTER_TP{max_tp_hit}"


def _expired_outcome(max_tp_hit: int) -> str:
    """Nama outcome saat sinyal kedaluwarsa."""
    if max_tp_hit <= 0:
        return "EXPIRED"

    return f"TP{max_tp_hit}_THEN_EXPIRED"


def update_open_signals(
    symbol: str,
    candles: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """
    Memperbarui outcome seluruh signal OPEN untuk satu symbol.

    Candle diproses kronologis. Candle yang sama dengan candle pembentuk
    signal tidak digunakan untuk menilai hasil.
    """
    initialize_database()

    canonical_symbol = str(symbol).strip().upper()
    normalized_candles = _normalize_candles(candles)
    now = _utc_now()

    counters = {
        "checked": 0,
        "updated": 0,
        "closed": 0,
        "expired": 0,
    }

    with _connect() as connection:
        open_rows = connection.execute(
            """
            SELECT *
            FROM analyses
            WHERE symbol = ?
              AND status = 'OPEN'
            ORDER BY created_at ASC
            """,
            (canonical_symbol,),
        ).fetchall()

        for row in open_rows:
            counters["checked"] += 1

            signal = str(row["signal"]).upper()
            entry = _safe_float(row["entry"])
            sl = _safe_float(row["sl"])
            tp1 = _safe_float(row["tp1"])
            tp2 = _safe_float(row["tp2"])
            tp3 = _safe_float(row["tp3"])

            if signal not in {"BUY", "SELL"} or entry is None or sl is None:
                logger.error(
                    "Signal open tidak valid | id=%s",
                    row["id"],
                )
                continue

            reference_time = _parse_datetime(
                row["reference_candle_time"]
            )
            last_checked = _parse_datetime(
                row["latest_checked_candle_time"]
            )
            threshold = last_checked or reference_time

            new_candles = [
                candle
                for candle in normalized_candles
                if threshold is None
                or candle["datetime"] > threshold
            ]

            max_tp_hit = int(row["max_tp_hit"] or 0)
            mfe = float(row["mfe"] or 0.0)
            mae = float(row["mae"] or 0.0)
            risk = abs(entry - sl)
            status = "OPEN"
            outcome = str(row["outcome"] or "PENDING")
            outcome_at = row["outcome_at"]
            outcome_price = row["outcome_price"]
            latest_checked = row["latest_checked_candle_time"]

            for candle in new_candles:
                favorable, adverse = _calculate_excursions(
                    signal,
                    entry,
                    float(candle["high"]),
                    float(candle["low"]),
                )
                # Breakeven-dini (Lever B) dicek pakai MFE SEBELUM candle
                # ini diproses -- konsisten dengan cara max_tp_hit dicek
                # untuk breakeven-setelah-TP1 (hindari asumsi urutan
                # intrabar: kita tidak tahu apakah high atau low duluan
                # dalam satu candle).
                mfe_before_candle = mfe
                mfe = max(mfe, favorable)
                mae = max(mae, adverse)
                latest_checked = _to_iso(candle["datetime"])

                # Breakeven setelah TP1: kalau TP1 sudah tersentuh di candle
                # SEBELUMNYA (max_tp_hit >= 1), SL efektif dipindah ke entry.
                # SL awal hanya berlaku selama trade belum menyentuh TP1.
                # Catatan urutan: pada candle di mana TP1 PERTAMA kali kena,
                # max_tp_hit di titik ini masih nilai dari candle sebelumnya
                # (0), sehingga SL awal tetap dipakai untuk candle itu --
                # konservatif dan menghindari asumsi urutan intrabar.
                breakeven_from_tp1 = (
                    BREAKEVEN_AFTER_TP1 and max_tp_hit >= 1
                )
                # Lever B: breakeven begitu MFE (sebelum candle ini) sudah
                # mencapai BREAKEVEN_AT_MFE_R x risk, walau TP1 (2R) belum
                # kena. BREAKEVEN_AT_MFE_R<=0 mematikan lever ini sepenuhnya
                # (perilaku identik dengan sebelum Lever B ada).
                breakeven_from_mfe = (
                    BREAKEVEN_AT_MFE_R > 0
                    and risk > 0
                    and mfe_before_candle >= (BREAKEVEN_AT_MFE_R * risk)
                )
                breakeven_active = breakeven_from_tp1 or breakeven_from_mfe
                effective_sl = entry if breakeven_active else sl

                # Konservatif: SL diperiksa sebelum TP di candle yang sama.
                if _sl_hit_for_candle(signal, candle, effective_sl):
                    status = "CLOSED"
                    if breakeven_active:
                        outcome = _breakeven_outcome(max_tp_hit)
                        outcome_price = entry
                    else:
                        outcome = _closed_outcome_after_sl(max_tp_hit)
                        outcome_price = sl
                    outcome_at = latest_checked
                    break

                candle_tp_hit = _tp_hit_for_candle(
                    signal,
                    candle,
                    tp1,
                    tp2,
                    tp3,
                )
                max_tp_hit = max(max_tp_hit, candle_tp_hit)

                if max_tp_hit >= 3:
                    status = "CLOSED"
                    outcome = "TP3"
                    outcome_at = latest_checked
                    outcome_price = tp3
                    break

                if max_tp_hit > 0:
                    outcome = f"TP{max_tp_hit}_OPEN"

            expires_at = _parse_datetime(row["expires_at"])
            if status == "OPEN" and expires_at is not None and now >= expires_at:
                status = "CLOSED"
                outcome = _expired_outcome(max_tp_hit)
                outcome_at = _to_iso(now)
                outcome_price = (
                    float(new_candles[-1]["close"])
                    if new_candles
                    else row["outcome_price"]
                )
                counters["expired"] += 1

            # --- Realized R (partial close) --------------------------
            # Dihitung hanya saat trade benar-benar ditutup. Selama OPEN
            # nilainya sengaja dibiarkan NULL supaya tidak ada angka
            # setengah jadi yang ikut masuk ke statistik.
            realized_r = _safe_float(row["realized_r"])
            if status == "CLOSED":
                realized_r = _realized_r(
                    signal=signal,
                    entry=entry,
                    risk=risk,
                    rr_ladder=(
                        _safe_float(row["rr1"]),
                        _safe_float(row["rr2"]),
                        _safe_float(row["rr3"]),
                    ),
                    weights=_partial_weights_from_metadata(
                        _safe_json_object(row["metadata_json"])
                    ),
                    max_tp_hit=max_tp_hit,
                    exit_price=outcome_price,
                    breakeven=str(outcome).startswith("BREAKEVEN"),
                )

            changed = (
                status != row["status"]
                or outcome != row["outcome"]
                or max_tp_hit != int(row["max_tp_hit"] or 0)
                or abs(mfe - float(row["mfe"] or 0.0)) > 1e-12
                or abs(mae - float(row["mae"] or 0.0)) > 1e-12
                or latest_checked != row["latest_checked_candle_time"]
                or realized_r != _safe_float(row["realized_r"])
            )

            if not changed:
                continue

            connection.execute(
                """
                UPDATE analyses
                SET updated_at = ?,
                    latest_checked_candle_time = ?,
                    status = ?,
                    outcome = ?,
                    outcome_at = ?,
                    outcome_price = ?,
                    max_tp_hit = ?,
                    mfe = ?,
                    mae = ?,
                    realized_r = ?
                WHERE id = ?
                """,
                (
                    _to_iso(now),
                    latest_checked,
                    status,
                    outcome,
                    outcome_at,
                    outcome_price,
                    max_tp_hit,
                    mfe,
                    mae,
                    realized_r,
                    row["id"],
                ),
            )

            counters["updated"] += 1
            if status == "CLOSED":
                counters["closed"] += 1

    logger.info(
        "Outcome tracker updated | symbol=%s | checked=%s | "
        "updated=%s | closed=%s",
        canonical_symbol,
        counters["checked"],
        counters["updated"],
        counters["closed"],
    )
    return counters


def get_open_symbols() -> list[str]:
    """Mengembalikan symbol yang masih mempunyai signal OPEN."""
    initialize_database()

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT symbol
            FROM analyses
            WHERE status = 'OPEN'
            ORDER BY symbol
            """
        ).fetchall()

    return [str(row["symbol"]) for row in rows]


def get_recent_signals(limit: int = 10) -> list[dict[str, Any]]:
    """Mengambil histori analisis terbaru."""
    initialize_database()

    safe_limit = max(1, min(int(limit), MAX_RECENT_SIGNAL_LIMIT))

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id,
                   created_at,
                   symbol,
                   signal,
                   confidence,
                   trend,
                   status,
                   outcome,
                   max_tp_hit,
                   entry,
                   sl,
                   tp1,
                   tp2,
                   tp3
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_signals_for_website(limit: int = 20) -> list[dict[str, Any]]:
    """
    Mengambil sinyal BUY/SELL terbaru dengan detail lengkap untuk Web API.

    Berbeda dari get_recent_signals() (dipakai /signals di Telegram, kolom
    terbatas): fungsi ini menyertakan rr1/rr2/rr3 dan reasons_json yang
    di-parse, supaya dashboard website bisa menampilkan RR dan alasan sinyal
    persis seperti pesan Telegram -- tanpa duplikasi logika penghitungan.

    Read-only, tidak menyentuh logika trading/scoring. Urutan: OPEN dulu
    (masih relevan untuk dipantau), baru CLOSED terbaru.
    """
    initialize_database()

    safe_limit = max(1, min(int(limit), MAX_RECENT_SIGNAL_LIMIT))

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id,
                   created_at,
                   symbol,
                   signal,
                   confidence,
                   trend,
                   status,
                   outcome,
                   max_tp_hit,
                   entry,
                   sl,
                   tp1,
                   tp2,
                   tp3,
                   rr1,
                   rr2,
                   rr3,
                   reasons_json
            FROM analyses
            WHERE signal IN ('BUY', 'SELL')
            ORDER BY
                CASE WHEN status = 'OPEN' THEN 0 ELSE 1 END,
                id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    signals: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            reasons = json.loads(item.pop("reasons_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            reasons = []
        item["reasons"] = reasons if isinstance(reasons, list) else []
        signals.append(item)

    return signals


def get_performance_summary(days: int = 30) -> dict[str, Any]:
    """Menghasilkan statistik ringkas untuk kalibrasi awal."""
    initialize_database()

    safe_days = max(1, min(int(days), 3650))
    start_time = _report_start_time(safe_days)

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_analyses,
                SUM(CASE WHEN signal = 'HOLD' THEN 1 ELSE 0 END) AS holds,
                SUM(CASE WHEN signal IN ('BUY', 'SELL') THEN 1 ELSE 0 END)
                    AS trade_signals,
                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END)
                    AS open_trades,
                SUM(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                         AND status = 'CLOSED'
                        THEN 1
                        ELSE 0
                    END
                ) AS completed_trades,
                SUM(CASE WHEN max_tp_hit >= 1 THEN 1 ELSE 0 END)
                    AS tp1_or_better,
                SUM(CASE WHEN max_tp_hit >= 2 THEN 1 ELSE 0 END)
                    AS tp2_or_better,
                SUM(CASE WHEN max_tp_hit >= 3 THEN 1 ELSE 0 END)
                    AS tp3_hits,
                SUM(CASE WHEN outcome = 'SL' THEN 1 ELSE 0 END)
                    AS direct_sl,
                SUM(CASE WHEN outcome LIKE 'SL_AFTER_TP%' THEN 1 ELSE 0 END)
                    AS sl_after_tp,
                SUM(
                    CASE
                        WHEN outcome LIKE 'BREAKEVEN_AFTER_TP%'
                        THEN 1
                        ELSE 0
                    END
                ) AS breakeven_after_tp,
                SUM(
                    CASE
                        WHEN outcome = 'BREAKEVEN_PRE_TP1'
                        THEN 1
                        ELSE 0
                    END
                ) AS breakeven_pre_tp1,
                SUM(CASE WHEN outcome LIKE '%EXPIRED%' THEN 1 ELSE 0 END)
                    AS expired,
                SUM(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                         AND status = 'CLOSED'
                         AND max_tp_hit >= 1
                        THEN 1
                        ELSE 0
                    END
                ) AS completed_tp1_or_better,
                SUM(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                         AND status = 'CLOSED'
                         AND entry IS NOT NULL
                         AND outcome_price IS NOT NULL
                         AND (
                             (signal = 'BUY' AND outcome_price > entry)
                             OR (signal = 'SELL' AND outcome_price < entry)
                         )
                        THEN 1
                        ELSE 0
                    END
                ) AS completed_wins,
                AVG(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                        THEN confidence
                        ELSE NULL
                    END
                ) AS average_trade_confidence,
                -- Ekspektasi dalam satuan R. Ini metrik paling jujur yang
                -- dimiliki sistem: win rate memperlakukan breakeven sebagai
                -- "bukan menang" dan trade partial sebagai kalah penuh,
                -- sedangkan total R menghitung yang benar-benar dibukukan.
                SUM(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                         AND status = 'CLOSED'
                         AND realized_r IS NOT NULL
                        THEN realized_r
                        ELSE 0
                    END
                ) AS total_r,
                SUM(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                         AND status = 'CLOSED'
                         AND realized_r IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS scored_trades
            FROM analyses
            WHERE created_at >= ?
            """,
            (start_time,),
        ).fetchone()

        bucket_rows = connection.execute(
            """
            SELECT
                CASE
                    WHEN confidence < 60 THEN '<60'
                    WHEN confidence < 70 THEN '60-69'
                    WHEN confidence < 80 THEN '70-79'
                    WHEN confidence < 90 THEN '80-89'
                    ELSE '90-100'
                END AS bucket,
                COUNT(*) AS signals,
                SUM(CASE WHEN max_tp_hit >= 1 THEN 1 ELSE 0 END)
                    AS tp1_or_better,
                SUM(CASE WHEN outcome = 'SL' THEN 1 ELSE 0 END)
                    AS direct_sl,
                SUM(
                    CASE
                        WHEN status = 'CLOSED' AND realized_r IS NOT NULL
                        THEN realized_r
                        ELSE 0
                    END
                ) AS bucket_total_r,
                SUM(
                    CASE
                        WHEN status = 'CLOSED' AND realized_r IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS bucket_scored
            FROM analyses
            WHERE created_at >= ?
              AND signal IN ('BUY', 'SELL')
            GROUP BY bucket
            ORDER BY
                CASE bucket
                    WHEN '<60' THEN 1
                    WHEN '60-69' THEN 2
                    WHEN '70-79' THEN 3
                    WHEN '80-89' THEN 4
                    ELSE 5
                END
            """,
            (start_time,),
        ).fetchall()

    summary = dict(row) if row is not None else {}

    # SQLite mengembalikan NULL (bukan 0) untuk SUM() di atas himpunan
    # kosong. Sebelum ada reset statistik, kondisi itu praktis tidak pernah
    # terjadi karena database selalu berisi data. Setelah STATS_SINCE
    # dipakai, `/stats` pertama pasca-reset memang menghitung nol baris --
    # dan seluruh agregat jadi None. Formatter Telegram kebetulan aman
    # (`int(x or 0)`), tapi konsumen lain (Web API, kalkulasi bucket)
    # belum tentu. Dinormalkan sekali di sini supaya kontrak fungsi ini
    # selalu mengembalikan angka, bukan campuran angka dan None.
    _COUNT_FIELDS = (
        "total_analyses",
        "holds",
        "trade_signals",
        "open_trades",
        "completed_trades",
        "tp1_or_better",
        "tp2_or_better",
        "tp3_hits",
        "direct_sl",
        "sl_after_tp",
        "breakeven_after_tp",
        "breakeven_pre_tp1",
        "expired",
        "completed_tp1_or_better",
        "completed_wins",
        "scored_trades",
    )
    for field in _COUNT_FIELDS:
        if field in summary:
            summary[field] = int(summary[field] or 0)

    if summary.get("average_trade_confidence") is None:
        summary["average_trade_confidence"] = 0.0

    completed = int(summary.get("completed_trades") or 0)
    completed_tp1_or_better = int(
        summary.get("completed_tp1_or_better") or 0
    )
    completed_wins = int(summary.get("completed_wins") or 0)

    # --- Ekspektasi R -------------------------------------------------
    # expectancy_r adalah rata-rata R per trade yang sudah selesai. Inilah
    # angka yang menentukan apakah sistem menghasilkan uang atau tidak;
    # win rate bisa terlihat rendah sementara ekspektasi positif (dan
    # sebaliknya). Trade lama yang ditutup sebelum kolom realized_r ada
    # tidak punya nilai dan sengaja TIDAK diperhitungkan -- lebih baik
    # sampel lebih kecil daripada angka yang dikarang.
    total_r = float(summary.get("total_r") or 0.0)
    scored_trades = int(summary.get("scored_trades") or 0)
    summary["total_r"] = round(total_r, 2)
    summary["scored_trades"] = scored_trades
    summary["expectancy_r"] = (
        round(total_r / scored_trades, 3) if scored_trades > 0 else 0.0
    )

    summary["days"] = safe_days
    # Titik nol statistik, supaya laporan Telegram bisa menyatakan terus
    # terang bahwa angkanya dihitung sejak reset -- bukan sepanjang histori.
    epoch = _stats_since()
    summary["stats_since"] = _to_iso(epoch) if epoch is not None else None

    # tp1_hit_rate_pct: dari trade yang SUDAH SELESAI (CLOSED), berapa persen
    # yang sempat menyentuh TP1 atau lebih. Pembilang dan penyebut kini SAMA-SAMA
    # hanya menghitung trade CLOSED. Sebelumnya pembilang keliru ikut menghitung
    # trade yang masih OPEN (max_tp_hit >= 1 tanpa filter status), sementara
    # penyebut hanya CLOSED -> angka bisa sangat menyesatkan (mis. 90-100%).
    summary["tp1_hit_rate_pct"] = (
        round((completed_tp1_or_better / completed) * 100, 1)
        if completed > 0
        else 0.0
    )

    # win_rate_closed_pct: win rate FINAL yang benar-benar terealisasi.
    # Sebuah trade dihitung MENANG hanya jika harga penutupannya (outcome_price)
    # berada di sisi profit relatif terhadap entry (BUY: close > entry,
    # SELL: close < entry). Trade yang ditutup impas di breakeven (outcome_price
    # == entry) TIDAK dihitung menang maupun kalah -- lihat breakeven_after_tp.
    # Outcome "SL setelah TP" (bila breakeven dimatikan) tetap dihitung KALAH.
    summary["win_rate_closed_pct"] = (
        round((completed_wins / completed) * 100, 1)
        if completed > 0
        else 0.0
    )

    summary["completed_tp1_or_better"] = completed_tp1_or_better
    summary["completed_wins"] = completed_wins
    summary["breakeven_after_tp"] = int(
        summary.get("breakeven_after_tp") or 0
    )
    summary["breakeven_pre_tp1"] = int(
        summary.get("breakeven_pre_tp1") or 0
    )
    # Bucket confidence kini juga membawa ekspektasi R-nya sendiri. Selama
    # ini bucket hanya menampilkan "berapa persen menyentuh TP1", dan itu
    # tidak cukup untuk memutuskan ambang: bucket bisa sering menyentuh TP1
    # tapi tetap merugi, atau jarang menyentuh TP1 tapi menguntungkan karena
    # pemenangnya besar. Dengan kolom R di sini, kalibrasi ambang berikutnya
    # bisa memakai angka yang benar.
    buckets = []
    for bucket_row in bucket_rows:
        bucket = dict(bucket_row)
        bucket_scored = int(bucket.pop("bucket_scored", 0) or 0)
        bucket_total_r = float(bucket.pop("bucket_total_r", 0.0) or 0.0)
        bucket["scored_trades"] = bucket_scored
        bucket["total_r"] = round(bucket_total_r, 2)
        bucket["expectancy_r"] = (
            round(bucket_total_r / bucket_scored, 3)
            if bucket_scored > 0
            else None
        )
        buckets.append(bucket)

    summary["confidence_buckets"] = buckets
    summary["database_path"] = str(get_database_path())
    return summary


def get_hold_blocker_census(days: int = 30) -> dict[str, Any]:
    """
    Menghitung gate mana yang paling sering mengubah sinyal menjadi HOLD.

    Read-only, tidak menyentuh logika trading. Ini jawaban berbasis angka
    untuk pertanyaan "kenapa HOLD terus": selama ini setiap HOLD tercatat
    tanpa sebab, sehingga pelonggaran ambang hanya bisa dilakukan dengan
    menebak. Dengan sensus ini, gate yang paling sering memblokir bisa
    dilonggarkan secara terarah -- satu gate, dengan alasan, lalu diukur.

    Catatan: hanya analisis yang direkam SETELAH fitur ini aktif yang punya
    data. HOLD lama tercatat sebagai 'tanpa_data' dan tidak dipaksa masuk ke
    kategori mana pun.
    """
    initialize_database()

    safe_days = max(1, min(int(days), 3650))
    start_time = _report_start_time(safe_days)

    counts: dict[str, int] = {}
    primary_counts: dict[str, int] = {}
    holds_with_data = 0
    holds_without_data = 0
    near_miss = 0

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT confidence, metadata_json
            FROM analyses
            WHERE created_at >= ?
              AND signal = 'HOLD'
            """,
            (start_time,),
        ).fetchall()

    for row in rows:
        metadata = _safe_json_object(row["metadata_json"])
        blockers = metadata.get("hold_blockers")

        if not isinstance(blockers, list) or not blockers:
            holds_without_data += 1
            continue

        holds_with_data += 1

        # "Nyaris lolos" = hanya SATU gate yang menahan. Kelompok ini yang
        # paling berharga: melonggarkan satu gate akan langsung mengubahnya
        # menjadi sinyal, tanpa menyentuh gate lain.
        if len(blockers) == 1:
            near_miss += 1

        for blocker in blockers:
            key = str(blocker)
            counts[key] = counts.get(key, 0) + 1

        # Pemblokir utama: gate pertama yang gagal. Trade lama tidak punya
        # field ini, jadi diturunkan dari urutan daftar sebagai cadangan.
        primary = metadata.get("hold_blocker_primary") or blockers[0]
        primary_counts[str(primary)] = primary_counts.get(str(primary), 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    return {
        "days": safe_days,
        "holds_total": len(rows),
        "holds_with_data": holds_with_data,
        "holds_without_data": holds_without_data,
        "near_miss_single_gate": near_miss,
        "blockers": [
            {"gate": gate, "count": count} for gate, count in ranked
        ],
        "primary_blockers": [
            {"gate": gate, "count": count}
            for gate, count in sorted(
                primary_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def get_direct_sl_diagnostics(days: int = 30) -> dict[str, Any]:
    """
    Diagnostik read-only untuk trade yang kena SL LANGSUNG.

    Fokus: trade BUY/SELL yang CLOSED dengan outcome='SL' -- yaitu kena SL
    tanpa pernah menyentuh TP1. Tujuannya menguji hipotesis "entry telat /
    lokasi entry buruk" tanpa mengubah logika trading apa pun.

    Kenapa MFE jadi proxy:
        overextension_ratio hanya mulai tersimpan untuk trade BARU (lihat
        record_analysis). Untuk trade lama nilai itu tidak ada, jadi proxy
        utamanya adalah MFE (Maximum Favorable Excursion) yang MEMANG sudah
        tersimpan sejak awal:
          - mfe_r  = MFE / risk (|entry - sl|). Mendekati 0 artinya harga
                     nyaris tidak pernah bergerak ke arah kita sebelum kena
                     SL -> ciri entry buruk/telat (mendukung hipotesis).
          - mfe_tp1= MFE / jarak(entry->TP1). Mendekati 1 artinya nyaris
                     menyentuh TP1 lalu berbalik -> soal volatilitas/target,
                     bukan lokasi entry.

    MAE (Maximum Adverse Excursion) untuk kalibrasi lebar SL:
        mae_r = MAE / risk (|entry - sl|). Untuk trade yang closed via SL,
        loop di update_open_signals() BERHENTI begitu SL kena (break), jadi
        mae_r punya batas bawah ~1.0 (SL memang tersentuh) -- yang informatif
        adalah SEBERAPA JAUH DI ATAS 1.0. mae_r ~1.0-1.1 artinya harga cuma
        sedikit melewati SL saat ini -- indikasi SL sedikit lebih lebar bisa
        menahan (relevan untuk noise crypto). mae_r jauh di atas 1.0 (mis.
        >1.5) artinya harga memang terus bergerak melawan kita -- pelebaran
        SL kemungkinan tidak akan menolong, itu memang pergerakan melawan.
        CATATAN: ini bukan simulasi SL lebih lebar yang presisi (harga
        setelah SL closed tidak lagi dilacak), tapi proxy murah yang cukup
        untuk mengarahkan kalibrasi awal.

    Return dict berisi ringkasan agregat + daftar per-trade (siap diformat
    untuk Telegram di tes_bot._format_diag_stats).
    """
    initialize_database()

    safe_days = max(1, min(int(days), 3650))
    start_time = _report_start_time(safe_days)

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT created_at, symbol, signal, confidence, trend,
                   market_structure, entry, sl, tp1, mfe, mae,
                   metadata_json
            FROM analyses
            WHERE created_at >= ?
              AND signal IN ('BUY', 'SELL')
              AND status = 'CLOSED'
              AND outcome = 'SL'
            ORDER BY created_at ASC
            """,
            (start_time,),
        ).fetchall()

    trades: list[dict[str, Any]] = []
    mfe_r_values: list[float] = []
    mfe_tp1_values: list[float] = []
    mae_r_values: list[float] = []
    overext_values: list[float] = []
    sideways_values: list[float] = []
    fbo_values: list[float] = []
    per_pair: dict[str, int] = {}
    per_bucket: dict[str, int] = {}
    never_moved = 0

    for row in rows:
        entry = _safe_float(row["entry"])
        sl = _safe_float(row["sl"])
        tp1 = _safe_float(row["tp1"])
        mfe = _safe_float(row["mfe"])
        mae = _safe_float(row["mae"])
        confidence = _safe_float(row["confidence"])

        risk = (
            abs(entry - sl)
            if entry is not None and sl is not None
            else None
        )
        tp1_distance = (
            abs(tp1 - entry)
            if tp1 is not None and entry is not None
            else None
        )
        mfe_r = (mfe / risk) if (mfe is not None and risk) else None
        mfe_tp1 = (
            (mfe / tp1_distance)
            if (mfe is not None and tp1_distance)
            else None
        )
        mae_r = (mae / risk) if (mae is not None and risk) else None

        symbol = str(row["symbol"])
        bucket = _confidence_bucket(confidence)
        per_pair[symbol] = per_pair.get(symbol, 0) + 1
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1

        if mfe_r is not None:
            mfe_r_values.append(mfe_r)
            if mfe_r < 0.10:
                never_moved += 1
        if mfe_tp1 is not None:
            mfe_tp1_values.append(mfe_tp1)
        if mae_r is not None:
            mae_r_values.append(mae_r)

        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        overext = _safe_float(meta.get("overextension_ratio_pct"))
        sideways = _safe_float(meta.get("sideways_ratio_pct"))
        fbo = _safe_float(meta.get("false_breakout_against_pct"))
        if overext is not None:
            overext_values.append(overext)
        if sideways is not None:
            sideways_values.append(sideways)
        if fbo is not None:
            fbo_values.append(fbo)

        trades.append(
            {
                "created_at": row["created_at"],
                "symbol": symbol,
                "signal": str(row["signal"]),
                "confidence": confidence,
                "trend": row["trend"],
                "mfe_r": mfe_r,
                "mfe_tp1": mfe_tp1,
                "mae_r": mae_r,
                "overextension_ratio_pct": overext,
            }
        )

    def _avg(values: list[float]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    total = len(trades)
    return {
        "days": safe_days,
        "count": total,
        "trades": trades,
        "per_pair": per_pair,
        "per_bucket": per_bucket,
        "avg_mfe_r": _avg(mfe_r_values),
        "avg_mfe_tp1": _avg(mfe_tp1_values),
        "avg_mae_r": _avg(mae_r_values),
        "max_mae_r": (max(mae_r_values) if mae_r_values else None),
        "never_moved": never_moved,
        "never_moved_pct": (never_moved / total * 100) if total else 0.0,
        "avg_overextension_pct": _avg(overext_values),
        "overextension_sample": len(overext_values),
        "avg_sideways_ratio_pct": _avg(sideways_values),
        "avg_false_breakout_against_pct": _avg(fbo_values),
        "database_path": str(get_database_path()),
    }


def get_direction_census(days: int = 30) -> dict[str, Any]:
    """
    Sensus arah BUY vs SELL (read-only) untuk cek bias arah bot.

    Menjawab pertanyaan: apakah bot benar-benar pernah mengeluarkan SELL,
    atau praktis long-only? Menghitung total sinyal per arah (semua status),
    pecahan closed, menang realized per arah, dan SL-langsung per arah.
    Tidak mengubah logika trading apa pun.

    "Menang" konsisten dengan get_performance_summary: outcome_price di sisi
    profit relatif entry (BUY: > entry, SELL: < entry). Breakeven (== entry)
    tidak dihitung menang.
    """
    initialize_database()

    safe_days = max(1, min(int(days), 3650))
    start_time = _report_start_time(safe_days)

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END)
                    AS buy_total,
                SUM(CASE WHEN signal = 'SELL' THEN 1 ELSE 0 END)
                    AS sell_total,
                SUM(CASE WHEN signal = 'HOLD' THEN 1 ELSE 0 END)
                    AS holds,
                SUM(CASE WHEN signal = 'BUY' AND status = 'OPEN'
                        THEN 1 ELSE 0 END) AS buy_open,
                SUM(CASE WHEN signal = 'SELL' AND status = 'OPEN'
                        THEN 1 ELSE 0 END) AS sell_open,
                SUM(CASE WHEN signal = 'BUY' AND status = 'CLOSED'
                        THEN 1 ELSE 0 END) AS buy_closed,
                SUM(CASE WHEN signal = 'SELL' AND status = 'CLOSED'
                        THEN 1 ELSE 0 END) AS sell_closed,
                SUM(
                    CASE
                        WHEN signal = 'BUY' AND status = 'CLOSED'
                         AND entry IS NOT NULL AND outcome_price IS NOT NULL
                         AND outcome_price > entry
                        THEN 1 ELSE 0
                    END
                ) AS buy_wins,
                SUM(
                    CASE
                        WHEN signal = 'SELL' AND status = 'CLOSED'
                         AND entry IS NOT NULL AND outcome_price IS NOT NULL
                         AND outcome_price < entry
                        THEN 1 ELSE 0
                    END
                ) AS sell_wins,
                SUM(CASE WHEN signal = 'BUY' AND outcome = 'SL'
                        THEN 1 ELSE 0 END) AS buy_direct_sl,
                SUM(CASE WHEN signal = 'SELL' AND outcome = 'SL'
                        THEN 1 ELSE 0 END) AS sell_direct_sl
            FROM analyses
            WHERE created_at >= ?
            """,
            (start_time,),
        ).fetchone()

    census = dict(row) if row is not None else {}
    for key in list(census.keys()):
        census[key] = int(census[key] or 0)
    census["days"] = safe_days
    return census


# =============================================================================
# Pelanggan alert live (untuk live scanner di tes_bot.py)
# =============================================================================
def add_alert_subscriber(chat_id: int) -> bool:
    """
    Mendaftarkan sebuah chat untuk menerima alert sinyal live.

    Return True jika baru ditambahkan, False jika sudah terdaftar.
    """
    initialize_database()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO alert_subscribers (chat_id, created_at)
            VALUES (?, ?)
            """,
            (int(chat_id), _to_iso(_utc_now())),
        )
        return cursor.rowcount == 1


def remove_alert_subscriber(chat_id: int) -> bool:
    """Menghapus pendaftaran alert live. Return True jika ada yang dihapus."""
    initialize_database()
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM alert_subscribers WHERE chat_id = ?",
            (int(chat_id),),
        )
        return cursor.rowcount > 0


def get_alert_subscribers() -> list[int]:
    """Mengembalikan seluruh chat_id yang berlangganan alert live."""
    initialize_database()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT chat_id FROM alert_subscribers ORDER BY created_at ASC"
        ).fetchall()
    return [int(row["chat_id"]) for row in rows]
