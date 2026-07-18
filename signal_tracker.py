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
MAX_RECENT_SIGNAL_LIMIT = 50

_SCHEMA_VERSION = 1
_DATABASE_PATH: Optional[Path] = None


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

            COMMIT;
            """
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


def _signal_expiry_hours(asset_class: str) -> int:
    """Menentukan masa berlaku sinyal berdasarkan kelas aset."""
    if asset_class.lower() == "crypto":
        return CRYPTO_SIGNAL_EXPIRY_HOURS

    configured = os.getenv("SIGNAL_MAX_AGE_HOURS", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            logger.warning(
                "SIGNAL_MAX_AGE_HOURS tidak valid: %s",
                configured,
            )

    return DEFAULT_SIGNAL_EXPIRY_HOURS


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
            hours=_signal_expiry_hours(asset_class)
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
                mfe = max(mfe, favorable)
                mae = max(mae, adverse)
                latest_checked = _to_iso(candle["datetime"])

                # Konservatif: SL diperiksa sebelum TP di candle yang sama.
                if _sl_hit_for_candle(signal, candle, sl):
                    status = "CLOSED"
                    outcome = _closed_outcome_after_sl(max_tp_hit)
                    outcome_at = latest_checked
                    outcome_price = sl
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

            changed = (
                status != row["status"]
                or outcome != row["outcome"]
                or max_tp_hit != int(row["max_tp_hit"] or 0)
                or abs(mfe - float(row["mfe"] or 0.0)) > 1e-12
                or abs(mae - float(row["mae"] or 0.0)) > 1e-12
                or latest_checked != row["latest_checked_candle_time"]
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
                    mae = ?
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


def get_performance_summary(days: int = 30) -> dict[str, Any]:
    """Menghasilkan statistik ringkas untuk kalibrasi awal."""
    initialize_database()

    safe_days = max(1, min(int(days), 3650))
    start_time = _to_iso(
        _utc_now() - timedelta(days=safe_days)
    )

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
                SUM(CASE WHEN outcome LIKE '%EXPIRED%' THEN 1 ELSE 0 END)
                    AS expired,
                AVG(
                    CASE
                        WHEN signal IN ('BUY', 'SELL')
                        THEN confidence
                        ELSE NULL
                    END
                ) AS average_trade_confidence
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
                    AS direct_sl
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
    completed = int(summary.get("completed_trades") or 0)
    tp1_or_better = int(summary.get("tp1_or_better") or 0)

    summary["days"] = safe_days
    summary["tp1_hit_rate_pct"] = (
        round((tp1_or_better / completed) * 100, 1)
        if completed > 0
        else 0.0
    )
    summary["confidence_buckets"] = [
        dict(bucket_row)
        for bucket_row in bucket_rows
    ]
    summary["database_path"] = str(get_database_path())
    return summary
