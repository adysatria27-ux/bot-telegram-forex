import os
import math
import time
import logging
import asyncio

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from xauusd_analysis import (
    get_market_data as get_multi_timeframe_market_data,
    generate_signal_message as generate_multi_timeframe_signal_message,
    get_market_candles,
    get_supported_assets,
    get_asset_config,
    resolve_symbol_from_menu_text,
    resolve_symbol_from_callback,
)

from signal_tracker import (
    initialize_database,
    record_analysis,
    update_open_signals,
    get_open_symbols,
    get_recent_signals,
    get_performance_summary,
)


# =============================================================================
# 1. KONFIGURASI
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

USER_ANALYSIS_COOLDOWN_SECONDS = 15
MENU_COLUMNS = 2

TRACKER_INTERVAL = "5min"
TRACKER_STATS_DAYS = 30
TRACKER_RECENT_LIMIT = 10
MAX_TRACKER_REFRESH_SYMBOLS = 8


# =============================================================================
# 2. MENU DINAMIS DARI GENERIC ENGINE
# =============================================================================
def _build_asset_keyboard() -> ReplyKeyboardMarkup:
    """Membuat menu dari satu registry aset di xauusd_analysis.py."""
    labels = [asset.menu_label for asset in get_supported_assets()]
    rows = [
        labels[index:index + MENU_COLUMNS]
        for index in range(0, len(labels), MENU_COLUMNS)
    ]
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
    )


# =============================================================================
# 3. WRAPPER KOMPATIBILITAS
# =============================================================================
async def get_market_data(symbol: str = "XAU/USD") -> dict:
    """Nama fungsi lama tetap tersedia untuk integrasi eksternal."""
    return await get_multi_timeframe_market_data(symbol)


def generate_signal_message(data: dict) -> str:
    """Nama formatter lama tetap tersedia."""
    return generate_multi_timeframe_signal_message(data)


# =============================================================================
# 4. HANDLER MENU TELEGRAM
# =============================================================================
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Menampilkan seluruh aset yang didukung secara dinamis."""
    if update.message is None:
        return

    await update.message.reply_text(
        "Selamat datang! Pilih instrumen untuk dianalisis:",
        reply_markup=_build_asset_keyboard(),
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Menampilkan tombol analisis untuk aset yang dipilih."""
    if update.message is None or update.message.text is None:
        return

    symbol = resolve_symbol_from_menu_text(update.message.text)
    if symbol is None:
        return

    asset = get_asset_config(symbol)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"📊 Analisa Pro {asset.symbol}",
                    callback_data=asset.callback_data,
                )
            ]
        ]
    )

    await update.message.reply_text(
        f"Tekan tombol di bawah untuk melihat analisis teknikal {asset.symbol}:",
        reply_markup=keyboard,
    )



# =============================================================================
# 4. SIGNAL LOGGER DAN OUTCOME TRACKER
# =============================================================================
async def _refresh_tracker_symbol(symbol: str) -> dict[str, int]:
    """
    Memperbarui signal lama dengan candle terbaru.

    Fungsi get_market_candles() memakai cache engine yang sama, sehingga
    pemanggilan setelah analisis biasanya tidak menambah request API.
    """
    candles = await get_market_candles(
        symbol,
        TRACKER_INTERVAL,
    )
    return await asyncio.to_thread(
        update_open_signals,
        symbol,
        candles,
    )


async def _track_analysis(
    symbol: str,
    analysis: dict,
) -> None:
    """
    Memperbarui outcome lama lalu mencatat analisis baru.

    Kegagalan tracker tidak boleh menggagalkan pesan analisis Telegram.
    """
    try:
        tracker_update = await _refresh_tracker_symbol(symbol)
        log_result = await asyncio.to_thread(
            record_analysis,
            analysis,
        )

        logger.info(
            "Signal tracker selesai | symbol=%s | analysis_id=%s | "
            "inserted=%s | checked=%s | closed=%s",
            symbol,
            log_result.get("id"),
            log_result.get("inserted"),
            tracker_update.get("checked", 0),
            tracker_update.get("closed", 0),
        )
    except Exception:
        logger.exception(
            "Signal tracker gagal, analisis Telegram tetap dilanjutkan | "
            "symbol=%s",
            symbol,
        )


async def _refresh_all_open_signals() -> None:
    """Memperbarui outcome beberapa symbol OPEN secara hemat API."""
    symbols = await asyncio.to_thread(get_open_symbols)

    for symbol in symbols[:MAX_TRACKER_REFRESH_SYMBOLS]:
        try:
            await _refresh_tracker_symbol(symbol)
        except Exception:
            logger.exception(
                "Gagal refresh outcome | symbol=%s",
                symbol,
            )


def _format_recent_signals(
    rows: list[dict],
) -> str:
    """Membuat daftar signal terbaru dalam teks Telegram."""
    if not rows:
        return (
            "📭 Belum ada histori analisis.\n"
            "Jalankan analisis aset terlebih dahulu."
        )

    lines = ["📋 Histori Signal Terbaru", ""]

    for row in rows:
        signal = row.get("signal", "HOLD")
        symbol = row.get("symbol", "-")
        confidence = float(row.get("confidence") or 0.0)
        outcome = row.get("outcome", "-")
        max_tp_hit = int(row.get("max_tp_hit") or 0)

        if signal == "BUY":
            icon = "🟢"
        elif signal == "SELL":
            icon = "🔴"
        else:
            icon = "🟡"

        tp_text = (
            f" | TP max: {max_tp_hit}"
            if max_tp_hit > 0
            else ""
        )
        lines.append(
            f"{icon} #{row.get('id')} {symbol} {signal} "
            f"| Conf {confidence:.1f}% | {outcome}{tp_text}"
        )

    return "\n".join(lines)


def _format_tracker_stats(
    summary: dict,
) -> str:
    """Membuat statistik outcome tracker."""
    completed = int(summary.get("completed_trades") or 0)
    open_trades = int(summary.get("open_trades") or 0)
    trade_signals = int(summary.get("trade_signals") or 0)
    holds = int(summary.get("holds") or 0)
    tp1_or_better = int(summary.get("tp1_or_better") or 0)
    tp2_or_better = int(summary.get("tp2_or_better") or 0)
    tp3_hits = int(summary.get("tp3_hits") or 0)
    direct_sl = int(summary.get("direct_sl") or 0)
    sl_after_tp = int(summary.get("sl_after_tp") or 0)
    expired = int(summary.get("expired") or 0)
    average_confidence = float(
        summary.get("average_trade_confidence") or 0.0
    )

    lines = [
        f"📈 Statistik Signal — {summary.get('days', 30)} Hari",
        "",
        f"Total analisis: {int(summary.get('total_analyses') or 0)}",
        f"HOLD: {holds}",
        f"BUY/SELL: {trade_signals}",
        f"Masih OPEN: {open_trades}",
        f"Selesai: {completed}",
        "",
        f"TP1 atau lebih: {tp1_or_better}",
        f"TP2 atau lebih: {tp2_or_better}",
        f"TP3: {tp3_hits}",
        f"SL langsung: {direct_sl}",
        f"SL setelah TP: {sl_after_tp}",
        f"Expired: {expired}",
        f"TP1 hit rate: {float(summary.get('tp1_hit_rate_pct') or 0.0):.1f}%",
        f"Rata-rata confidence trade: {average_confidence:.1f}%",
    ]

    buckets = summary.get("confidence_buckets") or []
    if buckets:
        lines.extend(["", "Confidence Buckets:"])
        for bucket in buckets:
            signals = int(bucket.get("signals") or 0)
            wins = int(bucket.get("tp1_or_better") or 0)
            rate = (wins / signals * 100) if signals else 0.0
            lines.append(
                f"• {bucket.get('bucket')}: {wins}/{signals} "
                f"mencapai TP1+ ({rate:.1f}%)"
            )

    lines.extend(
        [
            "",
            "Catatan: TP dan SL pada candle yang sama dihitung SL dahulu.",
        ]
    )
    return "\n".join(lines)


async def signals_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Perintah /signals untuk melihat histori terbaru."""
    if update.message is None:
        return

    try:
        rows = await asyncio.to_thread(
            get_recent_signals,
            TRACKER_RECENT_LIMIT,
        )
        await update.message.reply_text(
            _format_recent_signals(rows)
        )
    except Exception:
        logger.exception("Gagal membaca histori signal.")
        await update.message.reply_text(
            "❌ Gagal membaca histori signal."
        )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Perintah /stats untuk memperbarui dan menampilkan statistik."""
    if update.message is None:
        return

    try:
        await update.message.reply_text(
            "⏳ Memperbarui outcome signal yang masih aktif..."
        )
        await _refresh_all_open_signals()

        summary = await asyncio.to_thread(
            get_performance_summary,
            TRACKER_STATS_DAYS,
        )
        await update.message.reply_text(
            _format_tracker_stats(summary)
        )
    except Exception:
        logger.exception("Gagal membuat statistik signal.")
        await update.message.reply_text(
            "❌ Gagal membuat statistik signal."
        )

# =============================================================================
# 5. HANDLER ANALISIS GENERIC
# =============================================================================
async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Menangani seluruh callback aset menggunakan satu pipeline."""
    query = update.callback_query
    if query is None:
        return

    symbol = resolve_symbol_from_callback(query.data)
    if symbol is None:
        try:
            await query.answer()
        except Exception:
            logger.exception("Gagal menjawab callback Telegram.")
            return

        logger.warning("Callback tidak dikenal | callback=%s", query.data)
        try:
            await query.edit_message_text(
                "❌ Instrumen tidak dikenali. Silakan ketik /start dan pilih kembali."
            )
        except Exception:
            logger.exception("Gagal mengirim pesan callback tidak dikenal.")
        return

    cooldown_key = f"last_analysis_request:{symbol}"
    current_time = time.monotonic()
    last_request_time = context.user_data.get(cooldown_key)

    if isinstance(last_request_time, (int, float)):
        elapsed = current_time - last_request_time
        remaining = USER_ANALYSIS_COOLDOWN_SECONDS - elapsed
        if remaining > 0:
            try:
                await query.answer(
                    text=(
                        "Mohon tunggu "
                        f"{math.ceil(remaining)} detik sebelum analisis ulang."
                    ),
                    show_alert=True,
                )
            except Exception:
                logger.exception(
                    "Gagal mengirim cooldown | symbol=%s",
                    symbol,
                )
            return

    context.user_data[cooldown_key] = current_time

    try:
        await query.answer()
    except Exception:
        logger.exception("Gagal menjawab callback Telegram.")
        return

    if not API_KEY:
        logger.error("TWELVE_DATA_API_KEY belum diatur di Railway.")
        try:
            await query.edit_message_text(
                "❌ TWELVE_DATA_API_KEY belum diatur di Railway."
            )
        except Exception:
            logger.exception("Gagal mengirim pesan konfigurasi API key.")
        return

    try:
        await query.edit_message_text(
            f"⏳ Mengambil dan menganalisis data 4 timeframe "
            f"(5m/15m/30m/1h) untuk {symbol}..."
        )
    except Exception:
        logger.exception(
            "Gagal menampilkan pesan proses | symbol=%s",
            symbol,
        )
        return

    try:
        analysis = await get_market_data(symbol)
        await _track_analysis(symbol, analysis)
        message = generate_signal_message(analysis)

        logger.info(
            "Hasil analisis siap | symbol=%s | provider=%s | signal=%s | "
            "confidence=%s | cache_hit=%s | cache_age=%ss",
            symbol,
            analysis.get("provider_symbol"),
            analysis.get("signal"),
            analysis.get("confidence_pct"),
            analysis.get("cache_hit", False),
            analysis.get("cache_age_seconds", 0.0),
        )

        await query.edit_message_text(
            message,
            parse_mode="Markdown",
        )

    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Analisis gagal | symbol=%s | error=%s",
            symbol,
            exc,
        )
        try:
            await query.edit_message_text(
                "❌ Gagal mengambil atau menganalisis data pasar.\n\n"
                f"Detail: {exc}\n\n"
                "Silakan coba kembali beberapa saat lagi."
            )
        except Exception:
            logger.exception(
                "Gagal mengirim pesan error | symbol=%s",
                symbol,
            )

    except Exception:
        logger.exception(
            "Error tak terduga saat analisis | symbol=%s",
            symbol,
        )
        try:
            await query.edit_message_text(
                "❌ Terjadi kesalahan tak terduga saat analisis. "
                "Silakan coba kembali beberapa saat lagi."
            )
        except Exception:
            logger.exception(
                "Gagal mengirim pesan error tak terduga | symbol=%s",
                symbol,
            )


# =============================================================================
# 6. GLOBAL ERROR HANDLER
# =============================================================================
async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Mencatat error Telegram yang tidak tertangani."""
    logger.error(
        "Error Telegram tidak tertangani.",
        exc_info=context.error,
    )


# =============================================================================
# 7. MAIN RUNNER
# =============================================================================
if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError(
            "BOT_TOKEN belum diatur pada environment variable."
        )

    if not API_KEY:
        logger.warning(
            "TWELVE_DATA_API_KEY belum diatur. "
            "Bot dapat berjalan, tetapi analisis pasar akan gagal."
        )

    database_path = initialize_database()
    logger.info(
        "Signal tracker aktif | database=%s",
        database_path,
    )

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )
    app.add_handler(
        CallbackQueryHandler(
            button,
            pattern=r"^analyze_[a-z0-9]+$",
        )
    )
    app.add_error_handler(error_handler)

    # Polling tetap sama seperti versi production sebelumnya.
    app.run_polling(
        drop_pending_updates=True,
    )
