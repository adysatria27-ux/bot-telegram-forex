import os
import math
import time
import logging

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

# Pengguna yang sama perlu menunggu sebelum meminta symbol yang sama lagi.
USER_ANALYSIS_COOLDOWN_SECONDS = 15


# =============================================================================
# 2. DAFTAR INSTRUMEN
# =============================================================================
SYMBOL_MENU = {
    "Cek Harga XAUUSD": {
        "symbol": "XAU/USD",
        "callback_data": "analyze_xauusd",
    },
    "Cek Harga EURUSD": {
        "symbol": "EUR/USD",
        "callback_data": "analyze_eurusd",
    },
}

CALLBACK_SYMBOLS = {
    item["callback_data"]: item["symbol"]
    for item in SYMBOL_MENU.values()
}


# =============================================================================
# 3. WRAPPER KOMPATIBILITAS
# =============================================================================
async def get_market_data(symbol: str = "XAU/USD") -> dict:
    """
    Mempertahankan nama fungsi lama.

    Seluruh pengambilan data dan analisis sekarang dikerjakan oleh
    xauusd_analysis.py.
    """
    return await get_multi_timeframe_market_data(symbol)


def generate_signal_message(data: dict) -> str:
    """
    Mempertahankan nama fungsi lama.

    Seluruh format pesan analisis sekarang berasal dari xauusd_analysis.py.
    """
    return generate_multi_timeframe_signal_message(data)


# =============================================================================
# 4. HANDLER MENU TELEGRAM
# =============================================================================
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Menampilkan menu instrumen dengan alur yang sama seperti versi lama.
    """
    if update.message is None:
        return

    keyboard = [
        ["Cek Harga XAUUSD"],
        ["Cek Harga EURUSD"],
    ]

    await update.message.reply_text(
        "Selamat datang! Pilih instrumen untuk dianalisis:",
        reply_markup=ReplyKeyboardMarkup(
            keyboard,
            resize_keyboard=True,
        ),
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Menampilkan tombol analisis sesuai instrumen yang dipilih.
    """
    if update.message is None or update.message.text is None:
        return

    text = update.message.text.strip()
    instrument = SYMBOL_MENU.get(text)

    if instrument is None:
        return

    symbol = instrument["symbol"]
    callback_data = instrument["callback_data"]

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"📊 Analisa Pro {symbol}",
                    callback_data=callback_data,
                )
            ]
        ]
    )

    await update.message.reply_text(
        f"Tekan tombol di bawah untuk melihat analisis teknikal {symbol}:",
        reply_markup=keyboard,
    )


# =============================================================================
# 5. HANDLER ANALISIS MULTI-TIMEFRAME
# =============================================================================
async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Menangani tombol XAU/USD dan EUR/USD.
    """
    query = update.callback_query

    if query is None:
        return

    symbol = CALLBACK_SYMBOLS.get(query.data)

    if symbol is None:
        try:
            await query.answer()
        except Exception:
            logger.exception("Gagal menjawab callback Telegram.")
            return

        logger.warning(
            "Callback tidak dikenal | callback=%s",
            query.data,
        )
        try:
            await query.edit_message_text(
                "❌ Instrumen tidak dikenali. "
                "Silakan ketik /start dan pilih kembali."
            )
        except Exception:
            logger.exception(
                "Gagal mengirim pesan callback tidak dikenal."
            )
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
                    "Gagal mengirim pemberitahuan cooldown | symbol=%s",
                    symbol,
                )
            return

    # Waktu dicatat sebelum operasi jaringan agar klik ganda ikut tertahan.
    context.user_data[cooldown_key] = current_time

    try:
        await query.answer()
    except Exception:
        logger.exception("Gagal menjawab callback Telegram.")
        return

    if not API_KEY:
        logger.error(
            "TWELVE_DATA_API_KEY belum diatur di Railway."
        )
        try:
            await query.edit_message_text(
                "❌ TWELVE_DATA_API_KEY belum diatur di Railway."
            )
        except Exception:
            logger.exception(
                "Gagal mengirim pesan konfigurasi API key."
            )
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
        message = generate_signal_message(analysis)

        logger.info(
            "Hasil analisis siap | symbol=%s | signal=%s | "
            "cache_hit=%s | cache_age=%ss",
            symbol,
            analysis.get("signal"),
            analysis.get("cache_hit", False),
            analysis.get("cache_age_seconds", 0.0),
        )

        await query.edit_message_text(
            message,
            parse_mode="Markdown",
        )

    except RuntimeError as exc:
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
                "Gagal mengirim pesan RuntimeError | symbol=%s",
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
                "Gagal mengirim pesan error | symbol=%s",
                symbol,
            )


# =============================================================================
# 6. GLOBAL ERROR HANDLER
# =============================================================================
async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Mencatat error Telegram yang tidak tertangani.
    """
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

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    app.add_handler(
        CallbackQueryHandler(
            button,
            pattern=r"^analyze_(xauusd|eurusd)$",
        )
    )

    app.add_error_handler(error_handler)

    app.run_polling(
        drop_pending_updates=True,
    )
