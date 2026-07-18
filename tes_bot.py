import os
import logging
import asyncio
from typing import Any

import aiohttp
import pandas as pd
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

# =============================================================================
# 1. KONFIGURASI
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Nama environment variable tetap dipertahankan.
TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
REQUEST_TIMEOUT_SECONDS = 15
OUTPUT_SIZE = 50
ANALYSIS_INTERVAL = "15min"

SMA_PERIOD = 20
ATR_PERIOD = 14

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
# 2. HELPER DATA DAN FORMAT
# =============================================================================
def _validate_api_response(data: Any) -> list[dict]:
    """
    Memastikan respons Twelve Data memiliki struktur yang dibutuhkan.

    Mengembalikan daftar candle jika valid.
    Melempar RuntimeError dengan pesan yang aman jika respons tidak valid.
    """
    if not isinstance(data, dict):
        raise RuntimeError("Format respons data pasar tidak valid.")

    if data.get("status") == "error":
        api_message = data.get("message", "Twelve Data mengembalikan error.")
        raise RuntimeError(str(api_message))

    values = data.get("values")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Data candle tidak tersedia.")

    return values


def _prepare_dataframe(values: list[dict]) -> pd.DataFrame:
    """
    Mengubah data candle menjadi DataFrame yang bersih dan kronologis.

    Twelve Data umumnya mengirim candle terbaru terlebih dahulu.
    Karena indikator membutuhkan urutan lama -> baru, data wajib diurutkan
    berdasarkan datetime sebelum mengambil baris terakhir.
    """
    df = pd.DataFrame(values)

    required_columns = {"datetime", "open", "high", "low", "close"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"Kolom data pasar tidak lengkap: {missing}.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    numeric_columns = ["open", "high", "low", "close"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["datetime", *numeric_columns])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    minimum_rows = max(SMA_PERIOD, ATR_PERIOD) + 1
    if len(df) < minimum_rows:
        raise RuntimeError(
            f"Data candle tidak cukup. Minimal {minimum_rows} candle diperlukan, "
            f"tetapi hanya tersedia {len(df)}."
        )

    return df


def _calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    Menghitung ATR menggunakan True Range standar.

    True Range adalah nilai maksimum dari:
    1. High - Low
    2. |High - previous close|
    3. |Low - previous close|
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

    return true_range.rolling(window=period, min_periods=period).mean()


def _decimal_places(symbol: str) -> int:
    """
    Menentukan jumlah angka desimal agar hasil mudah dibaca.

    XAU/USD umumnya cukup dua desimal.
    Pair forex seperti EUR/USD membutuhkan lima desimal agar ATR kecil
    tidak terlihat sebagai 0.00.
    """
    if symbol == "XAU/USD":
        return 2
    return 5


# =============================================================================
# 3. LOGIKA ANALISIS TEKNIKAL
# =============================================================================
async def get_market_data(symbol: str = "XAU/USD") -> dict:
    """
    Mengambil candle 15 menit dari Twelve Data dan menghitung:
    - Harga candle terbaru setelah data diurutkan
    - SMA 20
    - ATR 14 berbasis True Range
    - Sinyal BUY atau SELL sesuai logika lama

    Struktur hasil tetap kompatibel dengan generate_signal_message().
    """
    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY belum diatur pada environment variable."
        )

    params = {
        "symbol": symbol,
        "interval": ANALYSIS_INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": API_KEY,
    }

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(TWELVE_DATA_URL, params=params) as response:
                if response.status == 429:
                    raise RuntimeError(
                        "Batas request Twelve Data telah tercapai. "
                        "Silakan coba beberapa saat lagi."
                    )

                if response.status != 200:
                    raise RuntimeError(
                        f"Twelve Data mengembalikan HTTP {response.status}."
                    )

                try:
                    data = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Respons Twelve Data bukan JSON yang valid."
                    ) from exc

    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Permintaan data pasar melewati batas waktu."
        ) from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(
            "Tidak dapat terhubung ke Twelve Data."
        ) from exc

    values = _validate_api_response(data)
    df = _prepare_dataframe(values)

    sma_series = df["close"].rolling(
        window=SMA_PERIOD,
        min_periods=SMA_PERIOD,
    ).mean()
    atr_series = _calculate_atr(df, ATR_PERIOD)

    price = float(df["close"].iloc[-1])
    sma = float(sma_series.iloc[-1])
    atr = float(atr_series.iloc[-1])

    if pd.isna(sma) or pd.isna(atr):
        raise RuntimeError(
            "Indikator belum dapat dihitung karena data tidak cukup."
        )

    # Logika sinyal lama tetap dipertahankan pada Tahap 1.
    signal = "BUY" if price > sma else "SELL"

    logger.info(
        "Analisis selesai | symbol=%s | candle=%s | price=%.5f | "
        "sma=%.5f | atr=%.5f | signal=%s",
        symbol,
        df["datetime"].iloc[-1],
        price,
        sma,
        atr,
        signal,
    )

    return {
        "symbol": symbol,
        "price": price,
        "signal": signal,
        "sma": sma,
        "atr": atr,
    }


def generate_signal_message(data: dict) -> str:
    """
    Membuat pesan Markdown Telegram.

    Susunan pesan lama tetap dipertahankan. Hanya jumlah desimal yang
    disesuaikan menurut instrumen agar EUR/USD tidak tampil sebagai ATR 0.00.
    """
    decimals = _decimal_places(data["symbol"])

    price = f"{data['price']:.{decimals}f}"
    sma = f"{data['sma']:.{decimals}f}"
    atr = f"{data['atr']:.{decimals}f}"

    return (
        f"📊 *Analisis Lengkap {data['symbol']}*\n\n"
        f"Harga Saat Ini: ${price}\n"
        f"Sinyal Trading: *{data['signal']}*\n"
        f"SMA 20: ${sma}\n"
        f"ATR (Volatilitas): {atr}\n\n"
        f"_Data diambil dari Twelve Data API_"
    )


# =============================================================================
# 4. HANDLER MENU DAN INTERAKSI TELEGRAM
# =============================================================================
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Menampilkan menu instrumen.

    Teks dan alur Telegram lama tetap dipertahankan.
    """
    keyboard = [
        ["Cek Harga XAUUSD"],
        ["Cek Harga EURUSD"],
    ]

    if update.message is None:
        return

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
    Menangani pilihan instrumen dari Reply Keyboard.
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


async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Menangani tombol analisis XAU/USD dan EUR/USD.
    """
    query = update.callback_query
    if query is None:
        return

    try:
        await query.answer()
    except Exception:
        logger.exception("Gagal menjawab callback Telegram.")
        return

    symbol = CALLBACK_SYMBOLS.get(query.data)
    if symbol is None:
        logger.warning("Callback tidak dikenal: %s", query.data)
        try:
            await query.edit_message_text(
                "❌ Instrumen tidak dikenali. Silakan ketik /start dan pilih kembali."
            )
        except Exception:
            logger.exception("Gagal mengirim pesan callback tidak dikenal.")
        return

    try:
        await query.edit_message_text(
            f"⏳ Sedang menghitung analisis {symbol}..."
        )
    except Exception:
        logger.exception("Gagal menampilkan pesan proses analisis.")
        return

    try:
        data = await get_market_data(symbol)
        message = generate_signal_message(data)

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
                f"❌ Gagal mengambil data: {exc}"
            )
        except Exception:
            logger.exception("Gagal mengirim pesan error analisis.")

    except Exception:
        logger.exception(
            "Terjadi error tak terduga saat menganalisis %s.",
            symbol,
        )
        try:
            await query.edit_message_text(
                "❌ Terjadi kesalahan tak terduga saat analisis. "
                "Silakan coba lagi beberapa saat."
            )
        except Exception:
            logger.exception("Gagal mengirim pesan error tak terduga.")


# =============================================================================
# 5. MAIN RUNNER
# =============================================================================
if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError(
            "BOT_TOKEN belum diatur pada environment variable."
        )

    app = ApplicationBuilder().token(TOKEN).build()

    # Mekanisme pembersihan webhook lama dipertahankan agar alur deployment
    # tidak berubah pada tahap pertama.
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            app.bot.delete_webhook(drop_pending_updates=True)
        )
    except Exception:
        logger.exception(
            "Pembersihan webhook gagal. Bot tetap mencoba menjalankan polling."
        )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )
    app.add_handler(
        CallbackQueryHandler(
            button,
            pattern="^analyze_",
        )
    )

    app.run_polling()
