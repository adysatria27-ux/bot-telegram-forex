import os
import logging
import aiohttp
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. PEMBERSIHAN TOKEN (Mencegah error InvalidURL) ---
# .strip() membuang spasi atau baris baru tersembunyi
TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

# --- 3. LOGIKA ANALISIS TEKNIKAL ---
async def fetch_and_analyze():
    if not API_KEY:
        return "❌ Error: API Key tidak ditemukan."
    
    async with aiohttp.ClientSession() as session:
        params = {"symbol": "XAU/USD", "interval": "15min", "outputsize": 50, "apikey": API_KEY}
        async with session.get("https://api.twelvedata.com/time_series", params=params) as resp:
            data = await resp.json()
            if "values" not in data:
                return "❌ Gagal mengambil data pasar."
            
            df = pd.DataFrame(data["values"])
            df[["close", "high", "low"]] = df[["close", "high", "low"]].astype(float)
            
            # Perhitungan indikator sederhana
            price = df["close"].iloc[-1]
            sma = df["close"].rolling(20).mean().iloc[-1]
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            
            signal = "BUY" if price > sma else "SELL"
            sl = price - (atr * 1.5) if signal == "BUY" else price + (atr * 1.5)
            tp = price + (atr * 2.5) if signal == "BUY" else price - (atr * 2.5)
            
            return (f"📊 *Analisis XAU/USD (15m)*\n\n"
                    f"Harga: ${price:.2f}\n"
                    f"Sinyal: *{signal}*\n"
                    f"SMA20: ${sma:.2f}\n"
                    f"SL: ${sl:.2f}\n"
                    f"TP: ${tp:.2f}\n"
                    f"ATR: {atr:.2f}")

# --- 4. HANDLER TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("📊 Analisa Pro", callback_data="analyze_xauusd")]]
    await update.message.reply_text("Siap menganalisa XAU/USD:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "analyze_xauusd":
        await query.edit_message_text("⏳ Menghitung teknikal...")
        msg = await fetch_and_analyze()
        await query.edit_message_text(msg, parse_mode="Markdown")

# --- 5. MAIN RUNNER ---
if __name__ == '__main__':
    if not TOKEN:
        raise ValueError("BOT_TOKEN hilang! Pastikan di-set di Railway Variables.")
    
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_xauusd$"))
    
    app.run_polling()
