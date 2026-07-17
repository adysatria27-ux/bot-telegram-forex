import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- KONFIGURASI ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mengambil variabel dari Railway (Sudah sesuai dengan dashboard Anda)
TOKEN = os.getenv("BOT_TOKEN")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"

# --- LOGIKA TEKNIKAL (Integrasi dari kode Pro) ---
def calculate_indicators(df):
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    
    # Indikator (RSI, SMA, ATR)
    df["sma"] = df["close"].rolling(20).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df["rsi"] = 100 - (100 / (1 + (gain/loss.replace(0, np.nan))))
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    
    return df.iloc[-1]

async def get_analysis_text():
    async with aiohttp.ClientSession() as session:
        params = {"symbol": "XAU/USD", "interval": "15min", "outputsize": 50, "apikey": TWELVE_DATA_API_KEY}
        async with session.get(BASE_URL, params=params) as resp:
            data = await resp.json()
            if "values" not in data: return "❌ Data tidak tersedia."
            
            df = pd.DataFrame(data["values"])
            res = calculate_indicators(df)
            
            signal = "BUY" if res.close > res.sma else "SELL"
            sl = res.close - (res.atr * 1.5) if signal == "BUY" else res.close + (res.atr * 1.5)
            
            return (f"📊 *Analisis XAU/USD (15m)*\n\n"
                    f"Harga: ${res.close:.2f}\n"
                    f"Sinyal: *{signal}*\n"
                    f"RSI: {res.rsi:.2f}\n"
                    f"SMA20: ${res.sma:.2f}\n"
                    f"SL: ${sl:.2f}\n"
                    f"ATR: {res.atr:.2f}")

# --- HANDLER TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("📊 Analisa Pro", callback_data="analyze_xauusd")]]
    await update.message.reply_text("Klik tombol untuk analisa teknikal:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "analyze_xauusd":
        await query.edit_message_text("⏳ Menghitung teknikal...")
        msg = await get_analysis_text()
        await query.edit_message_text(msg, parse_mode="Markdown")

# --- MAIN RUNNER ---
if __name__ == '__main__':
    if not TOKEN: raise ValueError("BOT_TOKEN hilang di Railway!")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_xauusd$"))
    print("Bot berhasil jalan...")
    app.run_polling()
