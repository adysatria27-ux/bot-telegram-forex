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

TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# --- LOGIKA TEKNIKAL PRO ---
def calculate_metrics(df):
    df[["close", "high", "low"]] = df[["close", "high", "low"]].astype(float)
    close = df["close"]
    
    # Indikator
    sma = close.rolling(20).mean().iloc[-1]
    rsi_delta = close.diff()
    rsi = 100 - (100 / (1 + (rsi_delta.clip(lower=0).ewm(alpha=1/14).mean() / -rsi_delta.clip(upper=0).ewm(alpha=1/14).mean())))
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    
    return close.iloc[-1], sma, rsi.iloc[-1], atr

async def fetch_analysis():
    if not API_KEY: return "❌ Error: API Key tidak ada."
    
    async with aiohttp.ClientSession() as session:
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=15min&outputsize=50&apikey={API_KEY}"
        async with session.get(url) as resp:
            data = await resp.json()
            if "values" not in data: return "❌ Gagal ambil data."
            
            df = pd.DataFrame(data["values"])
            price, sma, rsi, atr = calculate_metrics(df)
            
            # Sinyal & Manajemen Risiko
            signal = "BUY" if price > sma and rsi < 60 else "SELL" if price < sma and rsi > 40 else "NEUTRAL"
            sl = price - (atr * 1.5) if signal == "BUY" else price + (atr * 1.5)
            tp = price + (atr * 2.5) if signal == "BUY" else price - (atr * 2.5)
            
            return (f"📊 *Analisis XAU/USD (15m)*\n\n"
                    f"Harga: ${price:.2f}\n"
                    f"Sinyal: *{signal}*\n"
                    f"RSI: {rsi:.2f} | SMA20: ${sma:.2f}\n"
                    f"SL: ${sl:.2f}\n"
                    f"TP: ${tp:.2f}\n"
                    f"ATR: {atr:.2f}\n\n"
                    f"⚠️ _Gunakan manajemen risiko sendiri._")

# --- HANDLER ---
async def start(update, context):
    kb = [[InlineKeyboardButton("📊 Analisa Pro", callback_data="analisa")]]
    await update.message.reply_text("Siap menganalisis XAU/USD:", reply_markup=InlineKeyboardMarkup(kb))

async def button(update, context):
    query = update.callback_query
    await query.answer()
    if query.data == "analisa":
        await query.edit_message_text("⏳ Menghitung teknikal...")
        msg = await fetch_analysis()
        await query.edit_message_text(msg, parse_mode="Markdown")

if __name__ == '__main__':
    if not TOKEN: raise ValueError("BOT_TOKEN tidak ditemukan!")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))
    app.run_polling()
