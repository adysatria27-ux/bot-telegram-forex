import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# --- 1. SETUP ---
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

# --- 2. LOGIKA ANALISIS (FUNGSI UTAMA) ---
async def get_market_data(symbol="XAU/USD"):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&outputsize=50&apikey={API_KEY}"
        async with session.get(url) as resp:
            data = await resp.json()
            if "values" not in data: raise Exception("Data tidak ditemukan")
            df = pd.DataFrame(data["values"])
            df[["close", "high", "low"]] = df[["close", "high", "low"]].astype(float)
            price = df["close"].iloc[-1]
            sma = df["close"].rolling(20).mean().iloc[-1]
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            signal = "BUY" if price > sma else "SELL"
            return {"symbol": symbol, "price": price, "signal": signal, "sma": sma, "atr": atr}

def generate_signal_message(data):
    return (f"📊 *Analisis {data['symbol']}*\n\n"
            f"Harga: ${data['price']:.2f}\n"
            f"Sinyal: *{data['signal']}*\n"
            f"SMA20: ${data['sma']:.2f}\n"
            f"ATR: {data['atr']:.2f}")

# --- 3. HANDLER MENU & TOMBOL ---
async def start(update, context):
    kb = [["Cek Harga XAUUSD"], ["Cek Harga EURUSD"]]
    await update.message.reply_text("Silakan pilih menu:", reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))

async def handle_message(update, context):
    text = update.message.text
    if text in ["Cek Harga XAUUSD", "Cek Harga EURUSD"]:
        sym = "XAU/USD" if "XAU" in text else "EUR/USD"
        cb = "analyze_xauusd" if "XAU" in text else "analyze_eurusd"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(f"📊 Analisa {sym}", callback_data=cb)]])
        await update.message.reply_text(f"Klik tombol untuk analisa {sym}:", reply_markup=kb)

async def button(update, context):
    query = update.callback_query
    await query.answer()
    sym = "XAU/USD" if query.data == "analyze_xauusd" else "EUR/USD"
    await query.edit_message_text(f"⏳ Menganalisis {sym}...")
    try:
        data = await get_market_data(sym)
        await query.edit_message_text(generate_signal_message(data), parse_mode="Markdown")
    except:
        await query.edit_message_text(f"❌ Gagal mengambil data {sym}.")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_"))
    app.run_polling()
