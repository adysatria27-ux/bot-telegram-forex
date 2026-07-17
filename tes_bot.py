import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# --- 1. KONFIGURASI ---
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

# --- 2. LOGIKA ANALISIS TEKNIKAL ---
async def get_market_data(symbol="XAU/USD"):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&outputsize=50&apikey={API_KEY}"
        async with session.get(url) as resp:
            data = await resp.json()
            if "values" not in data: 
                raise Exception("Data tidak tersedia")
            
            df = pd.DataFrame(data["values"])
            df[["close", "high", "low"]] = df[["close", "high", "low"]].astype(float)
            
            price = df["close"].iloc[-1]
            sma = df["close"].rolling(20).mean().iloc[-1]
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            signal = "BUY" if price > sma else "SELL"
            
            return {"symbol": symbol, "price": price, "signal": signal, "sma": sma, "atr": atr}

def generate_signal_message(data):
    return (f"📊 *Analisis Lengkap {data['symbol']}*\n\n"
            f"Harga Saat Ini: ${data['price']:.2f}\n"
            f"Sinyal Trading: *{data['signal']}*\n"
            f"SMA 20: ${data['sma']:.2f}\n"
            f"ATR (Volatilitas): {data['atr']:.2f}\n\n"
            f"_Data diambil dari Twelve Data API_")

# --- 3. HANDLER MENU & INTERAKSI ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [["Cek Harga XAUUSD"], ["Cek Harga EURUSD"]]
    await update.message.reply_text("Selamat datang! Pilih instrumen untuk dianalisis:", 
                                    reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text in ["Cek Harga XAUUSD", "Cek Harga EURUSD"]:
        sym = "XAU/USD" if "XAU" in text else "EUR/USD"
        cb = "analyze_xauusd" if "XAU" in text else "analyze_eurusd"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(f"📊 Analisa Pro {sym}", callback_data=cb)]])
        await update.message.reply_text(f"Tekan tombol di bawah untuk melihat analisis teknikal {sym}:", 
                                        reply_markup=kb)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    sym = "XAU/USD" if query.data == "analyze_xauusd" else "EUR/USD"
    await query.edit_message_text(f"⏳ Sedang menghitung analisis {sym}...")
    
    try:
        data = await get_market_data(sym)
        await query.edit_message_text(generate_signal_message(data), parse_mode="Markdown")
    except Exception as e:
        await query.edit_message_text(f"❌ Gagal mengambil data: {str(e)}")

# --- 4. MAIN RUNNER (DENGAN PEMBERSIH WEBHOOK) ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Memastikan tidak ada konflik sesi lama
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(app.bot.delete_webhook(drop_pending_updates=True))
    except:
        pass

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_"))
    
    app.run_polling()
