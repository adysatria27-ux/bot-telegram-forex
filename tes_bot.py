import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# --- SETUP & KONFIGURASI ---
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

# [DI SINI: Masukkan seluruh logika get_market_data dan generate_signal_message 
# yang panjang dari kode sebelumnya]
# Pastikan semua fungsi tersebut ada di atas fungsi button() agar terbaca.

# --- HANDLER TOMBOL ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Cek Harga XAUUSD"], ["Cek Harga EURUSD"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Silakan pilih menu:", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "Cek Harga XAUUSD":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 Analisa Pro XAU/USD", callback_data="analyze_xauusd")]])
        await update.message.reply_text("Klik tombol di bawah untuk melihat analisis lengkap:", reply_markup=kb)
    elif text == "Cek Harga EURUSD":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 Analisa Pro EUR/USD", callback_data="analyze_eurusd")]])
        await update.message.reply_text("Klik tombol di bawah untuk melihat analisis lengkap:", reply_markup=kb)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Menentukan simbol berdasarkan tombol yang diklik
    symbol = "XAU/USD" if query.data == "analyze_xauusd" else "EUR/USD"
    
    await query.edit_message_text(f"⏳ Mengambil data {symbol}...")
    
    try:
        # Menjalankan analisis dan menampilkan hasil lengkap
        analysis = await get_market_data(symbol)
        msg = generate_signal_message(analysis)
        await query.edit_message_text(msg, parse_mode="Markdown")
    except Exception as e:
        await query.edit_message_text(f"❌ Gagal mengambil data {symbol}. Coba lagi nanti.")

# --- MAIN RUNNER ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_"))
    app.run_polling()
