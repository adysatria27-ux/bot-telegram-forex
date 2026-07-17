import os
import requests
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

# Konfigurasi langsung
API_KEY = "1551539deae5472f80e506c8a76b0aed"
TOKEN = "8866350485:AAE9aI9eUqFm1YynbVy2UfTLHYt_gPCDZFM"

def get_live_price(symbol):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}"
    try:
        response = requests.get(url).json()
        if 'message' in response:
            return f"Server: {response['message']}"
        return response.get('price', 'Data tidak ditemukan')
    except Exception as e:
        return f"Error: {str(e)}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Cek Harga XAUUSD", callback_data='xauusd')],
        [InlineKeyboardButton("📊 Cek Harga EURUSD", callback_data='eurusd')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🚀 Bot SignalForex Aktif!", reply_markup=reply_markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'xauusd':
        harga = get_live_price("XAU/USD")
        await query.edit_message_text(text=f"Harga XAUUSD saat ini: {harga}")
    elif query.data == 'eurusd':
        harga = get_live_price("EUR/USD")
        await query.edit_message_text(text=f"Harga EURUSD saat ini: {harga}")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.run_polling()
