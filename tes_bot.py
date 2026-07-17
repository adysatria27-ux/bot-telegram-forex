import os
import requests
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

# Mengambil API Key dari Variable Railway
API_KEY = os.getenv('TWELVE_API_KEY')

def get_live_price(symbol):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}"
    try:
        response = requests.get(url).json()
        return response.get('price', 'N/A')
    except:
        return "Error"

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
        await query.edit_message_text(text=f"Harga XAUUSD saat ini: ${harga}")
    elif query.data == 'eurusd':
        harga = get_live_price("EUR/USD")
        await query.edit_message_text(text=f"Harga EURUSD saat ini: ${harga}")

if __name__ == '__main__':
    TOKEN = os.getenv('BOT_TOKEN')
    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.run_polling()
