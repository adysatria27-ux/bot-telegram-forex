import os
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

# Konfigurasi API (Menggunakan API Key dan Token Anda)
API_KEY = "1551539deae5472f80e506c8a76b0aed"
TOKEN = "8866350485:AAE9aI9eUqFm1YynbVy2UfTLHYt_gPCDZFM"

def get_market_data(symbol):
    """Mengambil data harga dan RSI pada timeframe 5 menit untuk scalping."""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=1&apikey={API_KEY}"
    rsi_url = f"https://api.twelvedata.com/rsi?symbol={symbol}&interval=5min&time_period=14&apikey={API_KEY}"
    try:
        price_res = requests.get(url).json()
        rsi_res = requests.get(rsi_url).json()
        
        price = float(price_res['values'][0]['close'])
        rsi = float(rsi_res['values'][0]['rsi'])
        return price, rsi
    except Exception:
        return None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("⚡ Scalping XAU/USD (5m)", callback_data='signal_xau')]]
    await update.message.reply_text("🤖 Scalper Bot Aktif. Tekan tombol untuk sinyal:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'signal_xau':
        price, rsi = get_market_data("XAU/USD")
        if price:
            # Logika Sinyal Scalping agresif
            trend = "NEUTRAL"
            if rsi < 35: trend = "🟢 BUY (Oversold)"
            elif rsi > 65: trend = "🔴 SELL (Overbought)"
            
            # SL/TP ketat untuk Scalping
            sl = price - 2.0 if "BUY" in trend else price + 2.0
            tp = price + 3.0 if "BUY" in trend else price - 3.0
            
            msg = (f"⚡ **SCALPING ANALYTICS**\n"
                   f"Symbol: XAU/USD\n"
                   f"Price: ${price:.2f} | RSI: {rsi:.2f}\n"
                   f"Signal: {trend}\n\n"
                   f"🎯 TP: ${tp:.2f}\n🛡 SL: ${sl:.2f}\n\n"
                   f"💡 *Entry cepat, amankan profit!*")
            await query.edit_message_text(text=msg, parse_mode='Markdown')
        else:
            await query.edit_message_text(text="❌ Gagal ambil data. Coba lagi nanti.")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.run_polling()
