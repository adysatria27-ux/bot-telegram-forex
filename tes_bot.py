import os
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

# Konfigurasi Akses
API_KEY = "1551539deae5472f80e506c8a76b0aed"
TOKEN = "8866350485:AAE9aI9eUqFm1YynbVy2UfTLHYt_gPCDZFM"

def get_market_data(symbol, interval):
    """Mengambil data teknikal: Harga, RSI 14, dan SMA 20."""
    base_url = "https://api.twelvedata.com"
    try:
        # Panggilan API
        p_res = requests.get(f"{base_url}/price?symbol={symbol}&apikey={API_KEY}").json()
        r_res = requests.get(f"{base_url}/rsi?symbol={symbol}&interval={interval}&time_period=14&apikey={API_KEY}").json()
        s_res = requests.get(f"{base_url}/sma?symbol={symbol}&interval={interval}&time_period=20&apikey={API_KEY}").json()
        
        return float(p_res['price']), float(r_res['values'][0]['rsi']), float(s_res['values'][0]['sma'])
    except Exception:
        return None, None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menu utama untuk memilih timeframe."""
    keyboard = [
        [InlineKeyboardButton("⚡ 5m (Scalp)", callback_data='5min'), InlineKeyboardButton("⚡ 15m (Scalp)", callback_data='15min')],
        [InlineKeyboardButton("📊 30m (Trend)", callback_data='30min'), InlineKeyboardButton("📊 1h (Trend)", callback_data='1h')]
    ]
    await update.message.reply_text("Pilih Timeframe untuk Analisis XAU/USD:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Proses analisis berdasarkan pilihan timeframe."""
    query = update.callback_query
    interval = query.data
    await query.answer()
    
    price, rsi, sma = get_market_data("XAU/USD", interval)
    
    if price:
        confidence = 50
        trend = "NEUTRAL"
        
        # Logika Konfluensi (Harga & Tren SMA)
        if price > sma and rsi < 40: 
            trend = "🟢 BUY"
            confidence += 30
        elif price < sma and rsi > 60:
            trend = "🔴 SELL"
            confidence += 30
        
        # Bonus tambahan untuk RSI Ekstrim (Overbought/Oversold)
        if rsi < 30 or rsi > 70: confidence += 15

        # Perhitungan Dinamis SL/TP
        tp = (price + 3.0) if "BUY" in trend else (price - 3.0)
        sl = (price - 2.0) if "BUY" in trend else (price + 2.0)

        msg = (f"📈 **Analisis XAU/USD - {interval}**\n\n"
               f"💰 Price: ${price:.2f}\n"
               f"📉 RSI: {rsi:.1f} | SMA20: ${sma:.2f}\n"
               f"🎯 Signal: {trend} (Conf: {min(confidence, 95)}%)\n\n"
               f"🎯 Target Profit: ${tp:.2f}\n🛡 Stop Loss: ${sl:.2f}\n\n"
               f"⚠️ *Trading dengan disiplin & RRR yang ketat!*")
        
        await query.edit_message_text(text=msg, parse_mode='Markdown')
    else:
        await query.edit_message_text(text="❌ Data gagal dimuat. Coba lagi dalam beberapa detik.")

if __name__ == '__main__':
    # Membangun aplikasi bot
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.run_polling()
