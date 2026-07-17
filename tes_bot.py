import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler

# Import modul analisis baru
from xauusd_analysis import button as analysis_button

# Setup logging agar bisa lihat error di Railway
logging.basicConfig(level=logging.INFO)

# Token dari Environment Variable Railway (Lebih aman)
TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context):
    """Menu utama yang memicu fungsi analisis."""
    keyboard = [
        [InlineKeyboardButton("📊 Analisa XAU/USD", callback_data="analyze_xauusd")]
    ]
    await update.message.reply_text(
        "🤖 *Scalping Bot Pro*\nTekan tombol di bawah untuk analisis teknikal (RSI, SMA, BB, ATR):", 
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

if __name__ == '__main__':
    # Pastikan TOKEN ada
    if not TOKEN:
        raise ValueError("BOT_TOKEN belum di-set di Railway Variables!")
        
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Handler
    application.add_handler(CommandHandler('start', start))
    
    # Menghubungkan tombol ke fungsi button di xauusd_analysis.py
    application.add_handler(CallbackQueryHandler(analysis_button, pattern="^analyze_xauusd$"))
    
    application.run_polling()
