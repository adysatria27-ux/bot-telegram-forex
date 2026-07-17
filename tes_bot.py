import os
import logging
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# --- 1. SETUP ---
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("BOT_TOKEN", "").strip()
# (Masukkan logika teknikal dan fungsi get_market_data di sini)

# --- 2. FUNGSI HANDLER (Didefinisikan DI ATAS agar tidak error) ---
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    # Logika analisis Anda
    await query.edit_message_text("Hasil Analisis lengkap...")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Cek Harga XAUUSD"], ["Cek Harga EURUSD"], ["Hubungi Admin"]]
    await update.message.reply_text("Silakan pilih menu:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "Cek Harga XAUUSD":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 Analisa Pro XAU/USD", callback_data="analyze_xauusd")]])
        await update.message.reply_text("Klik untuk analisa:", reply_markup=kb)
    elif text == "Cek Harga EURUSD":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 Analisa Pro EUR/USD", callback_data="analyze_eurusd")]])
        await update.message.reply_text("Klik untuk analisa:", reply_markup=kb)
    else:
        await update.message.reply_text("Gunakan menu yang tersedia.")

# --- 3. MAIN RUNNER (Dipanggil PALING BAWAH) ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_xauusd$"))
    app.add_handler(CallbackQueryHandler(button, pattern="^analyze_eurusd$"))
    app.run_polling()
