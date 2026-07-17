from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Cek Harga XAUUSD", callback_data='xauusd')],
        [InlineKeyboardButton("Cek Harga EURUSD", callback_data='eurusd')],
        [InlineKeyboardButton("Hubungi Admin", url='https://t.me/username_anda')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Halo! Silakan pilih menu di bawah ini:', reply_markup=reply_markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == 'xauusd':
        await query.edit_message_text(text="Harga XAUUSD saat ini: $2415.50")
    elif query.data == 'eurusd':
        await query.edit_message_text(text="Harga EURUSD saat ini: 1.0920")

if __name__ == '__main__':
    application = ApplicationBuilder().token('8866350485:AAE9aI9eUqFm1YynbVy2UfTLHYt_gPCDZFM').build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.run_polling()