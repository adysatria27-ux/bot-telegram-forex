from technical_analysis import button, get_market_data

application.add_handler(CallbackQueryHandler(button, pattern="^analyze_xauusd$"))
