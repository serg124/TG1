# Crypto Volume Analysis Telegram Bot

This Telegram bot analyzes cryptocurrency volume changes on ByBit exchange to predict potential price movements (pump or dump).

## Features
- Real-time volume analysis for specified cryptocurrencies
- Price movement predictions based on volume changes
- Telegram notifications for potential trading opportunities

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your credentials:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
```

3. Run the bot:
```bash
python main.py
```

## Usage
1. Start the bot in Telegram
2. Use /start to begin
3. Use /analyze <symbol> to analyze a specific cryptocurrency (e.g., /analyze BTCUSDT)
4. Use /settings to configure analysis parameters

## Disclaimer
This bot is for educational purposes only. Cryptocurrency trading involves significant risks. 