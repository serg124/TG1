import os
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check if environment variables are set
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

logger.info("Checking environment variables...")
logger.info(f"TELEGRAM_TOKEN present: {'Yes' if TELEGRAM_TOKEN else 'No'}")
logger.info(f"BYBIT_API_KEY present: {'Yes' if BYBIT_API_KEY else 'No'}")
logger.info(f"BYBIT_API_SECRET present: {'Yes' if BYBIT_API_SECRET else 'No'}")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file!")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    logger.warning("ByBit API credentials not found in .env file. Some features may be limited.")

# Initialize ByBit client
def initialize_bybit_client():
    """Initialize Bybit client with retries."""
    global session
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to initialize ByBit client (attempt {attempt + 1}/{max_retries})...")
            
            if not BYBIT_API_KEY or not BYBIT_API_SECRET:
                raise ValueError("ByBit API credentials not found in .env file")
                
            session = HTTP(
                testnet=False,
                api_key=BYBIT_API_KEY,
                api_secret=BYBIT_API_SECRET,
                recv_window=20000
            )
            
            # Проверяем подключение
            test_response = session.get_instruments_info(
                category="spot",
                symbol="BTCUSDT"
            )
            
            if test_response['retCode'] == 0:
                logger.info("ByBit client initialized and tested successfully")
                return True
            else:
                logger.error(f"ByBit client test failed: {test_response['retMsg']}")
                
        except Exception as e:
            logger.error(f"Error initializing ByBit client (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            continue
            
    logger.error("Failed to initialize ByBit client after all retries")
    return False

# Constants for analysis
DEFAULT_SETTINGS = {
    'VOLUME_THRESHOLD': 2.0,  # Порог изменения объема
    'PRICE_CHANGE_THRESHOLD': 0.02,  # Порог изменения цены
    'LOOKBACK_PERIODS': 24,  # Number of periods to look back
    'SCAN_INTERVAL': 300,  # Интервал сканирования в секундах
    'MIN_VOLUME_USDT': 100_000_000,  # Минимальный 24-часовой объем (100M USDT)
    'NOTIFICATION_LEVEL': 'basic'  # 'basic', 'detailed', или 'all'
    #'MIN_MARKET_CAP': 100_000_000,  # Минимальная рыночная капитализация (100M USDT)
}

# Store settings for each user
user_settings = {}

# Store last signals to avoid duplicates
last_signals = {}
# Store active users
active_users = set()

def get_user_settings(user_id: int) -> dict:
    """Get settings for a specific user or return default settings."""
    return user_settings.get(user_id, DEFAULT_SETTINGS.copy())

async def get_all_symbols():
    """Get all valid symbols from Bybit SPOT market."""
    try:
        global session
        
        # Проверяем инициализацию сессии
        if session is None:
            logger.error("Bybit session is not initialized, attempting to reinitialize...")
            if not initialize_bybit_client():
                logger.error("Failed to reinitialize Bybit client")
                return []
        
        logger.info("Requesting SPOT instruments info...")
        try:
            spot_response = session.get_instruments_info(
                category="spot"
            )
        except Exception as api_error:
            logger.error(f"Error calling Bybit API: {str(api_error)}")
            # Пробуем переинициализировать сессию
            if initialize_bybit_client():
                spot_response = session.get_instruments_info(
                    category="spot"
                )
            else:
                return []
        
        valid_symbols = []
        
        if spot_response.get('retCode') == 0:
            spot_list = spot_response.get('result', {}).get('list', [])
            
            for item in spot_list:
                if isinstance(item, dict):
                    symbol = item.get('symbol', '')
                    if symbol.endswith('USDT'):
                        valid_symbols.append(symbol)
                        
            logger.info(f"Found {len(valid_symbols)} SPOT USDT symbols")
            if valid_symbols:
                logger.info(f"Sample symbols: {valid_symbols[:5]}")
        else:
            logger.error(f"Error in SPOT API response: {spot_response.get('retMsg', 'Unknown error')}")

        return valid_symbols

    except Exception as e:
        logger.error(f"Error in get_all_symbols: {str(e)}")
        return []

async def scan_markets(application: Application):
    """Scan all markets for signals."""
    try:
        if not active_users:
            logger.info("No active users, skipping scan")
            return

        symbols = await get_all_symbols()
        
        if not symbols:
            logger.error("No symbols found")
            return

        # Отправляем уведомление о начале сканирования
        scan_start_message = (
            f"🔄 *Начало сканирования рынка*\n"
            f"*Всего символов:* {len(symbols)}\n"
            f"*Время начала:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"*Условия анализа:*\n"
            f"📊 *Базовый анализ:*\n"
            f"• Изменение объема ≥ 50%\n"
            f"• Изменение OI ≥ 20%\n"
            f"• Ставка финансирования\n\n"
            f"📈 *Расширенный анализ:*\n"
            f"• RSI (перекупленность/перепроданность)\n"
            f"• VWAP (объемно-взвешенная цена)\n"
            f"• ATR (волатильность)\n"
            f"• Анализ ликвидности\n\n"
            f"*Статус:* Активен ✅"
        )
        
        for user_id in active_users:
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text=scan_start_message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error sending start message to user {user_id}: {str(e)}")
        
        start_time = datetime.now()
        processed_symbols = 0
        
        for symbol in symbols:
            processed_symbols += 1
            for user_id in active_users:
                user_settings = get_user_settings(user_id)
                try:
                    await analyze_symbol(application, symbol, user_id)
                    await analyze_symbol_advanced(application, symbol, user_id)
                except Exception as e:
                    logger.error(f"Error analyzing symbol {symbol} for user {user_id}: {str(e)}")
                    continue
            
            await asyncio.sleep(1)
        
        scan_duration = (datetime.now() - start_time).seconds
        
        # Отправляем финальное уведомление
        scan_end_message = (
            f"✅ *Сканирование завершено*\n"
            f"*Время завершения:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"*Длительность:* {scan_duration} секунд\n"
            f"*Обработано символов:* {processed_symbols}\n"
        )
        
        for user_id in active_users:
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text=scan_end_message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error sending end message to user {user_id}: {str(e)}")
        
        await asyncio.sleep(DEFAULT_SETTINGS['SCAN_INTERVAL'])
        
    except Exception as e:
        logger.error(f"Error in scanner task: {str(e)}")

async def analyze_symbol(application: Application, symbol: str, user_id: int):
    """Analyze a single symbol and send notification if signal detected."""
    try:
        settings = get_user_settings(user_id)
        logger.info(f"Analyzing symbol: {symbol} for user: {user_id}")

        # Get historical kline data
        klines = session.get_kline(
            symbol=symbol,
            interval="15",
            limit=settings['LOOKBACK_PERIODS']
        )

        if klines['retCode'] != 0:
            logger.warning(f"Failed to get kline data for {symbol}: {klines['retMsg']}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(klines['result']['list'])
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        # Convert to numeric
        for col in ['volume', 'close', 'open', 'high', 'low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate volume changes
        df['volume_change'] = df['volume'].pct_change()
        latest_volume_change = df['volume_change'].iloc[-1] * 100

        # Calculate price change
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100

        # Get OI data
        oi_data = session.get_open_interest(category="linear", symbol=symbol, intervalTime="5min")

        if oi_data.get('retCode', -1) != 0:
            logger.warning(f"Failed to get OI data for {symbol}: {oi_data.get('retMsg', 'No error message')}")
            return None

        result = oi_data.get('result', {})
        if not result or 'list' not in result or not result['list']:
            logger.error(f"Unexpected structure in OI data: {oi_data}")
            return None

        # Extract the first data point
        data_point = result['list'][0]
        current_oi = float(data_point.get('openInterest', 0))
        previous_oi = float(data_point.get('previousOpenInterest', 0))

        # Calculate OI change safely
        oi_change = ((current_oi - previous_oi) / previous_oi * 100) if previous_oi > 0 else 0

        # Get funding rate
        funding_rate_data = session.get_funding_rate_history(category="linear", symbol=symbol)

        if funding_rate_data.get('retCode', -1) != 0:
            logger.warning(f"Failed to get funding rate data for {symbol}")
            return None

        result = funding_rate_data.get('result', {})
        if not result or 'list' not in result or not result['list']:
            logger.error(f"Unexpected structure in funding rate data: {funding_rate_data}")
            return None

        latest_data = result['list'][0]
        funding_rate = float(latest_data.get('fundingRate', 0))

        # Determine pump or dump conditions and prepare messages
        pump_message = None
        dump_message = None

        if latest_volume_change >= 50 and price_change > 0 and oi_change >= 20 and funding_rate < 0.001:
            pump_message = (
                f"🚀 *Потенциальный Памп для {symbol}!* 🚀\n"
                f"*Последний объем:* {df['volume'].iloc[-1]:,.2f}\n"
                f"*Изменение объема:* {latest_volume_change:.2f}%\n"
                f"*Изменение цены:* {price_change:.2f}%\n"
                f"*Изменение OI:* {oi_change:.2f}%\n"
                f"*Ставка финансирования:* {funding_rate:.4f}\n"
                f"📈 *Проверьте на ByBit:* [Ссылка на {symbol}](https://www.bybit.com/en/trade/spot/{symbol})"
            )

        elif latest_volume_change >= 50 and price_change < 0 and oi_change >= 20 and funding_rate > 0.001:
            dump_message = (
                f"🔻 *Потенциальный Дамп для {symbol}!* 🔻\n"
                f"*Последний объем:* {df['volume'].iloc[-1]:,.2f}\n"
                f"*Изменение объема:* {latest_volume_change:.2f}%\n"
                f"*Изменение цены:* {price_change:.2f}%\n"
                f"*Изменение OI:* {oi_change:.2f}%\n"
                f"*Ставка финансирования:* {funding_rate:.4f}\n"
                f"📉 *Проверьте на ByBit:* [Ссылка на {symbol}](https://www.bybit.com/en/trade/spot/{symbol})"
            )

        # Send messages if conditions are met
        if pump_message:
            await application.bot.send_message(
                chat_id=user_id,
                text=pump_message,
                parse_mode='Markdown'
            )
        elif dump_message:
            await application.bot.send_message(
                chat_id=user_id,
                text=dump_message,
                parse_mode='Markdown'
            )

        return None

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

async def analyze_symbol_advanced(application: Application, symbol: str, user_id: int):
    """Advanced analysis for pump and dump signals using additional indicators."""
    try:
        settings = get_user_settings(user_id)
        logger.info(f"Advanced analysis for symbol: {symbol} for user: {user_id}")

        # Get historical kline data with more periods for better analysis
        klines = session.get_kline(
            symbol=symbol,
            interval="15",  # 15-minute intervals
            limit=100  # Больше периодов для лучшего анализа
        )

        if klines['retCode'] != 0:
            logger.warning(f"Failed to get kline data for {symbol}: {klines['retMsg']}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(klines['result']['list'])
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1. Calculate additional technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['turnover'].cumsum() / df['volume'].cumsum())

        # Average True Range (ATR)
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()

        # 2. Get market data
        market_data = session.get_tickers(
            category="spot",
            symbol=symbol
        )

        if market_data['retCode'] != 0:
            logger.warning(f"Failed to get market data for {symbol}: {market_data['retMsg']}")
            return None
        
        # 3. Get order book data for liquidity analysis
        orderbook = session.get_orderbook(
            category="spot",
            symbol=symbol
        )

        if orderbook['retCode'] != 0:
            logger.warning(f"Failed to get orderbook data for {symbol}: {orderbook['retMsg']}")
            return None
        
        # Calculate current market conditions
        current_price = float(market_data['result']['list'][0]['lastPrice'])
        price_24h_change = float(market_data['result']['list'][0]['price24hPcnt']) * 100
        volume_24h = float(market_data['result']['list'][0]['volume24h'])
        turnover_24h = float(market_data['result']['list'][0]['turnover24h'])

        # Calculate liquidity indicators
        bid_sum = sum(float(level[1]) for level in orderbook['result']['b'][:5])
        ask_sum = sum(float(level[1]) for level in orderbook['result']['a'][:5])
        liquidity_ratio = bid_sum / ask_sum if ask_sum > 0 else 0

        # Get latest indicators
        latest_rsi = df['RSI'].iloc[-1]
        latest_vwap = df['VWAP'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        
        # Calculate volume conditions
        volume_mean = df['volume'].rolling(24).mean().iloc[-1]
        latest_volume = df['volume'].iloc[-1]
        
        # Advanced Pump Detection Criteria
        pump_conditions = {
            'volume_spike': latest_volume > volume_mean * 3,
            'price_above_vwap': current_price > latest_vwap * 1.02,
            'high_rsi': latest_rsi > 70,
            'increasing_liquidity': liquidity_ratio > 1.2,
            'volatility_increase': latest_atr > df['ATR'].rolling(24).mean().iloc[-1] * 1.5
        }

        # Advanced Dump Detection Criteria
        dump_conditions = {
            'volume_spike': latest_volume > volume_mean * 3,
            'price_below_vwap': current_price < latest_vwap * 0.98,
            'low_rsi': latest_rsi < 30,
            'decreasing_liquidity': liquidity_ratio < 0.8,
            'volatility_increase': latest_atr > df['ATR'].rolling(24).mean().iloc[-1] * 1.5
        }

        # Calculate confidence scores
        pump_score = sum(pump_conditions.values()) / len(pump_conditions) * 100
        dump_score = sum(dump_conditions.values()) / len(dump_conditions) * 100

        # Send signals based on confidence scores
        if pump_score >= 80:
            pump_message = (
                f"🚀 *РАСШИРЕННЫЙ АНАЛИЗ - Вероятность Пампа {pump_score:.1f}%* 🚀\n"
                f"*Символ:* {symbol}\n"
                f"*Текущая цена:* ${current_price:,.8f}\n"
                f"*Изменение цены (24ч):* {price_24h_change:.2f}%\n"
                f"*Объём (24ч):* ${volume_24h:,.2f}\n"
                f"*RSI:* {latest_rsi:.2f}\n"
                f"*Отношение к VWAP:* {(current_price/latest_vwap - 1) * 100:.2f}%\n"
                f"*Коэффициент ликвидности:* {liquidity_ratio:.2f}\n\n"
                f"*Индикаторы пампа:*\n"
                f"{'✅' if pump_conditions['volume_spike'] else '❌'} Всплеск объема\n"
                f"{'✅' if pump_conditions['price_above_vwap'] else '❌'} Цена выше VWAP\n"
                f"{'✅' if pump_conditions['high_rsi'] else '❌'} Высокий RSI\n"
                f"{'✅' if pump_conditions['increasing_liquidity'] else '❌'} Растущая ликвидность\n"
                f"{'✅' if pump_conditions['volatility_increase'] else '❌'} Повышенная волатильность\n\n"
                f"📈 *Проверьте на ByBit:* [Ссылка на {symbol}](https://www.bybit.com/en/trade/spot/{symbol})"
            )
            await application.bot.send_message(
                chat_id=user_id,
                text=pump_message,
                parse_mode='Markdown'
            )

        elif dump_score >= 80:
            dump_message = (
                f"🔻 *РАСШИРЕННЫЙ АНАЛИЗ - Вероятность Дампа {dump_score:.1f}%* 🔻\n"
                f"*Символ:* {symbol}\n"
                f"*Текущая цена:* ${current_price:,.8f}\n"
                f"*Изменение цены (24ч):* {price_24h_change:.2f}%\n"
                f"*Объём (24ч):* ${volume_24h:,.2f}\n"
                f"*RSI:* {latest_rsi:.2f}\n"
                f"*Отношение к VWAP:* {(current_price/latest_vwap - 1) * 100:.2f}%\n"
                f"*Коэффициент ликвидности:* {liquidity_ratio:.2f}\n\n"
                f"*Индикаторы дампа:*\n"
                f"{'✅' if dump_conditions['volume_spike'] else '❌'} Всплеск объема\n"
                f"{'✅' if dump_conditions['price_below_vwap'] else '❌'} Цена ниже VWAP\n"
                f"{'✅' if dump_conditions['low_rsi'] else '❌'} Низкий RSI\n"
                f"{'✅' if dump_conditions['decreasing_liquidity'] else '❌'} Падающая ликвидность\n"
                f"{'✅' if dump_conditions['volatility_increase'] else '❌'} Повышенная волатильность\n\n"
                f"📉 *Проверьте на ByBit:* [Ссылка на {symbol}](https://www.bybit.com/en/trade/spot/{symbol})"
            )
            await application.bot.send_message(
                chat_id=user_id,
                text=dump_message,
                parse_mode='Markdown'
            )

        return None

    except Exception as e:
        logger.error(f"Error in advanced analysis for {symbol}: {str(e)}")
        return None

async def send_signal_message(application: Application, symbol: str, signal: str, latest_volume: float, avg_volume: float, latest_price_change: float, volume_change: float, volume_24h: float, user_id: int):
    emoji = "🟢" if "PUMP" in signal else "🔴"
    current_volume_display = f"${latest_volume / 1_000_000:,.2f}M"  # Форматируем текущий объем в миллионах
    average_volume_display = f"${avg_volume / 1_000_000:,.2f}M"  # Форматируем средний объем в миллионах
    ticker_link = f"[Посмотреть на ByBit](https://www.bybit.com/en/trade/spot/{symbol})"  # Создаем ссылку на тикер

    message = (
        f"{emoji} *Сигнал {signal.replace('_', ' ')} обнаружен!*\n\n"
        f"*Символ:* {symbol} {ticker_link}\n"  # Добавляем ссылку на символ
        f"*Текущий объем:* {current_volume_display}\n"
        f"*Средний объем:* {average_volume_display}\n"
        f"*Изменение объема:* {volume_change:.1f}%\n"
        f"*Изменение цены:* {latest_price_change:.1f}%\n"
        f"*24h объем:* ${volume_24h:,.2f}\n"
        f"*Анализ основан на последних {DEFAULT_SETTINGS['LOOKBACK_PERIODS']} периодах*"
    )

    await application.bot.send_message(
                                chat_id=user_id,
        text=message,
        parse_mode='Markdown'  # Включаем форматирование Markdown
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    logger.info("Start command received")
    if not update.message:
        logger.warning("Received update without message")
        return
        
    user_id = update.effective_user.id
    active_users.add(user_id)
    logger.info(f"Added user {user_id} to active users. Total active users: {len(active_users)}")
    
    welcome_message = (
        "👋 Welcome to the Crypto Volume Analysis Bot!\n\n"
        "This bot automatically scans all cryptocurrencies on ByBit for volume-based signals.\n\n"
        "You will receive notifications when:\n"
        "🟢 Volume increases significantly with positive price movement\n"
        "🔴 Volume increases significantly with negative price movement\n\n"
        "Available commands:\n"
        "/test_scan - Test Scan\n"
        "/settings - Configure bot parameters\n"
        "/status - Check scanner status\n"
        "/stop - Stop receiving signals\n"
        "/help - Show this help message"
    )
    try:
        await update.message.reply_text(welcome_message)
        logger.info(f"Successfully sent welcome message to user {user_id}")
    except Exception as e:
        logger.error(f"Error sending welcome message to user {user_id}: {str(e)}")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop receiving signals."""
    if not update.message:
        logger.warning("Received update without message")
        return
        
    user_id = update.effective_user.id
    active_users.discard(user_id)
    try:
        await update.message.reply_text("You have been unsubscribed from signals. Use /start to subscribe again.")
    except Exception as e:
        logger.error(f"Error sending stop message: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check scanner status."""
    if not update.message:
        logger.warning("Received update without message")
        return
        
    user_id = update.effective_user.id
    settings = get_user_settings(user_id)
    symbols = await get_all_symbols()
    
    status_message = (
        f"📊 Scanner Status:\n\n"
        f"Active: ✅\n"
        f"Active users: {len(active_users)}\n"
        f"Scanning {len(symbols)} symbols\n"
        f"Scan interval: {settings['SCAN_INTERVAL']:,} seconds\n"
        f"Volume threshold: {settings['VOLUME_THRESHOLD']:,.1f}x\n"
        f"Price threshold: {settings['PRICE_CHANGE_THRESHOLD']:.1%}\n"
        f"Min 24h volume: ${settings['MIN_VOLUME_USDT']:,.0f}\n"
    )
    try:
        await update.message.reply_text(status_message)
    except Exception as e:
        logger.error(f"Error sending status message: {str(e)}")

async def settings_keyboard():
    """Create settings keyboard markup."""
    keyboard = [
        [
            InlineKeyboardButton("Volume Threshold", callback_data="set_volume_threshold"),
            InlineKeyboardButton("Price Threshold", callback_data="set_price_threshold")
        ],
        [
            InlineKeyboardButton("Lookback Periods", callback_data="set_lookback"),
            InlineKeyboardButton("Scan Interval", callback_data="set_scan_interval")
        ],
        [
            InlineKeyboardButton("Min Volume", callback_data="set_min_volume"),
           # InlineKeyboardButton("Min Market Cap", callback_data="set_min_market_cap")
        ],
        [InlineKeyboardButton("Back to Main Menu", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show and manage bot settings."""
    if not update.message:
        logger.warning("Received update without message")
        return
        
    user_id = update.effective_user.id
    settings = get_user_settings(user_id)
    
    settings_message = (
        "⚙️ Bot Settings\n\n"
        "Current values:\n"
        f"Volume Threshold: {settings['VOLUME_THRESHOLD']:,.1f}x\n"
        f"Price Change Threshold: {settings['PRICE_CHANGE_THRESHOLD']:.1%}\n"
        f"Lookback Periods: {settings['LOOKBACK_PERIODS']:,}\n"
        f"Scan Interval: {settings['SCAN_INTERVAL']:,} seconds\n"
        f"Min 24h Volume: ${settings['MIN_VOLUME_USDT']:,.0f}\n"
       # f"Min Market Cap: ${settings['MIN_MARKET_CAP']:,.0f}\n\n"
        "Click a button below to change a setting:"
    )
    
    try:
        await update.message.reply_text(
            settings_message,
            reply_markup=await settings_keyboard()
        )
    except Exception as e:
        logger.error(f"Error sending settings message: {str(e)}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    if not query.data.startswith("set_"):
        return
    
    setting_name = query.data[4:]  # Remove 'set_' prefix
    user_id = query.from_user.id
    settings = get_user_settings(user_id)
    
    # Create value adjustment buttons
    current_value = None
    step = None
    
    if setting_name == "volume_threshold":
        current_value = settings['VOLUME_THRESHOLD']
        step = 0.5
        title = "Volume Threshold"
        unit = "x"
    elif setting_name == "price_threshold":
        current_value = settings['PRICE_CHANGE_THRESHOLD']
        step = 0.01
        title = "Price Threshold"
        unit = "%"
        current_value = current_value * 100  # Convert decimal to percentage
        step = step * 100  # Convert step to percentage
    elif setting_name == "lookback":
        current_value = settings['LOOKBACK_PERIODS']
        step = 6
        title = "Lookback Periods"
        unit = "periods"
    elif setting_name == "scan_interval":
        current_value = settings['SCAN_INTERVAL']
        step = 60
        title = "Scan Interval"
        unit = "sec"
    elif setting_name == "min_volume":
        current_value = settings['MIN_VOLUME_USDT']
        step = 10_000_000
        title = "Min Volume"
        unit = "USDT"
    #elif setting_name == "min_market_cap":
    #    current_value = settings['MIN_MARKET_CAP']
    #    step = 10_000_000
    #    title = "Min Market Cap"
    #    unit = "USDT"
    

    keyboard = [
        [
            InlineKeyboardButton("-10", callback_data=f"adj_{setting_name}_-10"),
            InlineKeyboardButton("-1", callback_data=f"adj_{setting_name}_-1"),
            InlineKeyboardButton("+1", callback_data=f"adj_{setting_name}_1"),
            InlineKeyboardButton("+10", callback_data=f"adj_{setting_name}_10")
        ],
        [InlineKeyboardButton("Enter Value Manually", callback_data=f"manual_{setting_name}")],
        [InlineKeyboardButton("Back to Settings", callback_data="back_to_settings")]
    ]
    
    # Format current value and step with thousand separators
    if setting_name in ["min_volume"]:
        current_value_display = f"${current_value:,.0f}"
        step_display = f"${step:,.0f}"
    elif setting_name == "price_threshold":
        current_value_display = f"{current_value:,.1f}"
        step_display = f"{step:,.1f}"
    else:
        current_value_display = f"{current_value:,.1f}"
        step_display = f"{step:,.1f}"
    
    message = (
        f"Adjust {title}\n\n"
        f"Current value: {current_value_display}{unit}\n"
        f"Step size: {step_display}{unit}\n\n"
        "Choose an option:"
    )
    
    await query.edit_message_text(
        text=message,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def manual_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle manual value input."""
    query = update.callback_query
    await query.answer()
    
    if not query.data.startswith("manual_"):
        return
        
    setting_name = query.data[7:]  # Remove 'manual_' prefix
    user_id = query.from_user.id
    
    # Store the setting name in context for later use
    context.user_data['manual_setting'] = setting_name
    
    # Define input prompts for each setting
    prompts = {
        "volume_threshold": "Enter volume threshold (1.0-10.0):",
        "price_threshold": "Enter price threshold (1-50):",
        "lookback": "Enter lookback periods (6-72):",
        "scan_interval": "Enter scan interval in seconds (60-3600):",
        "min_volume": "Enter minimum volume in USDT (1M-1B):"
        #"min_market_cap": "Enter minimum market cap in USDT (10M-1B):"
    }
    
    keyboard = [[InlineKeyboardButton("Cancel", callback_data="back_to_settings")]]
    
    await query.edit_message_text(
        text=f"{prompts[setting_name]}\n\nPlease enter the new value:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    
    # Set the next handler to process the manual input
    context.user_data['waiting_for_input'] = True

async def process_manual_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process manually entered value."""
    if not context.user_data.get('waiting_for_input'):
        return
        
    if not update.message:
        return
        
    try:
        setting_name = context.user_data['manual_setting']
        user_id = update.effective_user.id
        settings = get_user_settings(user_id)
        
        # Get the new value from the message
        new_value = float(update.message.text)
        
        # Define validation rules for each setting
        validation_rules = {
            "volume_threshold": {"min": 1.0, "max": 10.0, "key": "VOLUME_THRESHOLD", "unit": "x"},
            "price_threshold": {"min": 1, "max": 50, "key": "PRICE_CHANGE_THRESHOLD", "convert": lambda x: x / 100, "unit": "%"},
            "lookback": {"min": 6, "max": 72, "key": "LOOKBACK_PERIODS", "unit": "periods"},
            "scan_interval": {"min": 60, "max": 3600, "key": "SCAN_INTERVAL", "unit": "sec"},
            "min_volume": {"min": 1_000_000, "max": 1_000_000_000, "key": "MIN_VOLUME_USDT", "unit": "USDT", "is_money": True}
            #"min_market_cap": {"min": 10_000_000, "max": 1_000_000_000, "key": "MIN_MARKET_CAP", "unit": "USDT", "is_money": True}
        }
        
        rule = validation_rules[setting_name]
        
        # Convert value if needed
        if 'convert' in rule:
            new_value = rule['convert'](new_value)
        
        # Validate the value
        if not (rule['min'] <= new_value <= rule['max']):
            raise ValueError(f"Value must be between {rule['min']} and {rule['max']}")
        
        # Update the setting
        settings[rule['key']] = new_value
        user_settings[user_id] = settings
        
        # Format the display value
        if rule.get('is_money', False):
            value_display = f"${new_value:,.0f}"
        elif setting_name == "price_threshold":
            value_display = f"{new_value * 100:.1f}"
        else:
            value_display = f"{new_value:,.1f}"
        
        # Clear the waiting state
        context.user_data['waiting_for_input'] = False
        context.user_data['manual_setting'] = None
        
        # Send confirmation message
        await update.message.reply_text(
            f"✅ Setting updated successfully!\nNew value: {value_display}{rule['unit']}",
            reply_markup=await settings_keyboard()
        )
        
    except ValueError as e:
        await update.message.reply_text(
            f"❌ Invalid input: {str(e)}\nPlease try again:",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="back_to_settings")]])
        )
    except Exception as e:
        logger.error(f"Error processing manual input: {str(e)}")
        context.user_data['waiting_for_input'] = False
        context.user_data['manual_setting'] = None
        await update.message.reply_text(
            "❌ An error occurred. Please try again or use the buttons.",
            reply_markup=await settings_keyboard()
        )

async def adjust_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle setting adjustment callbacks."""
    query = update.callback_query
    await query.answer()
    
    if not query.data.startswith("adj_"):
        return
        
    # Fix the data parsing
    try:
        # Split the data into parts: adj_setting_name_multiplier
        parts = query.data.split("_")
        if len(parts) < 3:  # Changed from 4 to 3
            logger.error(f"Invalid callback data format: {query.data}")
            return
            
        # Handle both simple and compound setting names
        if len(parts) == 3:
            # Simple format: adj_setting_multiplier
            setting_name = parts[1]
            multiplier = int(parts[2])
        else:
            # Compound format: adj_setting_part1_part2_multiplier
            setting_name = "_".join(parts[1:-1])
            multiplier = int(parts[-1])
            
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing callback data: {str(e)}")
        return
    
    user_id = query.from_user.id
    settings = get_user_settings(user_id)
    
    # Define step sizes and limits for each setting
    setting_configs = {
        "volume_threshold": {
            "step": 0.5, 
            "min": 1.0, 
            "max": 10.0, 
            "key": "VOLUME_THRESHOLD", 
            "unit": "x",
            "title": "Volume Threshold"
        },
        "price_threshold": {
            "step": 0.01, 
            "min": 0.01, 
            "max": 0.5, 
            "key": "PRICE_CHANGE_THRESHOLD", 
            "unit": "%",
            "title": "Price Threshold",
            "is_percentage": True,
            "display_multiplier": 100
        },
        "lookback": {
            "step": 6, 
            "min": 6, 
            "max": 72, 
            "key": "LOOKBACK_PERIODS", 
            "unit": "periods",
            "title": "Lookback Periods"
        },
        "scan_interval": {
            "step": 60, 
            "min": 60, 
            "max": 3600, 
            "key": "SCAN_INTERVAL", 
            "unit": "sec",
            "title": "Scan Interval"
        },
        "min_volume": {
            "step": 10_000_000, 
            "min": 1_000_000, 
            "max": 1_000_000_000, 
            "key": "MIN_VOLUME_USDT", 
            "unit": "USDT",
            "title": "Min Volume",
            "is_money": True
        }
        #"min_market_cap": {
        #    "step": 10_000_000, 
        #     "min": 10_000_000, 
        #    "max": 1_000_000_000, 
        #    "key": "MIN_MARKET_CAP", 
        #    "unit": "USDT",
        #    "title": "Min Market Cap",
        #    "is_money": True
        #}
    }
    
    if setting_name not in setting_configs:
        logger.error(f"Invalid setting name: {setting_name}")
        return
    
    config = setting_configs[setting_name]
    current_value = settings[config["key"]]
    
    # Calculate new value
    step = config["step"]
    new_value = current_value + (step * multiplier)
    
    # Ensure new value is within limits
    new_value = max(config["min"], min(config["max"], new_value))
    
    # Update setting
    settings[config["key"]] = new_value
    user_settings[user_id] = settings
    
    # Format display values
    if config.get("is_money", False):
        value_display = f"${new_value:,.0f}"
        step_display = f"${step:,.0f}"
    elif config.get("is_percentage", False):
        display_multiplier = config.get("display_multiplier", 100)
        value_display = f"{new_value * display_multiplier:.1f}%"
        step_display = f"{step * display_multiplier:.1f}%"
    else:
        value_display = f"{new_value:,.1f}"
        step_display = f"{step:,.1f}"
    
    keyboard = [
        [
            InlineKeyboardButton("-10", callback_data=f"adj_{setting_name}_-10"),
            InlineKeyboardButton("-1", callback_data=f"adj_{setting_name}_-1"),
            InlineKeyboardButton("+1", callback_data=f"adj_{setting_name}_1"),
            InlineKeyboardButton("+10", callback_data=f"adj_{setting_name}_10")
        ],
        [InlineKeyboardButton("Enter Value Manually", callback_data=f"manual_{setting_name}")],
        [InlineKeyboardButton("Back to Settings", callback_data="back_to_settings")]
    ]
    
    message = (
        f"Adjust {config['title']}\n\n"
        f"Current value: {value_display} {config['unit']}\n"
        f"Step size: {step_display} {config['unit']}\n\n"
        "Choose an option:"
    )
    
    try:
        await query.edit_message_text(
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        logger.error(f"Error updating message: {str(e)}")
        await query.answer("Error updating value. Please try again.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    if not update.message:
        logger.warning("Received update without message")
        return
        
    help_text = (
        "Available commands:\n"
        "/start - Start receiving signals\n"
        "/stop - Stop receiving signals\n"
        "/status - Check scanner status\n"
        "/settings - Configure bot parameters\n"
        "/help - Show this help message"
    )
    try:
        await update.message.reply_text(help_text)
    except Exception as e:
        logger.error(f"Error sending help message: {str(e)}")

async def back_to_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle back to settings button."""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    settings = get_user_settings(user_id)
    
    settings_message = (
        "⚙️ Bot Settings\n\n"
        "Current values:\n"
        f"Volume Threshold: {settings['VOLUME_THRESHOLD']:,.1f}x\n"
        f"Price Change Threshold: {settings['PRICE_CHANGE_THRESHOLD']:.1%}\n"
        f"Lookback Periods: {settings['LOOKBACK_PERIODS']:,}\n"
        f"Scan Interval: {settings['SCAN_INTERVAL']:,} seconds\n"
        f"Min 24h Volume: ${settings['MIN_VOLUME_USDT']:,.0f}\n"
        #f"Min Market Cap: ${settings['MIN_MARKET_CAP']:,.0f}\n\n"
        "Click a button below to change a setting:"
    )
    
    await query.edit_message_text(
        text=settings_message,
        reply_markup=await settings_keyboard()
    )

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle back to main menu button."""
    query = update.callback_query
    await query.answer()
    
    welcome_message = (
        "👋 Welcome to the Crypto Volume Analysis Bot!\n\n"
        "This bot automatically scans all cryptocurrencies on ByBit for volume-based signals.\n\n"
        "You will receive notifications when:\n"
        "🟢 Volume increases significantly with positive price movement\n"
        "🔴 Volume increases significantly with negative price movement\n\n"
        "Available commands:\n"
        "/test_scan - Test Scan\n"
        "/settings - Configure bot parameters\n"
        "/status - Check scanner status\n"
        "/stop - Stop receiving signals\n"
        "/help - Show this help message"
    )
    
    await query.edit_message_text(
        text=welcome_message
    )

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello!")

async def test_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test scanner with current active users."""
    if not update.message:
        logger.warning("Received update without message")
        return
        
    user_id = update.effective_user.id
    
    # Показываем текущее состояние
    status_message = (
        f"📊 *Текущее состояние:*\n"
        f"Активные пользователи: {len(active_users)}\n"
        f"Ваш ID: {user_id}\n"
        f"Вы {'активны' if user_id in active_users else 'не активны'}\n\n"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("Запустить с активными", callback_data="test_scan_active"),
            InlineKeyboardButton("Запустить без активных", callback_data="test_scan_inactive")
        ]
    ]
    
    await update.message.reply_text(
        status_message,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def handle_test_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle test scan callbacks."""
    query = update.callback_query
    await query.answer()
    
    test_type = query.data.split('_')[2]  # 'active' или 'inactive'
    user_id = query.from_user.id
    
    # Сохраняем текущее состояние активных пользователей
    original_active_users = active_users.copy()
    
    try:
        if test_type == 'inactive':
            # Очищаем список активных пользователей
            active_users.clear()
            await query.edit_message_text(
                "🔄 Запуск сканирования без активных пользователей..."
            )
        else:
            # Добавляем текущего пользователя в активные
            active_users.add(user_id)
            await query.edit_message_text(
                "🔄 Запуск сканирования с активными пользователями..."
            )
        
        # Запускаем сканирование
        await scan_markets(context.application)
        
    except Exception as e:
        logger.error(f"Error during test scan: {str(e)}")
        await query.edit_message_text(
            f"❌ Ошибка при сканировании: {str(e)}"
        )
    finally:
        # Восстанавливаем оригинальное состояние активных пользователей
        active_users.clear()
        active_users.update(original_active_users)

def run_bot():
    """Run the bot."""
    try:
        logger.info("Initializing bot application...")
        
        # Инициализируем Bybit клиент перед запуском бота
        if not initialize_bybit_client():
            logger.error("Failed to initialize Bybit client. Bot will start with limited functionality.")
        
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Set up bot commands menu
        commands = [
            ("start", "Start receiving signals"),
            ("test_scan", "Test Scan"),
            ("settings", "Configure bot parameters"),
            ("status", "Check scanner status"),
            ("stop", "Stop receiving signals"),
            ("help", "Show help message")
        ]
        
        # Добавляем новые обработчики для тестирования
        application.add_handler(CommandHandler("test_scan", test_scan))
        application.add_handler(CallbackQueryHandler(handle_test_scan, pattern="^test_scan_"))

        # Add command handlers
        logger.info("Adding command handlers...")
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stop", stop))
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("settings", settings))
        application.add_handler(CommandHandler("hello", hello))
        
        # Add callback handlers
        application.add_handler(CallbackQueryHandler(button_callback, pattern="^set_"))
        application.add_handler(CallbackQueryHandler(adjust_setting, pattern="^adj_"))
        application.add_handler(CallbackQueryHandler(back_to_settings, pattern="^back_to_settings$"))
        application.add_handler(CallbackQueryHandler(back_to_main, pattern="^back_to_main$"))
        application.add_handler(CallbackQueryHandler(manual_input, pattern="^manual_"))
        
        # Add message handler for manual input
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_manual_input))
        
        logger.info("Command and callback handlers added successfully")

        # Add error handler
        async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            logger.error(f"Update {update} caused error: {context.error}")
            if update and update.message:
                try:
                    await update.message.reply_text("Sorry, something went wrong. Please try again later.")
                except Exception as e:
                    logger.error(f"Error sending error message: {str(e)}")

        application.add_error_handler(error_handler)
        logger.info("Error handler added successfully")

        # Start the scanner task with proper task management
        logger.info("Starting scanner task...")
        application.job_queue.run_repeating(
            scan_markets,
            interval=DEFAULT_SETTINGS['SCAN_INTERVAL'],
            first=1
        )
        logger.info("Scanner task started successfully")

        # Define post init callback to set up commands
        async def post_init(app: Application) -> None:
            await app.bot.set_my_commands(commands)
            logger.info("Bot commands menu set up successfully")

        # Start the Bot with post init callback
        logger.info("Starting bot polling...")
        application.post_init = post_init
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Error during bot startup: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Starting bot...")
        run_bot()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {str(e)}")
        raise  # Re-raise the exception to see the full traceback 

async def send_analysis_notification(application, user_id, message, level='basic'):
    settings = get_user_settings(user_id)
    if settings['NOTIFICATION_LEVEL'] == 'all' or (
        settings['NOTIFICATION_LEVEL'] == 'detailed' and level in ['basic', 'detailed']
    ) or (settings['NOTIFICATION_LEVEL'] == 'basic' and level == 'basic'):
        await application.bot.send_message(
            chat_id=user_id,
            text=message,
            parse_mode='Markdown'
        ) 