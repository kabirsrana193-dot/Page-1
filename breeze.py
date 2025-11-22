import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from kiteconnect import KiteConnect, KiteTicker
import time
import threading
import pytz
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# Page config
st.set_page_config(
    page_title="F&O Dashboard - Kite Connect (Live)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
API_KEY = "aj0gv6rpjm11ecac"
API_SECRET = "mgso1jdnxj3xeei228dcciyqqx7axl77"
IST = pytz.timezone('Asia/Kolkata')

# FNO Stocks List
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "ITC", "SBIN", "HCLTECH", "AXISBANK",
    "KOTAKBANK", "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
    "TATAMOTORS", "ADANIPORTS", "ADANIENT", "TECHM", "POWERGRID", "NTPC", "COALINDIA", "TATASTEEL",
    "BAJAJFINSV", "HEROMOTOCO", "INDUSINDBK", "M&M", "GRASIM", "HINDALCO", "JSWSTEEL", "SBILIFE",
    "ICICIGI", "BAJAJ-AUTO", "HDFCLIFE", "ADANIGREEN", "SHREECEM", "EICHERMOT", "UPL", "TATACONSUM",
    "BRITANNIA", "NESTLEIND", "HINDUNILVR", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP"
]

YAHOO_SYMBOLS = {stock: f"{stock}.NS" for stock in FNO_STOCKS}

# Initialize session state
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'kite_connected' not in st.session_state:
    st.session_state.kite_connected = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'instruments_nse' not in st.session_state:
    st.session_state.instruments_nse = None
if 'instruments_nfo' not in st.session_state:
    st.session_state.instruments_nfo = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'ticker_active' not in st.session_state:
    st.session_state.ticker_active = False
if 'kws' not in st.session_state:
    st.session_state.kws = None
if 'live_monitor_running' not in st.session_state:
    st.session_state.live_monitor_running = False
if 'yahoo_live_data' not in st.session_state:
    st.session_state.yahoo_live_data = {}

# Login Management
st.title("📈 F&O Dashboard - Kite Connect")
if not st.session_state.kite_connected:
    st.header("🔐 Login to Kite Connect")
    st.markdown("""
    ### How to get your Access Token:
    1. Click the login link below
    2. After login, copy the `request_token` from URL
    3. Paste it below and generate access token
    """)
    
    login_url = f"https://kite.zerodha.com/connect/login?api_key={API_KEY}&v=3"
    st.markdown(f"### Step 1: [Click here to Login to Kite]({login_url})")
    st.markdown("### Step 2: Enter Request Token")
    
    request_token = st.text_input("Paste Request Token here:", key="request_token_input")
    if st.button("🔑 Generate Access Token", key="generate_token"):
        if request_token and API_SECRET != "YOUR_API_SECRET_HERE":
            try:
                with st.spinner("Generating access token..."):
                    kite = KiteConnect(api_key=API_KEY)
                    data = kite.generate_session(request_token, api_secret=API_SECRET)
                    access_token = data["access_token"]
                    kite.set_access_token(access_token)
                    profile = kite.profile()
                    st.session_state.kite = kite
                    st.session_state.access_token = access_token
                    st.session_state.kite_connected = True
                    st.session_state.profile = profile
                    st.success(f"✅ Connected! Welcome {profile.get('user_name', 'User')}")
                    st.info(f"💾 Save this Access Token: `{access_token}`")
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        elif API_SECRET == "YOUR_API_SECRET_HERE":
            st.error("⚠️ Please set your API_SECRET in the code!")
        else:
            st.warning("⚠️ Please enter the request token")
    
    st.markdown("---")
    st.markdown("### OR Use Existing Access Token")
    manual_token = st.text_input("Paste Access Token:", key="manual_token")
    if st.button("🔗 Connect", key="connect_token"):
        if manual_token:
            try:
                with st.spinner("Connecting..."):
                    kite = KiteConnect(api_key=API_KEY)
                    kite.set_access_token(manual_token)
                    profile = kite.profile()
                    st.session_state.kite = kite
                    st.session_state.access_token = manual_token
                    st.session_state.kite_connected = True
                    st.session_state.profile = profile
                    st.success(f"✅ Connected! Welcome {profile.get('user_name', 'User')}")
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter an access token")
    st.stop()

# Helper Functions
@st.cache_data(ttl=600)
def get_instruments_nfo():
    try:
        instruments = st.session_state.kite.instruments("NFO")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NFO instruments: {e}")
        return None

@st.cache_data(ttl=600)
def get_instruments_nse():
    try:
        instruments = st.session_state.kite.instruments("NSE")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NSE instruments: {e}")
        return None

def filter_market_hours(df):
    if df is None or df.empty:
        return df
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        elif df.index.tz != IST:
            df.index = df.index.tz_convert(IST)
        df_filtered = df.between_time('09:15', '15:30')
        return df_filtered
    except:
        return df

def fetch_historical_data(symbol, days=30, interval="day"):
    try:
        kite = st.session_state.kite
        instruments_nse = get_instruments_nse()
        if instruments_nse is None:
            return None
        result = instruments_nse[instruments_nse['tradingsymbol'] == symbol]
        if result.empty:
            return None
        instrument_token = result.iloc[0]['instrument_token']
        to_date = datetime.now(IST)
        if interval in ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"]:
            if days > 30:
                days = 30
            if to_date.weekday() >= 5:
                days_back = to_date.weekday() - 4
                to_date = to_date - timedelta(days=days_back)
            to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)
            from_date = to_date - timedelta(days=days)
            from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            from_date = to_date - timedelta(days=days)
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date.replace(tzinfo=None),
            to_date=to_date.replace(tzinfo=None),
            interval=interval
        )
        if data:
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            if interval != "day":
                df = filter_market_hours(df)
            df = df[(df[['open', 'high', 'low', 'close']] != 0).all(axis=1)]
            return df
        return None
    except Exception as e:
        st.warning(f"Data fetch error: {e}")
        return None

def get_spot_price(symbol):
    try:
        kite = st.session_state.kite
        quote = kite.quote(f"NSE:{symbol}")
        if quote and f"NSE:{symbol}" in quote:
            return quote[f"NSE:{symbol}"]["last_price"]
        return None
    except Exception as e:
        return None

def calculate_iv(option_price, spot_price, strike, expiry_days, rate=0.06, option_type='CE'):
    """Calculate IV using Black-Scholes"""
    try:
        T = max(expiry_days / 365.0, 0.001)
        
        def black_scholes(S, K, T, r, sigma, opt_type):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if opt_type == 'CE':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        def objective(sigma):
            bs_price = black_scholes(spot_price, strike, T, rate, sigma, option_type)
            return abs(bs_price - option_price)
        
        if option_price <= 0:
            return 0
        
        result = minimize_scalar(objective, bounds=(0.001, 3), method='bounded')
        iv = result.x * 100
        return iv if 0 < iv < 300 else 0
    except:
        return 0

def fetch_fii_dii_latest():
    """Fetch FII/DII data from NSE"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/'
        }
        session = requests.Session()
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
        time.sleep(0.5)
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching FII/DII: {e}")
        return None

def parse_fii_dii_response(api_response):
    """Parse FII/DII response into segments"""
    try:
        if not api_response or not isinstance(api_response, (dict, list)):
            return None
        
        segments = {
            'FII_Equity': {'buy': 0, 'sell': 0, 'net': 0},
            'DII_Equity': {'buy': 0, 'sell': 0, 'net': 0},
            'FII_Derivatives': {'buy': 0, 'sell': 0, 'net': 0},
            'DII_Derivatives': {'buy': 0, 'sell': 0, 'net': 0},
        }
        
        if isinstance(api_response, list):
            for item in api_response:
                if isinstance(item, dict):
                    category = item.get('category', '').upper()
                    buy_val = float(item.get('buyValue', 0) or 0)
                    sell_val = float(item.get('sellValue', 0) or 0)
                    net_val = float(item.get('netValue', 0) or (buy_val - sell_val))
                    
                    if 'FPI' in category or 'FII' in category:
                        if 'EQUITY' in category or 'CASH' in category:
                            segments['FII_Equity']['buy'] += buy_val
                            segments['FII_Equity']['sell'] += sell_val
                            segments['FII_Equity']['net'] += net_val
                        elif 'DERIVATIVE' in category or 'FUTURES' in category or 'OPTION' in category:
                            segments['FII_Derivatives']['buy'] += buy_val
                            segments['FII_Derivatives']['sell'] += sell_val
                            segments['FII_Derivatives']['net'] += net_val
                    elif 'DII' in category or 'DOMESTIC' in category:
                        if 'EQUITY' in category or 'CASH' in category:
                            segments['DII_Equity']['buy'] += buy_val
                            segments['DII_Equity']['sell'] += sell_val
                            segments['DII_Equity']['net'] += net_val
                        elif 'DERIVATIVE' in category or 'FUTURES' in category or 'OPTION' in category:
                            segments['DII_Derivatives']['buy'] += buy_val
                            segments['DII_Derivatives']['sell'] += sell_val
                            segments['DII_Derivatives']['net'] += net_val
        
        elif isinstance(api_response, dict):
            for key, value in api_response.items():
                if isinstance(value, dict):
                    buy_val = float(value.get('buyValue', value.get('buy', 0)) or 0)
                    sell_val = float(value.get('sellValue', value.get('sell', 0)) or 0)
                    net_val = float(value.get('netValue', value.get('net', buy_val - sell_val)) or (buy_val - sell_val))
                    
                    if 'FPI' in key.upper() or 'FII' in key.upper():
                        if 'EQUITY' in key.upper() or 'CASH' in key.upper():
                            segments['FII_Equity']['buy'] += buy_val
                            segments['FII_Equity']['sell'] += sell_val
                            segments['FII_Equity']['net'] += net_val
                        else:
                            segments['FII_Derivatives']['buy'] += buy_val
                            segments['FII_Derivatives']['sell'] += sell_val
                            segments['FII_Derivatives']['net'] += net_val
                    elif 'DII' in key.upper() or 'DOMESTIC' in key.upper():
                        if 'EQUITY' in key.upper() or 'CASH' in key.upper():
                            segments['DII_Equity']['buy'] += buy_val
                            segments['DII_Equity']['sell'] += sell_val
                            segments['DII_Equity']['net'] += net_val
                        else:
                            segments['DII_Derivatives']['buy'] += buy_val
                            segments['DII_Derivatives']['sell'] += sell_val
                            segments['DII_Derivatives']['net'] += net_val
        
        return segments if any(s['net'] != 0 for s in segments.values()) else None
    except Exception as e:
        print(f"Parse error: {e}")
        return None

# Technical Indicators
def calculate_sma(data, period):
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.rolling(window=period, min_periods=1).mean()

def calculate_ema(data, period):
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.ewm(span=period, adjust=False, min_periods=1).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_supertrend(df, period=10, multiplier=3):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            if direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
    return supertrend, direction

def calculate_ichimoku(df):
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    chikou_span = df['close'].shift(-26)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_fibonacci_levels(df, lookback=50):
    recent_data = df.tail(lookback)
    high = recent_data['high'].max()
    low = recent_data['low'].min()
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }
    return levels, high, low

def calculate_obv(df):
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

def calculate_cmf(df, period=20):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf

def detect_candlestick_patterns(df):
    patterns = []
    if len(df) < 3:
        return patterns
    current = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    body_current = abs(current['close'] - current['open'])
    body_prev1 = abs(prev1['close'] - prev1['open'])
    if body_current < (current['high'] - current['low']) * 0.1:
        patterns.append(("Doji", "Neutral", "Indecision in market"))
    lower_shadow = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
    upper_shadow = current['high'] - current['close'] if current['close'] > current['open'] else current['high'] - current['open']
    if lower_shadow > body_current * 2 and upper_shadow < body_current * 0.3:
        if current['close'] < prev1['close']:
            patterns.append(("Hammer", "Bullish", "Potential reversal at bottom"))
    if upper_shadow > body_current * 2 and lower_shadow < body_current * 0.3:
        if current['close'] > prev1['close']:
            patterns.append(("Shooting Star", "Bearish", "Potential reversal at top"))
    if (current['close'] > current['open'] and prev1['close'] < prev1['open'] and current['open'] < prev1['close'] and current['close'] > prev1['open']):
        patterns.append(("Bullish Engulfing", "Bullish", "Strong reversal signal"))
    if (current['close'] < current['open'] and prev1['close'] > prev1['open'] and current['open'] > prev1['close'] and current['close'] < prev1['open']):
        patterns.append(("Bearish Engulfing", "Bearish", "Strong reversal signal"))
    if prev2 is not None:
        body_prev2 = abs(prev2['close'] - prev2['open'])
        if (prev2['close'] < prev2['open'] and body_prev1 < body_prev2 * 0.3 and current['close'] > current['open'] and current['close'] > (prev2['open'] + prev2['close']) / 2):
            patterns.append(("Morning Star", "Bullish", "Strong reversal pattern"))
        if (prev2['close'] > prev2['open'] and body_prev1 < body_prev2 * 0.3 and current['close'] < current['open'] and current['close'] < (prev2['open'] + prev2['close']) / 2):
            patterns.append(("Evening Star", "Bearish", "Strong reversal pattern"))
    return patterns

def detect_chart_patterns(df, window=20):
    patterns = []
    if len(df) < window:
        return patterns
    recent = df.tail(window)
    highs = recent['high']
    lows = recent['low']
    closes = recent['close']
    if len(recent) >= 5:
        mid_point = len(recent) // 2
        left_shoulder = highs.iloc[:mid_point-1].max()
        head = highs.iloc[mid_point-1:mid_point+2].max()
        right_shoulder = highs.iloc[mid_point+2:].max()
        if head > left_shoulder * 1.02 and head > right_shoulder * 1.02:
            if abs(left_shoulder - right_shoulder) / left_shoulder < 0.02:
                patterns.append(("Head & Shoulders", "Bearish", "Major reversal pattern"))
    peaks = []
    for i in range(2, len(recent)-2):
        if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
            peaks.append((i, highs.iloc[i]))
    if len(peaks) >= 2:
        last_two_peaks = peaks[-2:]
        if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
            patterns.append(("Double Top", "Bearish", "Reversal pattern"))
    recent_highs = highs.tail(10)
    recent_lows = lows.tail(10)
    high_trend = (recent_highs.iloc[-1] - recent_highs.iloc[0]) / recent_highs.iloc[0]
    low_trend = (recent_lows.iloc[-1] - recent_lows.iloc[0]) / recent_lows.iloc[0]
    if abs(high_trend) < 0.02 and low_trend > 0.03:
        patterns.append(("Ascending Triangle", "Bullish", "Continuation pattern"))
    elif high_trend < -0.03 and abs(low_trend) < 0.02:
        patterns.append(("Descending Triangle", "Bearish", "Continuation pattern"))
    elif abs(high_trend) < 0.02 and abs(low_trend) < 0.02:
        if (recent_highs.max() - recent_lows.min()) / recent_lows.min() < 0.05:
            patterns.append(("Symmetrical Triangle", "Neutral", "Breakout expected"))
    return patterns

def get_options_chain(symbol, expiry_date):
    try:
        kite = st.session_state.kite
        instruments_nfo = get_instruments_nfo()
        if instruments_nfo is None:
            return None
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        today = datetime.now(IST).date()
        days_to_expiry = max((expiry_dt - today).days, 1)
        options_data = instruments_nfo[
            (instruments_nfo['name'] == symbol) &
            (instruments_nfo['expiry'] == expiry_dt) &
            (instruments_nfo['instrument_type'].isin(['CE', 'PE']))
        ].copy()
        if options_data.empty:
            symbol_variations = [symbol, symbol.replace('-', ''), symbol.replace('&', '')]
            for var in symbol_variations:
                options_data = instruments_nfo[
                    (instruments_nfo['name'] == var) &
                    (instruments_nfo['expiry'] == expiry_dt) &
                    (instruments_nfo['instrument_type'].isin(['CE', 'PE']))
                ].copy()
                if not options_data.empty:
                    break
        if options_data.empty:
            return None
        spot_price = get_spot_price(symbol)
        if not spot_price:
            spot_price = 0
        symbols_list = [f"NFO:{ts}" for ts in options_data['tradingsymbol'].tolist()]
        chunk_size = 500
        all_quotes = {}
        for i in range(0, len(symbols_list), chunk_size):
            chunk = symbols_list[i:i + chunk_size]
            quotes = kite.quote(chunk)
            all_quotes.update(quotes)
        options_data['ltp'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('last_price', 0)
        )
        options_data['volume'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('volume', 0)
        )
        options_data['oi'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('oi', 0)
        )
        options_data['iv'] = options_data.apply(
            lambda row: calculate_iv(
                row['ltp'], spot_price, row['strike'], 
                days_to_expiry, option_type=row['instrument_type']
            ), axis=1
        )
        return options_data
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None

def get_yahoo_live_price(symbol):
    try:
        yahoo_symbol = YAHOO_SYMBOLS.get(symbol, f"{symbol}.NS")
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(period='1d', interval='1m')
        if not data.empty:
            latest = data.iloc[-1]
            return {
                'ltp': latest['Close'],
                'open': data.iloc[0]['Open'],
                'high': data['High'].max(),
                'low': data['Low'].min(),
                'volume': data['Volume'].sum(),
                'change': latest['Close'] - data.iloc[0]['Open'],
                'timestamp': datetime.now(IST)
            }
        return None
    except Exception as e:
        print(f"Error fetching Yahoo data for {symbol}: {e}")
        return None

def get_yahoo_intraday_data(symbol, days=1):
    try:
        yahoo_symbol = YAHOO_SYMBOLS.get(symbol, f"{symbol}.NS")
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(period=f'{days}d', interval='1m')
        if not data.empty:
            data = data.between_time('09:15', '15:30')
            return data
        return None
    except Exception as e:
        print(f"Error fetching Yahoo intraday data for {symbol}: {e}")
        return None

def start_websocket(symbols):
    try:
        kite = st.session_state.kite
        instruments_nse = get_instruments_nse()
        if instruments_nse is None:
            return False
        tokens_map = {}
        for symbol in symbols:
            result = instruments_nse[instruments_nse['tradingsymbol'] == symbol]
            if not result.empty:
                tokens_map[symbol] = result.iloc[0]['instrument_token']
        if not tokens_map:
            return False
        tokens = list(tokens_map.values())
        kws = KiteTicker(API_KEY, st.session_state.access_token)
        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick['instrument_token']
                symbol = None
                for sym, tok in tokens_map.items():
                    if tok == token:
                        symbol = sym
                        break
                if symbol:
                    st.session_state.live_data[symbol] = {
                        'ltp': tick.get('last_price', 0),
                        'change': tick.get('change', 0),
                        'volume': tick.get('volume', 0),
                        'oi': tick.get('oi', 0),
                        'timestamp': datetime.now(IST),
                        'high': tick.get('ohlc', {}).get('high', 0),
                        'low': tick.get('ohlc', {}).get('low', 0),
                        'open': tick.get('ohlc', {}).get('open', 0),
                        'close': tick.get('ohlc', {}).get('close', 0)
                    }
        def on_connect(ws, response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            st.session_state.ticker_active = True
        def on_close(ws, code, reason):
            st.session_state.ticker_active = False
        def on_error(ws, code, reason):
            pass
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        st.session_state.kws = kws
        def run_websocket():
            kws.connect(threaded=True)
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        return True
    except Exception as e:
        st.error(f"WebSocket Error: {e}")
        return False

def stop_websocket():
    try:
        if st.session_state.kws:
            st.session_state.kws.close()
            st.session_state.ticker_active = False
            st.session_state.kws = None
    except:
        pass

# Main Dashboard
profile = st.session_state.profile
col1, col2 = st.columns([3, 1])
with col1:
    st.success(f"✅ Connected | User: {profile.get('user_name', 'N/A')}")
with col2:
    if st.button("🔓 Logout", key="logout"):
        stop_websocket()
        st.session_state.kite_connected = False
        st.session_state.kite = None
        st.rerun()

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ Options Chain", "💹 Charts & Indicators", "🔴 LIVE Monitor", "📊 Portfolio", "💰 FII/DII Data"])

# TAB 1: OPTIONS CHAIN
with tab1:
    st.header("⚡ Options Chain Analysis")
    st.caption("📊 Real-time Call & Put Options Data with IV | Market Hours: 9:15 AM - 3:30 PM IST")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_stock_oc = st.selectbox("Select Stock", FNO_STOCKS, key="options_stock")
    with col2:
        today = datetime.now(IST).date()
        def get_last_tuesday(year, month):
            if month == 12:
                last_day = datetime(year, month, 31).date()
            else:
                next_month = datetime(year, month + 1, 1).date()
                last_day = next_month - timedelta(days=1)
            days_to_subtract = (last_day.weekday() - 1) % 7
            last_tuesday = last_day - timedelta(days=days_to_subtract)
            return last_tuesday
        expiries = []
        current_year = today.year
        current_month = today.month
        for i in range(6):
            month = current_month + i
            year = current_year
            if month > 12:
                month = month - 12
                year += 1
            expiry = get_last_tuesday(year, month)
            if expiry >= today:
                expiries.append(expiry.strftime("%Y-%m-%d"))
        if len(expiries) < 6:
            for i in range(6, 12):
                month = current_month + i
                year = current_year
                if month > 12:
                    month = month - 12
                    year += 1
                expiry = get_last_tuesday(year, month)
                expiries.append(expiry.strftime("%Y-%m-%d"))
        selected_expiry = st.selectbox(
            "Expiry Date", expiries,
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%d %b %Y"),
            key="options_expiry"
        )
    with col3:
        if st.button("🔄 Refresh", key="refresh_options"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner(f"Loading options chain for {selected_stock_oc}..."):
        spot_price = get_spot_price(selected_stock_oc)
        if spot_price:
            st.info(f"💹 **Spot Price:** ₹{spot_price:.2f}")
            options_df = get_options_chain(selected_stock_oc, selected_expiry)
            if options_df is not None and not options_df.empty:
                ce_data = options_df[options_df['instrument_type'] == 'CE'].copy()
                pe_data = options_df[options_df['instrument_type'] == 'PE'].copy()
                ce_data = ce_data.sort_values('strike')
                pe_data = pe_data.sort_values('strike')
                strikes = sorted(options_df['strike'].unique())
                spot_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
                start_idx = max(0, spot_idx - 10)
                end_idx = min(len(strikes), spot_idx + 11)
                filtered_strikes = strikes[start_idx:end_idx]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_ce_oi = ce_data[ce_data['strike'].isin(filtered_strikes)]['oi'].sum()
                    st.metric("Total CALL OI", f"{total_ce_oi:,.0f}")
                with col2:
                    total_pe_oi = pe_data[pe_data['strike'].isin(filtered_strikes)]['oi'].sum()
                    st.metric("Total PUT OI", f"{total_pe_oi:,.0f}")
                with col3:
                    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
                    pcr_signal = "🟢 Bullish" if pcr > 1.2 else "🔴 Bearish" if pcr < 0.8 else "⚪ Neutral"
                    st.metric("Put-Call Ratio", f"{pcr:.2f}", pcr_signal)
                with col4:
                    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
                    st.metric("ATM Strike", f"₹{atm_strike:.0f}")
                st.markdown("---")
                st.subheader(f"Options Chain - {selected_stock_oc}")
                chain_data = []
                for strike in filtered_strikes:
                    ce_row = ce_data[ce_data['strike'] == strike]
                    pe_row = pe_data[pe_data['strike'] == strike]
                    row = {
                        'CE OI': ce_row['oi'].values[0] if not ce_row.empty else 0,
                        'CE Vol': ce_row['volume'].values[0] if not ce_row.empty else 0,
                        'CE IV': ce_row['iv'].values[0] if not ce_row.empty else 0,
                        'CE LTP': ce_row['ltp'].values[0] if not ce_row.empty else 0,
                        'Strike': strike,
                        'PE LTP': pe_row['ltp'].values[0] if not pe_row.empty else 0,
                        'PE IV': pe_row['iv'].values[0] if not pe_row.empty else 0,
                        'PE Vol': pe_row['volume'].values[0] if not pe_row.empty else 0,
                        'PE OI': pe_row['oi'].values[0] if not pe_row.empty else 0,
                    }
                    chain_data.append(row)
                chain_df = pd.DataFrame(chain_data)
                def highlight_options_chain(row):
                    styles = [''] * len(row)
                    if row['Strike'] == atm_strike:
                        styles = ['background-color: #FFFACD; font-weight: bold; color: #000000'] * len(row)
                    strike_idx = row.index.get_loc('Strike')
                    if row['Strike'] == atm_strike:
                        styles[strike_idx] = 'background-color: #FFFFFF; font-weight: bold; color: #000000; border-left: 2px solid #999; border-right: 2px solid #999;'
                    else:
                        styles[strike_idx] = 'background-color: #FFFFFF; color: #000000; border-left: 2px solid #999; border-right: 2px solid #999;'
                    return styles
                styled_df = chain_df.style.apply(highlight_options_chain, axis=1).format({
                    'CE OI': '{:,.0f}',
                    'CE Vol': '{:,.0f}',
                    'CE IV': '{:.2f}%',
                    'CE LTP': '₹{:.2f}',
                    'Strike': '₹{:.0f}',
                    'PE LTP': '₹{:.2f}',
                    'PE IV': '{:.2f}%',
                    'PE Vol': '{:,.0f}',
                    'PE OI': '{:,.0f}'
                })
                st.dataframe(styled_df, use_container_width=True, height=600)
                st.caption("💡 **ATM Strike** highlighted in faint yellow | **Strike Price** column in white | **IV** calculated using Black-Scholes")
                st.markdown("---")
                st.subheader("📊 Open Interest Distribution")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['CE OI'], name='CALL OI', marker_color='#ef5350', opacity=0.7))
                fig_oi.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['PE OI'], name='PUT OI', marker_color='#26a69a', opacity=0.7))
                fig_oi.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text=f"Spot: ₹{spot_price:.2f}")
                fig_oi.update_layout(title="Call vs Put OI", xaxis_title="Strike (₹)", yaxis_title="OI", height=400, barmode='group')
                st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.warning(f"❌ No options data for {selected_stock_oc} on {selected_expiry}")
        else:
            st.error("❌ Unable to fetch spot price")

# TAB 2: CHARTS
with tab2:
    st.header("Stock Charts with Technical Indicators")
    st.caption("📊 Advanced charting with Ichimoku, Fibonacci, Volume indicators & Pattern Recognition")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_stock = st.selectbox("Select Stock", FNO_STOCKS, key="chart_stock")
    with col2:
        period = st.selectbox("Period", ["1 Day", "1 Week", "2 Weeks", "1 Month", "3 Months"], key="chart_period")
        days_map = {"1 Day": 1, "1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
        days = days_map[period]
    with col3:
        if period == "1 Day":
            default_intervals = ["5minute", "15minute", "30minute", "60minute"]
            interval = st.selectbox("Interval", default_intervals, format_func=lambda x: {"60minute": "60 Min", "30minute": "30 Min", "15minute": "15 Min", "5minute": "5 Min"}[x], key="chart_interval")
        else:
            interval = st.selectbox("Interval", ["day", "60minute", "30minute", "15minute", "5minute"], format_func=lambda x: {"day": "Daily", "60minute": "60 Min", "30minute": "30 Min", "15minute": "15 Min", "5minute": "5 Min"}[x], key="chart_interval")
    with col4:
        chart_type = st.selectbox("Chart Type", ["Candlestick + MA", "Ichimoku Cloud", "Fibonacci", "Volume Analysis"], key="chart_type")
    if interval != "day" and days > 30:
        days = 7
        st.info("📌 Intraday data limited to 1 week")
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = fetch_historical_data(selected_stock, days, interval)
        if df is not None and not df.empty:
            df['EMA_9'] = calculate_ema(df['close'], 9)
            df['EMA_21'] = calculate_ema(df['close'], 21)
            df['EMA_50'] = calculate_ema(df['close'], 50)
            df['SMA_20'] = calculate_sma(df['close'], 20)
            df['SMA_50'] = calculate_sma(df['close'], 50)
            df['SMA_200'] = calculate_sma(df['close'], 200)
            df['RSI'] = calculate_rsi(df['close'])
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
            df['Supertrend'], df['ST_direction'] = calculate_supertrend(df)
            df['Tenkan'], df['Kijun'], df['SpanA'], df['SpanB'], df['Chikou'] = calculate_ichimoku(df)
            df['OBV'] = calculate_obv(df)
            df['CMF'] = calculate_cmf(df)
            candlestick_patterns = detect_candlestick_patterns(df)
            chart_patterns = detect_chart_patterns(df)
            current = df['close'].iloc[-1]
            prev = df['close'].iloc[0]
            change = current - prev
            change_pct = (change / prev) * 100
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current", f"₹{current:.2f}")
            with col2:
                st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
            with col3:
                st.metric("High", f"₹{df['high'].max():.2f}")
            with col4:
                st.metric("Low", f"₹{df['low'].min():.2f}")
            if candlestick_patterns or chart_patterns:
                st.markdown("---")
                st.subheader("🎯 Pattern Recognition")
                if candlestick_patterns:
                    st.markdown("**📊 Candlestick Patterns Detected:**")
                    cols = st.columns(len(candlestick_patterns))
                    for idx, (pattern_name, signal, description) in enumerate(candlestick_patterns):
                        with cols[idx]:
                            color = "🟢" if signal == "Bullish" else "🔴" if signal == "Bearish" else "⚪"
                            st.markdown(f"{color} **{pattern_name}**")
                            st.caption(f"{signal} - {description}")
                if chart_patterns:
                    st.markdown("**📈 Chart Patterns Detected:**")
                    cols = st.columns(len(chart_patterns))
                    for idx, (pattern_name, signal, description) in enumerate(chart_patterns):
                        with cols[idx]:
                            color = "🟢" if signal == "Bullish" else "🔴" if signal == "Bearish" else "⚪"
                            st.markdown(f"{color} **{pattern_name}**")
                            st.caption(f"{signal} - {description}")
            st.markdown("---")
            if interval != 'day':
                df_plot = df.copy()
                df_plot.index = df_plot.index.strftime('%d %b %H:%M')
                x_data = df_plot.index
                xaxis_type = 'category'
                tickformat = None
            else:
                df_plot = df.copy()
                x_data = df.index
                xaxis_type = 'date'
                tickformat = '%d %b %Y'
            if chart_type == "Candlestick + MA":
                st.subheader(f"📊 {selected_stock} - Price Chart with Moving Averages")
                fig_candle = go.Figure()
                fig_candle.add_trace(go.Candlestick(
                    x=x_data, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'],
                    name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ))
                fig_candle.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_9'], name='EMA 9', line=dict(color='#4CAF50', width=1.5), mode='lines'))
                fig_candle.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_21'], name='EMA 21', line=dict(color='#FF9800', width=1.5), mode='lines'))
                fig_candle.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_50'], name='EMA 50', line=dict(color='#9C27B0', width=1.5), mode='lines'))
                fig_candle.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_20'], name='SMA 20', line=dict(color='#FF5722', width=1.5, dash='dash'), mode='lines'))
                fig_candle.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_50'], name='SMA 50', line=dict(color='#FFC107', width=1.5, dash='dash'), mode='lines'))
                if len(df_plot) >= 200:
                    fig_candle.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_200'], name='SMA 200', line=dict(color='#795548', width=2, dash='dash'), mode='lines'))
                fig_candle.update_layout(
                    title=f"{selected_stock} - Candlestick Chart with Moving Averages", yaxis_title="Price (₹)", xaxis_title="Time (IST)",
                    height=650, xaxis_rangeslider_visible=False, hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="gray", borderwidth=1),
                    xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15, showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', autorange=True)
                )
                st.plotly_chart(fig_candle, use_container_width=True)
                st.info("💡 **Tip:** EMA lines (solid) react faster to price changes than SMA lines (dashed)")
                if interval != "day":
                    st.markdown("---")
                    st.subheader(f"📊 {selected_stock} - Intraday Price Movement")
                    fig_intraday = go.Figure()
                    price_min = df['close'].min()
                    price_max = df['close'].max()
                    price_range = price_max - price_min
                    y_padding = max(price_range * 0.05, price_min * 0.002)
                    fig_intraday.add_trace(go.Scatter(
                        x=x_data, y=df_plot['close'], mode='lines', name='Close Price', line=dict(color='#2196F3', width=2),
                        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'
                    ))
                    fig_intraday.update_layout(
                        title=f"{selected_stock} - Intraday Price Trend", yaxis_title="Price (₹)", xaxis_title="Time (IST)", height=400,
                        hovermode='x unified',
                        xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15, showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        yaxis=dict(range=[price_min - y_padding, price_max + y_padding], showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    )
                    st.plotly_chart(fig_intraday, use_container_width=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Intraday High", f"₹{df['high'].max():.2f}")
                    with col2:
                        st.metric("Intraday Low", f"₹{df['low'].min():.2f}")
                    with col3:
                        st.metric("Day Range", f"₹{df['high'].max() - df['low'].min():.2f}")
                    with col4:
                        st.metric("Avg Volume", f"{df['volume'].mean():,.0f}")
            elif chart_type == "Ichimoku Cloud":
                st.subheader(f"☁️ {selected_stock} - Ichimoku Cloud")
                fig_ichi = go.Figure()
                fig_ichi.add_trace(go.Candlestick(x=x_data, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
                fig_ichi.add_trace(go.Scatter(x=x_data, y=df_plot['Tenkan'], name='Tenkan-sen (9)', line=dict(color='#FF6B6B', width=1.5), mode='lines'))
                fig_ichi.add_trace(go.Scatter(x=x_data, y=df_plot['Kijun'], name='Kijun-sen (26)', line=dict(color='#4ECDC4', width=1.5), mode='lines'))
                fig_ichi.add_trace(go.Scatter(x=x_data, y=df_plot['SpanA'], name='Senkou Span A', line=dict(color='rgba(0, 255, 0, 0.3)', width=0.5), mode='lines', showlegend=True))
                fig_ichi.add_trace(go.Scatter(x=x_data, y=df_plot['SpanB'], name='Senkou Span B', line=dict(color='rgba(255, 0, 0, 0.3)', width=0.5), fill='tonexty', fillcolor='rgba(124, 252, 0, 0.2)', mode='lines', showlegend=True))
                fig_ichi.add_trace(go.Scatter(x=x_data, y=df_plot['Chikou'], name='Chikou Span', line=dict(color='#9B59B6', width=1.5, dash='dot'), mode='lines'))
                fig_ichi.update_layout(title=f"{selected_stock} - Ichimoku Cloud Analysis", yaxis_title="Price (₹)", xaxis_title="Time (IST)", height=650, xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
                st.plotly_chart(fig_ichi, use_container_width=True)
                st.info("💡 **Ichimoku Tips:** Price above cloud = Bullish | Price below cloud = Bearish | Tenkan-Kijun cross = Signal")
            elif chart_type == "Fibonacci":
                st.subheader(f"📐 {selected_stock} - Fibonacci Retracement")
                fib_levels, fib_high, fib_low = calculate_fibonacci_levels(df)
                fig_fib = go.Figure()
                fig_fib.add_trace(go.Candlestick(x=x_data, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
                colors = ['#FF0000', '#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#00FF00', '#0000FF']
                for idx, (level_name, level_value) in enumerate(fib_levels.items()):
                    fig_fib.add_hline(y=level_value, line_dash="dash", line_color=colors[idx % len(colors)], annotation_text=f"Fib {level_name} (₹{level_value:.2f})", annotation_position="right")
                fig_fib.update_layout(title=f"{selected_stock} - Fibonacci Retracement Levels", yaxis_title="Price (₹)", xaxis_title="Time (IST)", height=650, xaxis_rangeslider_visible=False, hovermode='x unified', xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
                st.plotly_chart(fig_fib, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Swing High", f"₹{fib_high:.2f}")
                with col2:
                    st.metric("Swing Low", f"₹{fib_low:.2f}")
                st.info("💡 **Fibonacci Tips:** 0.382, 0.5, 0.618 are key retracement levels")
            elif chart_type == "Volume Analysis":
                st.subheader(f"📊 {selected_stock} - Volume Analysis (OBV & CMF)")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Candlestick(x=x_data, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Price', yaxis='y', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
                colors = ['#26a69a' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else '#ef5350' for i in range(len(df_plot))]
                fig_vol.add_trace(go.Bar(x=x_data, y=df_plot['volume'], name='Volume', yaxis='y2', marker_color=colors, opacity=0.5))
                fig_vol.update_layout(title=f"{selected_stock} - Price and Volume", yaxis_title="Price (₹)", yaxis2=dict(title="Volume", overlaying='y', side='right'), xaxis_title="Time (IST)", height=450, xaxis_rangeslider_visible=False, hovermode='x unified', xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15))
                st.plotly_chart(fig_vol, use_container_width=True)
                st.markdown("**On-Balance Volume (OBV)**")
                fig_obv = go.Figure()
                fig_obv.add_trace(go.Scatter(x=x_data, y=df_plot['OBV'], name='OBV', line=dict(color='#2196F3', width=2), fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'))
                fig_obv.update_layout(title="On-Balance Volume Indicator", yaxis_title="OBV", xaxis_title="Time (IST)", height=300, hovermode='x unified', xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=12))
                st.plotly_chart(fig_obv, use_container_width=True)
                st.markdown("**Chaikin Money Flow (CMF)**")
                fig_cmf = go.Figure()
                fig_cmf.add_trace(go.Scatter(x=x_data, y=df_plot['CMF'], name='CMF', line=dict(color='#9C27B0', width=2), fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)'))
                fig_cmf.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                fig_cmf.add_hline(y=0.2, line_dash="dash", line_color="green", annotation_text="Strong Buying")
                fig_cmf.add_hline(y=-0.2, line_dash="dash", line_color="red", annotation_text="Strong Selling")
                fig_cmf.update_layout(title="Chaikin Money Flow Indicator", yaxis_title="CMF Value", xaxis_title="Time (IST)", height=300, hovermode='x unified', xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=12))
                st.plotly_chart(fig_cmf, use_container_width=True)
                st.info("💡 **Volume Tips:** OBV confirms trends | CMF > 0.2 = Buying | CMF < -0.2 = Selling")
            st.subheader("📊 Bollinger Bands (20, 2)")
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=x_data, y=df_plot['BB_upper'], name='Upper Band', line=dict(color='#ef5350', width=1, dash='dash'), mode='lines'))
            fig_bb.add_trace(go.Scatter(x=x_data, y=df_plot['BB_middle'], name='Middle Band (SMA 20)', line=dict(color='#FFC107', width=2), mode='lines'))
            fig_bb.add_trace(go.Scatter(x=x_data, y=df_plot['BB_lower'], name='Lower Band', line=dict(color='#26a69a', width=1, dash='dash'), mode='lines', fill='tonexty', fillcolor='rgba(156, 39, 176, 0.1)'))
            fig_bb.add_trace(go.Scatter(x=x_data, y=df_plot['close'], name='Close Price', line=dict(color='#2196F3', width=2), mode='lines'))
            fig_bb.update_layout(title="Bollinger Bands Analysis", yaxis_title="Price (₹)", xaxis_title="Time (IST)", height=450, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="gray", borderwidth=1), xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
            st.plotly_chart(fig_bb, use_container_width=True)
            st.info("💡 **Tip:** Price touching upper band = resistance, lower band = support")
            st.subheader("🎯 Supertrend (10, 3)")
            fig_st = go.Figure()
            fig_st.add_trace(go.Candlestick(x=x_data, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350', showlegend=False))
            if 'ST_direction' in df_plot.columns:
                i = 0
                while i < len(df_plot):
                    if pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == 1:
                        start_idx = i
                        while i < len(df_plot) and pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == 1:
                            i += 1
                        end_idx = i
                        fig_st.add_trace(go.Scatter(x=x_data[start_idx:end_idx], y=df_plot['Supertrend'].iloc[start_idx:end_idx], name='Buy Signal', line=dict(color='#4CAF50', width=2), mode='lines', showlegend=(start_idx == 0)))
                    elif pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == -1:
                        start_idx = i
                        while i < len(df_plot) and pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == -1:
                            i += 1
                        end_idx = i
                        fig_st.add_trace(go.Scatter(x=x_data[start_idx:end_idx], y=df_plot['Supertrend'].iloc[start_idx:end_idx], name='Sell Signal', line=dict(color='#ef5350', width=2), mode='lines', showlegend=(start_idx == 0)))
                    else:
                        i += 1
            fig_st.update_layout(title="Supertrend Indicator", yaxis_title="Price (₹)", xaxis_title="Time (IST)", height=450, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=15, rangeslider_visible=False), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
            st.plotly_chart(fig_st, use_container_width=True)
            st.subheader("📊 RSI (Relative Strength Index)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=x_data, y=df_plot['RSI'], name='RSI', line=dict(color='#9C27B0', width=2), fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)', visible=True))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="right")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="right")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)", annotation_position="right")
            fig_rsi.update_layout(title="RSI Indicator", yaxis_title="RSI Value", xaxis_title="Time (IST)", height=300, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=12), yaxis=dict(range=[0, 100], showgrid=True))
            st.plotly_chart(fig_rsi, use_container_width=True)
            st.subheader("📈 MACD (Moving Average Convergence Divergence)")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=x_data, y=df_plot['MACD'], name='MACD', line=dict(color='#2196F3', width=2)))
            fig_macd.add_trace(go.Scatter(x=x_data, y=df_plot['MACD_signal'], name='Signal', line=dict(color='#FF5722', width=2)))
            colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in df_plot['MACD_hist']]
            fig_macd.add_trace(go.Bar(x=x_data, y=df_plot['MACD_hist'], name='Histogram', marker_color=colors_macd, opacity=0.5))
            fig_macd.update_layout(title="MACD Indicator", yaxis_title="MACD Value", xaxis_title="Time (IST)", height=300, hovermode='x unified', xaxis=dict(type=xaxis_type, tickformat=tickformat, tickangle=-45, nticks=12), yaxis=dict(showgrid=True))
            st.plotly_chart(fig_macd, use_container_width=True)
            st.markdown("---")
            st.subheader("🎯 Technical Signals Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**📊 Moving Averages**")
                current_price = df['close'].iloc[-1]
                signals = []
                if not pd.isna(df['EMA_9'].iloc[-1]) and current_price > df['EMA_9'].iloc[-1]:
                    signals.append("✅ Above EMA 9")
                else:
                    signals.append("❌ Below EMA 9")
                if not pd.isna(df['EMA_21'].iloc[-1]) and current_price > df['EMA_21'].iloc[-1]:
                    signals.append("✅ Above EMA 21")
                else:
                    signals.append("❌ Below EMA 21")
                if not pd.isna(df['SMA_50'].iloc[-1]) and current_price > df['SMA_50'].iloc[-1]:
                    signals.append("✅ Above SMA 50")
                else:
                    signals.append("❌ Below SMA 50")
                for signal in signals:
                    st.caption(signal)
            with col2:
                st.markdown("**📈 RSI Signal**")
                current_rsi = df['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    if current_rsi > 70:
                        st.error(f"🔴 Overbought: {current_rsi:.2f}")
                    elif current_rsi < 30:
                        st.success(f"🟢 Oversold: {current_rsi:.2f}")
                    else:
                        st.info(f"⚪ Neutral: {current_rsi:.2f}")
                else:
                    st.caption("Calculating...")
            with col3:
                st.markdown("**📊 MACD Signal**")
                current_macd = df['MACD'].iloc[-1]
                current_signal = df['MACD_signal'].iloc[-1]
                if not pd.isna(current_macd) and not pd.isna(current_signal):
                    if current_macd > current_signal:
                        st.success("🟢 Bullish Crossover")
                    else:
                        st.error("🔴 Bearish Crossover")
                    st.caption(f"MACD: {current_macd:.2f}")
                    st.caption(f"Signal: {current_signal:.2f}")
                else:
                    st.caption("Calculating...")
            with col4:
                st.markdown("**🎯 Bollinger & Supertrend**")
                bb_upper = df['BB_upper'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    if current_price > bb_upper:
                        st.caption("🔴 Above BB Upper")
                    elif current_price < bb_lower:
                        st.caption("🟢 Below BB Lower")
                    else:
                        st.caption("⚪ Within BB Range")
                st_direction = df['ST_direction'].iloc[-1]
                if not pd.isna(st_direction):
                    if st_direction == 1:
                        st.success("🟢 Supertrend BUY")
                    else:
                        st.error("🔴 Supertrend SELL")
                else:
                    st.caption("Calculating...")
        else:
            st.error(f"❌ No data for {selected_stock}")

# TAB 3: LIVE MONITOR
with tab3:
    st.header("🔴 LIVE Intraday Monitor (Yahoo Finance)")
    st.caption("⏰ Market Hours: 9:15 AM - 3:30 PM IST")
    now = datetime.now(IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_open = market_open <= now <= market_close and now.weekday() < 5
    col1, col2, col3 = st.columns(3)
    with col1:
        watchlist = st.multiselect("Select Stocks (max 8)", FNO_STOCKS, default=["RELIANCE", "TCS", "HDFCBANK", "INFY"], max_selections=8, key="live_stocks")
    with col2:
        if is_market_open:
            st.success("✅ Market is OPEN")
        else:
            st.error("❌ Market is CLOSED")
        if now.weekday() >= 5:
            st.info("📅 Weekend")
        elif now < market_open:
            st.info(f"⏰ Opens at 9:15 AM")
        else:
            st.info(f"⏰ Closed at 3:30 PM")
    with col3:
        if st.button("🔴 Start Live Monitor", key="start_yahoo_live"):
            st.session_state.live_monitor_running = True
            st.rerun()
        if st.button("⏹ Stop Monitor", key="stop_yahoo_live"):
            st.session_state.live_monitor_running = False
            st.session_state.yahoo_live_data = {}
            st.rerun()
    st.markdown("---")
    if st.session_state.live_monitor_running and watchlist:
        st.success(f"🔴 LIVE: Monitoring {len(watchlist)} stocks")
        num_cols = 2 if len(watchlist) <= 4 else 3
        num_rows = (len(watchlist) + num_cols - 1) // num_cols
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                stock_idx = row * num_cols + col_idx
                if stock_idx < len(watchlist):
                    symbol = watchlist[stock_idx]
                    with col:
                        yahoo_data = get_yahoo_live_price(symbol)
                        if yahoo_data:
                            st.session_state.yahoo_live_data[symbol] = yahoo_data
                            ltp = yahoo_data['ltp']
                            change = yahoo_data['change']
                            change_pct = (change / yahoo_data['open'] * 100) if yahoo_data['open'] > 0 else 0
                            arrow = "🟢" if change >= 0 else "🔴"
                            st.markdown(f"### {arrow} **{symbol}**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("LTP", f"₹{ltp:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                            with col_b:
                                st.metric("Volume", f"{yahoo_data['volume']:,.0f}")
                            st.caption(f"**O:** ₹{yahoo_data['open']:.2f} | **H:** ₹{yahoo_data['high']:.2f} | **L:** ₹{yahoo_data['low']:.2f}")
                            st.caption(f"⏱ {yahoo_data['timestamp'].strftime('%H:%M:%S IST')}")
                        else:
                            st.info(f"⏳ Loading {symbol}...")
        st.markdown("---")
        st.subheader("📈 Intraday Chart (Live)")
        selected_for_chart = st.selectbox("Select stock for chart:", watchlist, key="chart_selection")
        if selected_for_chart:
            with st.spinner(f"Loading intraday chart..."):
                intraday_df = get_yahoo_intraday_data(selected_for_chart, days=1)
                if intraday_df is not None and not intraday_df.empty:
                    fig_intraday = go.Figure()
                    y_min = intraday_df['Close'].min()
                    y_max = intraday_df['Close'].max()
                    y_range = y_max - y_min
                    y_padding = max(y_range * 0.05, y_min * 0.002)
                    fig_intraday.add_trace(go.Scatter(x=intraday_df.index, y=intraday_df['Close'], mode='lines', name='Price', line=dict(color='#2196F3', width=2), fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'))
                    fig_intraday.update_layout(title=f"{selected_for_chart} - Live Intraday", xaxis_title="Time (IST)", yaxis_title="Price (₹)", height=400, hovermode='x unified', yaxis=dict(range=[y_min - y_padding, y_max + y_padding], showgrid=True), xaxis=dict(showgrid=True))
                    st.plotly_chart(fig_intraday, use_container_width=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current", f"₹{intraday_df['Close'].iloc[-1]:.2f}")
                    with col2:
                        day_change = intraday_df['Close'].iloc[-1] - intraday_df['Open'].iloc[0]
                        st.metric("Day Change", f"₹{day_change:.2f}")
                    with col3:
                        st.metric("Day High", f"₹{intraday_df['High'].max():.2f}")
                    with col4:
                        st.metric("Day Low", f"₹{intraday_df['Low'].min():.2f}")
        time.sleep(5)
        st.rerun()
    elif watchlist and not st.session_state.live_monitor_running:
        st.info("👆 Click 'Start Live Monitor'")

# TAB 4: PORTFOLIO
with tab4:
    st.header("📊 Your Portfolio")
    try:
        kite = st.session_state.kite
        st.subheader("💼 Holdings")
        holdings = kite.holdings()
        if holdings:
            df_holdings = pd.DataFrame(holdings)
            st.dataframe(df_holdings[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], use_container_width=True, height=400)
        else:
            st.info("📭 No holdings")
        st.markdown("---")
        st.subheader("📈 Open Positions")
        positions = kite.positions()
        if positions and positions.get('net'):
            df_positions = pd.DataFrame(positions['net'])
            df_positions = df_positions[df_positions['quantity'] != 0]
            if not df_positions.empty:
                st.dataframe(df_positions[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], use_container_width=True, height=300)
            else:
                st.info("📭 No open positions")
        else:
            st.info("📭 No open positions")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# TAB 5: FII/DII
with tab5:
    st.header("💰 FII/DII Data - Daily Activity")
    st.caption("📊 Foreign & Domestic Institutional Investors")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_fii_dii"):
            st.cache_data.clear()
            st.rerun()
    with st.spinner("Fetching FII/DII data..."):
        fii_dii_raw = fetch_fii_dii_latest()
        if fii_dii_raw:
            segments = parse_fii_dii_response(fii_dii_raw)
            if segments:
                st.success("✅ Data loaded")
                fii_total_buy = segments['FII_Equity']['buy'] + segments['FII_Derivatives']['buy']
                fii_total_sell = segments['FII_Equity']['sell'] + segments['FII_Derivatives']['sell']
                fii_total_net = segments['FII_Equity']['net'] + segments['FII_Derivatives']['net']
                dii_total_buy = segments['DII_Equity']['buy'] + segments['DII_Derivatives']['buy']
                dii_total_sell = segments['DII_Equity']['sell'] + segments['DII_Derivatives']['sell']
                dii_total_net = segments['DII_Equity']['net'] + segments['DII_Derivatives']['net']
                total_net = fii_total_net + dii_total_net
                st.markdown("---")
                st.subheader("📊 Market Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    fii_color = "🟢" if fii_total_net >= 0 else "🔴"
                    st.markdown(f"### {fii_color} FII/FPI")
                    st.metric("Net", f"₹{abs(fii_total_net):.2f} Cr", delta=f"{'Inflow' if fii_total_net >= 0 else 'Outflow'}", delta_color="normal" if fii_total_net >= 0 else "inverse")
                    st.caption(f"Buy: ₹{fii_total_buy:.2f} Cr")
                    st.caption(f"Sell: ₹{fii_total_sell:.2f} Cr")
                with col2:
                    dii_color = "🟢" if dii_total_net >= 0 else "🔴"
                    st.markdown(f"### {dii_color} DII")
                    st.metric("Net", f"₹{abs(dii_total_net):.2f} Cr", delta=f"{'Inflow' if dii_total_net >= 0 else 'Outflow'}", delta_color="normal" if dii_total_net >= 0 else "inverse")
                    st.caption(f"Buy: ₹{dii_total_buy:.2f} Cr")
                    st.caption(f"Sell: ₹{dii_total_sell:.2f} Cr")
                with col3:
                    total_color = "🟢" if total_net >= 0 else "🔴"
                    st.markdown(f"### {total_color} Combined")
                    st.metric("Net", f"₹{abs(total_net):.2f} Cr", delta=f"{'Net Buy' if total_net >= 0 else 'Net Sell'}", delta_color="normal" if total_net >= 0 else "inverse")
                    st.caption(f"Buy: ₹{fii_total_buy + dii_total_buy:.2f} Cr")
                    st.caption(f"Sell: ₹{fii_total_sell + dii_total_sell:.2f} Cr")
                st.markdown("---")
                st.subheader("📊 Segment Analysis")
                seg_tabs = st.tabs(["📈 Overview", "💵 Equity", "🔄 Derivatives"])
                with seg_tabs[0]:
                    segments_list = ['Equity', 'Derivatives']
                    fii_seg_data = [segments['FII_Equity']['net'], segments['FII_Derivatives']['net']]
                    dii_seg_data = [segments['DII_Equity']['net'], segments['DII_Derivatives']['net']]
                    fig_seg = go.Figure()
                    fig_seg.add_trace(go.Bar(name='FII/FPI', x=segments_list, y=fii_seg_data, marker_color='#2196F3', text=[f"₹{v:.0f} Cr" for v in fii_seg_data], textposition='outside'))
                    fig_seg.add_trace(go.Bar(name='DII', x=segments_list, y=dii_seg_data, marker_color='#FF9800', text=[f"₹{v:.0f} Cr" for v in dii_seg_data], textposition='outside'))
                    fig_seg.add_hline(y=0, line_dash="solid", line_color="gray")
                    fig_seg.update_layout(title="FII vs DII by Segment", height=400, barmode='group')
                    st.plotly_chart(fig_seg, use_container_width=True)
                    st.markdown("---")
                    if fii_total_net > 0 and dii_total_net > 0:
                        st.success("🟢 **Bullish:** Both FII and DII are buyers")
                    elif fii_total_net < 0 and dii_total_net < 0:
                        st.error("🔴 **Bearish:** Both FII and DII are sellers")
                    elif fii_total_net > 0:
                        st.info("🔵 **Mixed:** FII buying, DII selling")
                    else:
                        st.warning("🟡 **Mixed:** FII selling, DII buying")
                with seg_tabs[1]:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("FII Equity", f"₹{segments['FII_Equity']['net']:.2f} Cr")
                    with col2:
                        st.metric("DII Equity", f"₹{segments['DII_Equity']['net']:.2f} Cr")
                    with col3:
                        st.metric("Total", f"₹{segments['FII_Equity']['net'] + segments['DII_Equity']['net']:.2f} Cr")
                    st.markdown("---")
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Bar(x=['FII/FPI', 'DII'], y=[segments['FII_Equity']['buy'], segments['DII_Equity']['buy']], name='Buy', marker_color='#4CAF50'))
                    fig_eq.add_trace(go.Bar(x=['FII/FPI', 'DII'], y=[segments['FII_Equity']['sell'], segments['DII_Equity']['sell']], name='Sell', marker_color='#ef5350'))
                    fig_eq.update_layout(title="Equity: Buy vs Sell", height=400, barmode='group')
                    st.plotly_chart(fig_eq, use_container_width=True)
                with seg_tabs[2]:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("FII Derivatives", f"₹{segments['FII_Derivatives']['net']:.2f} Cr")
                    with col2:
                        st.metric("DII Derivatives", f"₹{segments['DII_Derivatives']['net']:.2f} Cr")
                    with col3:
                        st.metric("Total", f"₹{segments['FII_Derivatives']['net'] + segments['DII_Derivatives']['net']:.2f} Cr")
                    st.markdown("---")
                    fig_der = go.Figure()
                    fig_der.add_trace(go.Bar(x=['FII/FPI', 'DII'], y=[segments['FII_Derivatives']['buy'], segments['DII_Derivatives']['buy']], name='Buy', marker_color='#4CAF50'))
                    fig_der.add_trace(go.Bar(x=['FII/FPI', 'DII'], y=[segments['FII_Derivatives']['sell'], segments['DII_Derivatives']['sell']], name='Sell', marker_color='#ef5350'))
                    fig_der.update_layout(title="Derivatives: Buy vs Sell", height=400, barmode='group')
                    st.plotly_chart(fig_der, use_container_width=True)
            else:
                st.warning("❌ Unable to parse FII/DII data")
        else:
            st.error("❌ Unable to fetch FII/DII")
            st.info("Try again later")

st.markdown("---")
st.caption("🔴 Dashboard by Zerodha Kite Connect API")
st.caption("⚠️ For educational purposes only")
st.caption(f"📅 {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
