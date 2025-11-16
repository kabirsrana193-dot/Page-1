"""
Kite Connect F&O Trading Dashboard with WebSocket Live Streaming
Real-time tick data using KiteTicker (Threaded Mode)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import feedparser
from kiteconnect import KiteConnect, KiteTicker
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(
    page_title="F&O Dashboard - Kite Connect (Live)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------
# Configuration
# --------------------------
API_KEY = "aj0gv6rpjm11ecac"
ACCESS_TOKEN = "SmCnbRkg9WhWv7FnF3cXpjEGBJkWqihw"

# --------------------------
# FNO Stocks List
# --------------------------
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "ITC", "SBIN", "HCLTECH", "AXISBANK",
    "KOTAKBANK", "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "TATAMOTORS",
]

# RSS Feeds
FINANCIAL_RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

# --------------------------
# Initialize session state
# --------------------------
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'kite_connected' not in st.session_state:
    st.session_state.kite_connected = False
if 'instruments_df' not in st.session_state:
    st.session_state.instruments_df = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'ticker_running' not in st.session_state:
    st.session_state.ticker_running = False
if 'kws' not in st.session_state:
    st.session_state.kws = None
if 'subscribed_tokens' not in st.session_state:
    st.session_state.subscribed_tokens = {}
if 'token_to_symbol' not in st.session_state:
    st.session_state.token_to_symbol = {}

# --------------------------
# Kite Connection
# --------------------------
@st.cache_resource
def init_kite():
    """Initialize Kite Connect"""
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
        profile = kite.profile()
        return kite, True, profile
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None, False, None

if not st.session_state.kite_connected:
    kite, connected, profile = init_kite()
    st.session_state.kite = kite
    st.session_state.kite_connected = connected
    st.session_state.profile = profile

# --------------------------
# Helper Functions
# --------------------------
@st.cache_data(ttl=300)
def get_instruments():
    """Get and cache instruments list"""
    try:
        instruments = st.session_state.kite.instruments("NSE")
        df = pd.DataFrame(instruments)
        return df
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return None

def get_instrument_token(symbol):
    """Get instrument token for a symbol"""
    if st.session_state.instruments_df is None:
        st.session_state.instruments_df = get_instruments()
    
    if st.session_state.instruments_df is not None:
        result = st.session_state.instruments_df[
            st.session_state.instruments_df['tradingsymbol'] == symbol
        ]
        if not result.empty:
            return result.iloc[0]['instrument_token']
    return None

def get_instrument_tokens(symbols):
    """Get instrument tokens for multiple symbols"""
    tokens = {}
    for symbol in symbols:
        token = get_instrument_token(symbol)
        if token:
            tokens[symbol] = token
    return tokens

# --------------------------
# WebSocket Functions
# --------------------------
def start_ticker(symbols):
    """Start KiteTicker WebSocket connection"""
    try:
        # Stop existing connection if any
        if st.session_state.kws:
            try:
                st.session_state.kws.close()
            except:
                pass
        
        # Get instrument tokens
        tokens_map = get_instrument_tokens(symbols)
        if not tokens_map:
            st.error("❌ Could not get instrument tokens")
            return False
        
        tokens = list(tokens_map.values())
        
        # Create reverse mapping: token -> symbol
        st.session_state.token_to_symbol = {v: k for k, v in tokens_map.items()}
        st.session_state.subscribed_tokens = tokens_map
        
        # Initialize KiteTicker
        kws = KiteTicker(API_KEY, ACCESS_TOKEN)
        
        def on_ticks(ws, ticks):
            """Callback to receive ticks"""
            for tick in ticks:
                token = tick['instrument_token']
                symbol = st.session_state.token_to_symbol.get(token)
                
                if symbol:
                    # Store tick data
                    st.session_state.live_data[symbol] = {
                        'ltp': tick.get('last_price', 0),
                        'volume': tick.get('volume_traded', 0),
                        'oi': tick.get('oi', 0),
                        'timestamp': datetime.now(),
                        'ohlc': tick.get('ohlc', {}),
                        'change': tick.get('change', 0),
                        'last_traded_quantity': tick.get('last_traded_quantity', 0),
                        'average_traded_price': tick.get('average_traded_price', 0),
                        'mode': tick.get('mode', 'quote')
                    }
        
        def on_connect(ws, response):
            """Callback on successful connect"""
            logging.info(f"✅ WebSocket connected: {response}")
            # Subscribe to tokens
            ws.subscribe(tokens)
            # Set to FULL mode for detailed data
            ws.set_mode(ws.MODE_FULL, tokens)
            st.session_state.ticker_running = True
        
        def on_close(ws, code, reason):
            """Callback on connection close"""
            logging.info(f"WebSocket closed: {code} - {reason}")
            st.session_state.ticker_running = False
        
        def on_error(ws, code, reason):
            """Callback on error"""
            logging.error(f"WebSocket error: {code} - {reason}")
        
        def on_reconnect(ws, attempts_count):
            """Callback on reconnection"""
            logging.info(f"WebSocket reconnecting... Attempt: {attempts_count}")
        
        def on_noreconnect(ws):
            """Callback when max reconnection attempts reached"""
            logging.error("WebSocket max reconnection attempts reached")
            st.session_state.ticker_running = False
        
        # Assign callbacks
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        kws.on_reconnect = on_reconnect
        kws.on_noreconnect = on_noreconnect
        
        # Store WebSocket instance
        st.session_state.kws = kws
        
        # Connect in threaded mode (non-blocking)
        kws.connect(threaded=True)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Ticker Error: {e}")
        logging.error(f"Ticker error: {e}")
        return False

def stop_ticker():
    """Stop KiteTicker WebSocket"""
    try:
        if st.session_state.kws:
            st.session_state.kws.close()
            st.session_state.kws = None
            st.session_state.ticker_running = False
            st.session_state.live_data = {}
            logging.info("WebSocket stopped")
            return True
    except Exception as e:
        logging.error(f"Error stopping ticker: {e}")
    return False

# --------------------------
# Technical Indicators
# --------------------------
def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))

def fetch_historical_data(symbol, days=30, interval="day"):
    """Fetch historical data from Kite"""
    try:
        kite = st.session_state.kite
        if not kite:
            return None
        
        instrument_token = get_instrument_token(symbol)
        if not instrument_token:
            return None
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        if data:
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df
        return None
    except:
        return None

# --------------------------
# Sentiment Analysis
# --------------------------
def analyze_sentiment(text):
    POSITIVE = ['surge', 'rally', 'gain', 'profit', 'growth', 'rise', 'bullish', 
                'strong', 'beats', 'outperform', 'jumps', 'soars', 'upgrade']
    NEGATIVE = ['fall', 'drop', 'loss', 'decline', 'weak', 'crash', 'bearish',
                'concern', 'risk', 'plunge', 'slump', 'miss', 'downgrade']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE if w in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

def fetch_news(num_articles=12):
    all_articles = []
    seen_titles = set()
    
    for feed_url, source_name in FINANCIAL_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = getattr(entry, 'title', '')
                if not title or title in seen_titles:
                    continue
                
                sentiment = analyze_sentiment(title)
                
                all_articles.append({
                    "Title": title,
                    "Source": source_name,
                    "Sentiment": sentiment,
                    "Link": entry.link,
                    "Published": getattr(entry, 'published', 'Recent')
                })
                seen_titles.add(title)
                
                if len(all_articles) >= num_articles:
                    break
        except:
            continue
    
    return all_articles[:num_articles]

# --------------------------
# Streamlit App
# --------------------------

st.title("📈 F&O Dashboard - Kite Connect")

# Connection Status
if st.session_state.kite_connected:
    profile = st.session_state.profile
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ Kite API Connected | User: {profile.get('user_name', 'N/A')}")
    with col2:
        if st.session_state.ticker_running:
            st.success("🔴 LIVE")
        else:
            st.info("⚪ Offline")
else:
    st.error("❌ Not connected to Kite API")
    st.stop()

st.markdown("---")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📰 News", "💹 Charts", "🔴 LIVE Monitor"])

# TAB 1: NEWS
with tab1:
    st.header("Market News & Sentiment")
    
    if st.button("🔄 Refresh News"):
        news = fetch_news(12)
        
        if news:
            for article in news:
                sentiment_colors = {"positive": "#28a745", "neutral": "#6c757d", "negative": "#dc3545"}
                sentiment_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
                
                st.markdown(f"**[{article['Title']}]({article['Link']})**")
                st.markdown(
                    f"<span style='background-color: {sentiment_colors[article['Sentiment']]}; "
                    f"color: white; padding: 3px 10px; border-radius: 4px; font-size: 11px;'>"
                    f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()}</span>",
                    unsafe_allow_html=True
                )
                st.caption(f"Source: {article['Source']} | {article['Published']}")
                st.markdown("---")

# TAB 2: CHARTS
with tab2:
    st.header("Stock Charts with Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_stock = st.selectbox("Select Stock", FNO_STOCKS)
    
    with col2:
        period = st.selectbox("Period", ["1 Week", "1 Month", "3 Months"])
        days_map = {"1 Week": 7, "1 Month": 30, "3 Months": 90}
        days = days_map[period]
    
    df = fetch_historical_data(selected_stock, days, "day")
    
    if df is not None and not df.empty:
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['RSI'] = calculate_rsi(df['close'])
        
        current = df['close'].iloc[-1]
        prev = df['close'].iloc[0]
        change = current - prev
        change_pct = (change / prev) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current", f"₹{current:.2f}")
        with col2:
            st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
        with col3:
            st.metric("High", f"₹{df['high'].max():.2f}")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], 
            low=df['low'], close=df['close']
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue')))
        fig.update_layout(title=f"{selected_stock}", height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: LIVE MONITOR
with tab3:
    st.header("🔴 LIVE Intraday Monitor (WebSocket)")
    
    st.info("💡 WebSocket streams work only during market hours (9:15 AM - 3:30 PM, Mon-Fri)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        watchlist = st.multiselect(
            "Select Stocks (max 6)",
            FNO_STOCKS,
            default=["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            max_selections=6,
            key="live_stocks"
        )
    
    with col2:
        if st.button("🔴 Start Live Stream", key="start_ws"):
            if watchlist:
                with st.spinner("Connecting WebSocket..."):
                    if start_ticker(watchlist):
                        st.success("✅ WebSocket Connected!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to connect")
    
    with col3:
        if st.button("⏹️ Stop Stream", key="stop_ws"):
            if stop_ticker():
                st.success("✅ WebSocket Stopped")
                time.sleep(1)
                st.rerun()
    
    # Display status
    if st.session_state.ticker_running:
        st.success(f"🔴 LIVE: Streaming {len(watchlist)} stocks | Auto-refreshing every 2 seconds")
        
        # Auto refresh every 2 seconds
        time.sleep(2)
        st.rerun()
    
    st.markdown("---")
    
    # Display live data
    if watchlist:
        if st.session_state.ticker_running and st.session_state.live_data:
            num_cols = 2 if len(watchlist) <= 4 else 3
            num_rows = (len(watchlist) + num_cols - 1) // num_cols
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx, col in enumerate(cols):
                    stock_idx = row * num_cols + col_idx
                    
                    if stock_idx < len(watchlist):
                        symbol = watchlist[stock_idx]
                        
                        with col:
                            if symbol in st.session_state.live_data:
                                data = st.session_state.live_data[symbol]
                                ohlc = data.get('ohlc', {})
                                
                                ltp = data['ltp']
                                open_price = ohlc.get('open', 0)
                                change = ltp - open_price if open_price > 0 else 0
                                change_pct = (change / open_price * 100) if open_price > 0 else 0
                                
                                arrow = "🟢" if change >= 0 else "🔴"
                                
                                # Display card
                                st.markdown(f"### {arrow} {symbol}")
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("LTP", f"₹{ltp:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                                with col_b:
                                    st.metric("Volume", f"{data['volume']:,}")
                                
                                # OHLC
                                st.caption(
                                    f"O: ₹{ohlc.get('open', 0):.2f} | "
                                    f"H: ₹{ohlc.get('high', 0):.2f} | "
                                    f"L: ₹{ohlc.get('low', 0):.2f} | "
                                    f"C: ₹{ohlc.get('close', 0):.2f}"
                                )
                                
                                # Timestamp
                                timestamp = data['timestamp'].strftime('%H:%M:%S')
                                st.caption(f"⏱️ {timestamp} | Mode: {data.get('mode', 'quote').upper()}")
                            else:
                                st.info(f"⏳ Waiting for {symbol} data...")
        
        elif not st.session_state.ticker_running:
            st.warning("⚪ WebSocket not running. Click 'Start Live Stream' to begin.")
        else:
            st.info("⏳ Connecting and waiting for first tick...")
    
    else:
        st.info("👆 Select stocks to monitor")

# Footer
st.markdown("---")
st.caption("🔴 LIVE Dashboard powered by Zerodha Kite Connect WebSocket API")
st.caption("⚠️ **Disclaimer:** For educational purposes only. Not financial advice.")
if st.session_state.ticker_running:
    st.caption(f"🔴 Live since: {datetime.now().strftime('%H:%M:%S')}")
