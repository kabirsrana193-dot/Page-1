"""
Kite Connect F&O Trading Dashboard with WebSocket Live Streaming
With Token Management System
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from kiteconnect import KiteConnect, KiteTicker
import time
import threading
import pytz

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

# YOU NEED TO ADD YOUR API SECRET HERE (get from Kite Connect app dashboard)
API_SECRET = "mgso1jdnxj3xeei228dcciyqqx7axl77"  # ⚠️ REPLACE THIS

# Indian market hours
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
IST = pytz.timezone('Asia/Kolkata')

# --------------------------
# FNO Stocks List
# --------------------------
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "ITC", "SBIN", "HCLTECH", "AXISBANK",
    "KOTAKBANK", "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "TATAMOTORS",
    "ADANIPORTS", "ADANIENT", "TECHM", "POWERGRID", "NTPC",
    "COALINDIA", "TATASTEEL", "BAJAJFINSV", "HEROMOTOCO", "INDUSINDBK",
    "M&M", "GRASIM", "HINDALCO", "JSWSTEEL", "SBILIFE",
    "ICICIGI", "BAJAJ-AUTO", "HDFCLIFE", "ADANIGREEN", "SHREECEM",
    "EICHERMOT", "UPL", "TATACONSUM", "BRITANNIA", "NESTLEIND",
    "HINDUNILVR", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP"
]

# --------------------------
# Initialize session state
# --------------------------
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'kite_connected' not in st.session_state:
    st.session_state.kite_connected = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'instruments_df' not in st.session_state:
    st.session_state.instruments_df = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'ticker_active' not in st.session_state:
    st.session_state.ticker_active = False
if 'kws' not in st.session_state:
    st.session_state.kws = None

# --------------------------
# Login Management
# --------------------------
st.title("📈 F&O Dashboard - Kite Connect")

if not st.session_state.kite_connected:
    st.header("🔐 Login to Kite Connect")
    
    st.markdown("""
    ### How to get your Access Token:
    1. Click the button below to login to Kite
    2. After login, you'll be redirected to a URL with `request_token`
    3. Copy the `request_token` from the URL
    4. Paste it below and click Generate Access Token
    """)
    
    # Generate login URL
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
                    st.info(f"💾 Save this Access Token for today: `{access_token}`")
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure your API_SECRET is correct in the code")
        elif API_SECRET == "YOUR_API_SECRET_HERE":
            st.error("⚠️ Please set your API_SECRET in the code first!")
            st.info("Get your API Secret from: https://kite.trade/ → My Apps → Your App")
        else:
            st.warning("⚠️ Please enter the request token")
    
    st.markdown("---")
    st.markdown("### OR Use Existing Access Token")
    manual_token = st.text_input("Paste Access Token directly (if you have one):", key="manual_token")
    
    if st.button("🔗 Connect with Access Token", key="connect_token"):
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
                st.error(f"❌ Connection Error: {str(e)}")
                st.info("Your access token might be expired. Generate a new one above.")
        else:
            st.warning("⚠️ Please enter an access token")
    
    st.stop()

# --------------------------
# Helper Functions
# --------------------------
@st.cache_data(ttl=300)
def get_instruments_nfo():
    """Get NFO instruments"""
    try:
        instruments = st.session_state.kite.instruments("NFO")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NFO instruments: {e}")
        return None

@st.cache_data(ttl=300)
def get_instruments_nse():
    """Get NSE instruments"""
    try:
        instruments = st.session_state.kite.instruments("NSE")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NSE instruments: {e}")
        return None

def filter_market_hours(df):
    """Filter dataframe to market hours"""
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
    except Exception as e:
        return df

def fetch_historical_data(symbol, days=30, interval="day"):
    """Fetch historical data"""
    try:
        kite = st.session_state.kite
        if not kite:
            return None
        
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
    """Get current spot price"""
    try:
        kite = st.session_state.kite
        quote = kite.quote(f"NSE:{symbol}")
        
        if quote and f"NSE:{symbol}" in quote:
            return quote[f"NSE:{symbol}"]["last_price"]
        return None
    except Exception as e:
        st.error(f"Error getting spot price: {e}")
        return None

def get_options_chain(symbol, expiry_date):
    """Fetch options chain"""
    try:
        kite = st.session_state.kite
        
        instruments_nfo = get_instruments_nfo()
        if instruments_nfo is None:
            return None
        
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        
        options_data = instruments_nfo[
            (instruments_nfo['name'] == symbol) & 
            (instruments_nfo['expiry'] == expiry_dt) &
            (instruments_nfo['instrument_type'].isin(['CE', 'PE']))
        ].copy()
        
        if options_data.empty:
            return None
        
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
        options_data['bid'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('depth', {}).get('buy', [{}])[0].get('price', 0) if all_quotes.get(f"NFO:{x}", {}).get('depth') else 0
        )
        options_data['ask'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('depth', {}).get('sell', [{}])[0].get('price', 0) if all_quotes.get(f"NFO:{x}", {}).get('depth') else 0
        )
        
        return options_data
        
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None

# --------------------------
# Technical Indicators
# --------------------------
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

# --------------------------
# Main Dashboard
# --------------------------

profile = st.session_state.profile
col1, col2 = st.columns([3, 1])
with col1:
    st.success(f"✅ Connected | User: {profile.get('user_name', 'N/A')} | Email: {profile.get('email', 'N/A')}")
with col2:
    if st.button("🔓 Logout", key="logout"):
        st.session_state.kite_connected = False
        st.session_state.kite = None
        st.session_state.access_token = None
        st.rerun()

st.markdown("---")

# Main tabs
tab1, tab2, tab3 = st.tabs(["⚡ Options Chain", "💹 Charts & Indicators", "📊 Portfolio"])

# TAB 1: OPTIONS CHAIN
with tab1:
    st.header("⚡ Options Chain Analysis")
    st.caption("📊 Real-time Call & Put Options Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock_oc = st.selectbox(
            "Select Stock",
            FNO_STOCKS,
            key="options_stock"
        )
    
    with col2:
        today = datetime.now(IST).date()
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_expiry = today + timedelta(days=days_ahead)
        
        expiries = []
        for i in range(4):
            expiry = next_expiry + timedelta(weeks=i)
            expiries.append(expiry.strftime("%Y-%m-%d"))
        
        selected_expiry = st.selectbox(
            "Expiry Date",
            expiries,
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%d %b %Y"),
            key="options_expiry"
        )
    
    with col3:
        if st.button("🔄 Refresh Options Data", key="refresh_options"):
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
                    st.metric("Put-Call Ratio", f"{pcr:.2f}")
                
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
                        'CE_OI': ce_row['oi'].values[0] if not ce_row.empty else 0,
                        'CE_Volume': ce_row['volume'].values[0] if not ce_row.empty else 0,
                        'CE_LTP': ce_row['ltp'].values[0] if not ce_row.empty else 0,
                        'Strike': strike,
                        'PE_LTP': pe_row['ltp'].values[0] if not pe_row.empty else 0,
                        'PE_Volume': pe_row['volume'].values[0] if not pe_row.empty else 0,
                        'PE_OI': pe_row['oi'].values[0] if not pe_row.empty else 0,
                    }
                    chain_data.append(row)
                
                chain_df = pd.DataFrame(chain_data)
                
                def highlight_atm(row):
                    if row['Strike'] == atm_strike:
                        return ['background-color: #ffffcc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    chain_df.style.apply(highlight_atm, axis=1).format({
                        'CE_OI': '{:,.0f}',
                        'CE_Volume': '{:,.0f}',
                        'CE_LTP': '₹{:.2f}',
                        'Strike': '₹{:.0f}',
                        'PE_LTP': '₹{:.2f}',
                        'PE_Volume': '{:,.0f}',
                        'PE_OI': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=600
                )
                
                st.caption("💡 **ATM Strike** is highlighted in yellow")
                
                # OI Chart
                st.markdown("---")
                st.subheader("📊 Open Interest Distribution")
                
                fig_oi = go.Figure()
                
                fig_oi.add_trace(go.Bar(
                    x=chain_df['Strike'],
                    y=chain_df['CE_OI'],
                    name='CALL OI',
                    marker_color='#ef5350',
                    opacity=0.7
                ))
                
                fig_oi.add_trace(go.Bar(
                    x=chain_df['Strike'],
                    y=chain_df['PE_OI'],
                    name='PUT OI',
                    marker_color='#26a69a',
                    opacity=0.7
                ))
                
                fig_oi.add_vline(
                    x=spot_price,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Spot: ₹{spot_price:.2f}",
                    annotation_position="top"
                )
                
                fig_oi.update_layout(
                    title="Call vs Put Open Interest",
                    xaxis_title="Strike Price (₹)",
                    yaxis_title="Open Interest",
                    height=400,
                    barmode='group',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_oi, use_container_width=True)
                
            else:
                st.warning(f"❌ No options data available for {selected_stock_oc}")
        else:
            st.error("❌ Unable to fetch spot price")

# TAB 2: CHARTS
with tab2:
    st.header("Stock Charts with Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock = st.selectbox("Select Stock", FNO_STOCKS, key="chart_stock")
    
    with col2:
        period = st.selectbox("Period", ["1 Week", "2 Weeks", "1 Month"], key="chart_period")
        days_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}
        days = days_map[period]
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["day", "60minute", "15minute"],
            format_func=lambda x: {"day": "Daily", "60minute": "60 Min", "15minute": "15 Min"}[x],
            key="chart_interval"
        )
    
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = fetch_historical_data(selected_stock, days, interval)
    
    if df is not None and not df.empty:
        df['EMA_9'] = calculate_ema(df['close'], 9)
        df['EMA_21'] = calculate_ema(df['close'], 21)
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['SMA_50'] = calculate_sma(df['close'], 50)
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
        
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
        
        st.markdown("---")
        
        # Candlestick Chart
        st.subheader(f"📊 {selected_stock} - Price Chart")
        
        if interval != 'day':
            df_plot = df.copy()
            df_plot.index = df_plot.index.strftime('%d %b %H:%M')
            x_data = df_plot.index
            xaxis_type = 'category'
        else:
            df_plot = df.copy()
            x_data = df.index
            xaxis_type = 'date'
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=x_data,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price'
        ))
        
        fig.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_9'], name='EMA 9', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_21'], name='EMA 21', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_50'], name='SMA 50', line=dict(color='blue', width=1, dash='dash')))
        
        fig.update_layout(
            title=f"{selected_stock} Chart",
            yaxis_title="Price (₹)",
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            xaxis=dict(type=xaxis_type)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI
        st.subheader("📊 RSI")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=x_data, y=df_plot['RSI'], name='RSI', line=dict(color='purple', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="RSI", height=250, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    else:
        st.error(f"❌ No data available for {selected_stock}")

# TAB 3: PORTFOLIO
with tab3:
    st.header("📊 Your Portfolio")
    
    try:
        kite = st.session_state.kite
        
        st.subheader("💼 Holdings")
        holdings = kite.holdings()
        
        if holdings:
            df_holdings = pd.DataFrame(holdings)
            st.dataframe(df_holdings[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], 
                        use_container_width=True)
        else:
            st.info("📭 No holdings found")
        
        st.markdown("---")
        
        st.subheader("📈 Open Positions")
        positions = kite.positions()
        
        if positions and positions.get('net'):
            df_positions = pd.DataFrame(positions['net'])
            df_positions = df_positions[df_positions['quantity'] != 0]
            
            if not df_positions.empty:
                st.dataframe(df_positions[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], 
                            use_container_width=True)
            else:
                st.info("📭 No open positions")
        else:
            st.info("📭 No open positions")
        
    except Exception as e:
        st.error(f"❌ Error fetching portfolio data: {e}")

# Footer
st.markdown("---")
st.caption("🔴 Dashboard powered by Zerodha Kite Connect API")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption(f"📅 Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
