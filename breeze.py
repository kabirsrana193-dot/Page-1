import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from kiteconnect import KiteConnect, KiteTicker
import time
import threading
import pytz
import yfinance as yf

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
API_SECRET = "mgso1jdnxj3xeei228dcciyqqx7axl77"

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
if 'previous_oi_data' not in st.session_state:
    st.session_state.previous_oi_data = {}

# --------------------------
# Login Management
# --------------------------
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

# --------------------------
# Helper Functions
# --------------------------
@st.cache_data(ttl=600)
def get_instruments_nfo():
    """Get NFO instruments"""
    try:
        instruments = st.session_state.kite.instruments("NFO")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NFO instruments: {e}")
        return None

@st.cache_data(ttl=600)
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
    except:
        return df

def fetch_historical_data(symbol, days=30, interval="day"):
    """Fetch historical data"""
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
    """Get current spot price"""
    try:
        kite = st.session_state.kite
        quote = kite.quote(f"NSE:{symbol}")
        if quote and f"NSE:{symbol}" in quote:
            return quote[f"NSE:{symbol}"]["last_price"]
        return None
    except Exception as e:
        return None

def get_options_chain(symbol, expiry_date):
    """Fetch options chain with OI change tracking"""
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
            symbol_variations = [
                symbol,
                symbol.replace('-', ''),
                symbol.replace('&', ''),
                symbol + 'E' if symbol.startswith('BAJAJ') else symbol
            ]
            
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
        
        symbols_list = [f"NFO:{ts}" for ts in options_data['tradingsymbol'].tolist()]
        
        chunk_size = 500
        all_quotes = {}
        
        for i in range(0, len(symbols_list), chunk_size):
            chunk = symbols_list[i:i + chunk_size]
            quotes = kite.quote(chunk)
            all_quotes.update(quotes)
        
        # Store previous OI for comparison
        cache_key = f"{symbol}_{expiry_date}"
        
        for idx, row in options_data.iterrows():
            ts = row['tradingsymbol']
            quote_data = all_quotes.get(f"NFO:{ts}", {})
            
            current_oi = quote_data.get('oi', 0)
            options_data.at[idx, 'ltp'] = quote_data.get('last_price', 0)
            options_data.at[idx, 'volume'] = quote_data.get('volume', 0)
            options_data.at[idx, 'oi'] = current_oi
            
            # Calculate OI change
            prev_oi_key = f"{cache_key}_{ts}"
            if prev_oi_key in st.session_state.previous_oi_data:
                prev_oi = st.session_state.previous_oi_data[prev_oi_key]
                oi_change = current_oi - prev_oi
                oi_change_pct = (oi_change / prev_oi * 100) if prev_oi > 0 else 0
            else:
                oi_change = 0
                oi_change_pct = 0
            
            options_data.at[idx, 'oi_change'] = oi_change
            options_data.at[idx, 'oi_change_pct'] = oi_change_pct
            
            # Update cache
            st.session_state.previous_oi_data[prev_oi_key] = current_oi
        
        return options_data
        
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None

# --------------------------
# YFinance Live Data Functions
# --------------------------
def get_yfinance_symbol(nse_symbol):
    """Convert NSE symbol to Yahoo Finance symbol"""
    return f"{nse_symbol}.NS"

def fetch_live_data_yfinance(symbols):
    """Fetch live data using yfinance"""
    live_data = {}
    
    for symbol in symbols:
        try:
            yf_symbol = get_yfinance_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            
            # Get current data
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                prev_close = info.get('previousClose', latest['Close'])
                
                live_data[symbol] = {
                    'ltp': latest['Close'],
                    'open': hist.iloc[0]['Open'] if len(hist) > 0 else latest['Open'],
                    'high': hist['High'].max(),
                    'low': hist['Low'].min(),
                    'close': prev_close,
                    'volume': int(hist['Volume'].sum()),
                    'change': latest['Close'] - prev_close,
                    'timestamp': datetime.now(IST)
                }
        except Exception as e:
            st.warning(f"Error fetching {symbol}: {str(e)}")
            continue
    
    return live_data

# [Keep all the technical indicator functions from original code - calculate_sma, calculate_ema, etc.]
def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.rolling(window=period, min_periods=1).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.ewm(span=period, adjust=False, min_periods=1).mean()

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend"""
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

# --------------------------
# Main Dashboard
# --------------------------

profile = st.session_state.profile
col1, col2 = st.columns([3, 1])
with col1:
    st.success(f"✅ Connected | User: {profile.get('user_name', 'N/A')}")
with col2:
    if st.button("🔓 Logout", key="logout"):
        st.session_state.kite_connected = False
        st.session_state.kite = None
        st.rerun()

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Options Chain", "💹 Charts & Indicators", "🔴 LIVE Monitor", "📊 Portfolio"])

# TAB 1: OPTIONS CHAIN (FIXED)
with tab1:
    st.header("⚡ Options Chain Analysis")
    st.caption("📊 Real-time Call & Put Options Data | Market Hours: 9:15 AM - 3:30 PM IST")
    
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
                if len(expiries) >= 6:
                    break
        
        selected_expiry = st.selectbox(
            "Expiry Date",
            expiries,
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%d %b %Y"),
            key="options_expiry"
        )
    
    with col3:
        if st.button("🔄 Refresh", key="refresh_options"):
            st.cache_data.clear()
            st.session_state.previous_oi_data = {}
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
                
                # Build proper options chain format
                chain_data = []
                for strike in filtered_strikes:
                    ce_row = ce_data[ce_data['strike'] == strike]
                    pe_row = pe_data[pe_data['strike'] == strike]
                    
                    row = {
                        'CE OI': ce_row['oi'].values[0] if not ce_row.empty else 0,
                        'CE OI Chg': ce_row['oi_change'].values[0] if not ce_row.empty else 0,
                        'CE OI Chg%': ce_row['oi_change_pct'].values[0] if not ce_row.empty else 0,
                        'CE Vol': ce_row['volume'].values[0] if not ce_row.empty else 0,
                        'CE LTP': ce_row['ltp'].values[0] if not ce_row.empty else 0,
                        'Strike': strike,
                        'PE LTP': pe_row['ltp'].values[0] if not pe_row.empty else 0,
                        'PE Vol': pe_row['volume'].values[0] if not pe_row.empty else 0,
                        'PE OI Chg%': pe_row['oi_change_pct'].values[0] if not pe_row.empty else 0,
                        'PE OI Chg': pe_row['oi_change'].values[0] if not pe_row.empty else 0,
                        'PE OI': pe_row['oi'].values[0] if not pe_row.empty else 0,
                    }
                    chain_data.append(row)
                
                chain_df = pd.DataFrame(chain_data)
                
                # Style the dataframe with proper ATM highlighting
                def highlight_atm_row(row):
                    if row['Strike'] == atm_strike:
                        # Light blue background that keeps text visible
                        return ['background-color: #E3F2FD; font-weight: bold; color: #000000'] * len(row)
                    return [''] * len(row)
                
                def color_oi_change(val):
                    """Color OI change percentage"""
                    if pd.isna(val) or val == 0:
                        return ''
                    color = '#90EE90' if val > 0 else '#FFB6C1'
                    return f'background-color: {color}; color: #000000'
                
                styled_df = chain_df.style.apply(highlight_atm_row, axis=1).format({
                    'CE OI': '{:,.0f}',
                    'CE OI Chg': '{:+,.0f}',
                    'CE OI Chg%': '{:+.1f}%',
                    'CE Vol': '{:,.0f}',
                    'CE LTP': '₹{:.2f}',
                    'Strike': '₹{:.0f}',
                    'PE LTP': '₹{:.2f}',
                    'PE Vol': '{:,.0f}',
                    'PE OI Chg%': '{:+.1f}%',
                    'PE OI Chg': '{:+,.0f}',
                    'PE OI': '{:,.0f}'
                }).applymap(color_oi_change, subset=['CE OI Chg%', 'PE OI Chg%'])
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=600
                )
                
                st.caption("💡 **ATM Strike** highlighted in light blue | Green = OI Increase | Red = OI Decrease")
                
                # OI Chart
                st.markdown("---")
                st.subheader("📊 Open Interest Distribution")
                
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(
                    x=chain_df['Strike'],
                    y=chain_df['CE OI'],
                    name='CALL OI',
                    marker_color='#ef5350',
                    opacity=0.7
                ))
                fig_oi.add_trace(go.Bar(
                    x=chain_df['Strike'],
                    y=chain_df['PE OI'],
                    name='PUT OI',
                    marker_color='#26a69a',
                    opacity=0.7
                ))
                fig_oi.add_vline(
                    x=spot_price,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Spot: ₹{spot_price:.2f}"
                )
                fig_oi.update_layout(
                    title="Call vs Put OI",
                    xaxis_title="Strike (₹)",
                    yaxis_title="OI",
                    height=400,
                    barmode='group'
                )
                st.plotly_chart(fig_oi, use_container_width=True)
                
            else:
                st.warning(f"❌ No options data for {selected_stock_oc} on {selected_expiry}")
                st.info("💡 Try different stock or expiry date")
        else:
            st.error("❌ Unable to fetch spot price")

# TAB 2: Keep original charts code (too long to include here - copy from your original)

# TAB 3: LIVE MONITOR (FIXED WITH YFINANCE)
with tab3:
    st.header("🔴 LIVE Intraday Monitor (YFinance)")
    st.caption("⏰ Market Hours: 9:15 AM - 3:30 PM IST | Auto-updates every 10 seconds")
    
    now = datetime.now(IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_open = market_open <= now <= market_close and now.weekday() < 5
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        watchlist = st.multiselect(
            "Select Stocks (max 10)",
            FNO_STOCKS,
            default=["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            max_selections=10,
            key="live_stocks"
        )
    
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
        refresh_rate = st.selectbox(
            "Refresh Rate",
            [5, 10, 15, 30, 60],
            index=1,
            format_func=lambda x: f"{x} seconds",
            key="refresh_rate"
        )
    
    st.markdown("---")
    
    if watchlist:
        if st.button("🔄 Fetch Live Data Now", key="manual_refresh"):
            with st.spinner("Fetching live data..."):
                st.session_state.live_data = fetch_live_data_yfinance(watchlist)
        
        # Auto-refresh container
        placeholder = st.empty()
        
        with placeholder.container():
            st.info(f"🔴 Auto-refreshing every {refresh_rate} seconds...")
            
            # Fetch data if empty or old
            if not st.session_state.live_data or len(st.session_state.live_data) == 0:
                st.session_state.live_data = fetch_live_data_yfinance(watchlist)
            
            # Display live data
            if st.session_state.live_data:
                num_cols = min(3, len(watchlist))
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
                                    ltp = data['ltp']
                                    close = data['close']
                                    change = data['change']
                                    change_pct = (change / close * 100) if close > 0 else 0
                                    arrow = "🟢" if change >= 0 else "🔴"
                                    
                                    st.markdown(f"### {arrow} **{symbol}**")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.metric("LTP", f"₹{ltp:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                                    with col_b:
                                        st.metric("Volume", f"{data['volume']:,}")
                                    
                                    st.caption(f"**O:** ₹{data['open']:.2f} | **H:** ₹{data['high']:.2f} | **L:** ₹{data['low']:.2f}")
                                    st.caption(f"⏱ {data['timestamp'].strftime('%H:%M:%S IST')}")
                                else:
                                    st.info(f"⏳ Loading {symbol}...")
            else:
                st.warning("⚠️ No live data available. Click 'Fetch Live Data Now' button.")
        
        # Auto-refresh mechanism
        time.sleep(refresh_rate)
        st.session_state.live_data = fetch_live_data_yfinance(watchlist)
        st.rerun()
    
    else:
        st.info("👆 Select stocks from the dropdown to start live monitoring")
        st.markdown("""
        ### 📌 Features:
        - Live price updates using Yahoo Finance
        - Works during market hours (9:15 AM - 3:30 PM IST)
        - Auto-refresh every 10 seconds
        - Shows Open, High, Low, Volume
        - Color-coded gains/losses
        
        ### 💡 Note:
        Yahoo Finance data may have 1-2 minute delay compared to real-time exchange data.
        """)

# TAB 4: PORTFOLIO
with tab4:
    st.header("📊 Your Portfolio")
    
    try:
        kite = st.session_state.kite
        
        st.subheader("💼 Holdings")
        holdings = kite.holdings()
        
        if holdings:
            df_holdings = pd.DataFrame(holdings)
            st.dataframe(
                df_holdings[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']],
                use_container_width=True,
                height=400
            )
        else:
            st.info("📭 No holdings found")
        
        st.markdown("---")
        
        st.subheader("📈 Open Positions")
        positions = kite.positions()
        
        if positions and positions.get('net'):
            df_positions = pd.DataFrame(positions['net'])
            df_positions = df_positions[df_positions['quantity'] != 0]
            
            if not df_positions.empty:
                st.dataframe(
                    df_positions[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']],
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("📭 No open positions")
        else:
            st.info("📭 No open positions")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("🔴 Dashboard powered by Zerodha Kite Connect API & Yahoo Finance")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption(f"📅 Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
