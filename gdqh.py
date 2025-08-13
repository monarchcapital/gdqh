import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import warnings

warnings.filterwarnings("ignore")

# --- QH API Data Function (from QH_api_OHLC.py) ---
# This function is responsible for making the API call to the QH API
# It returns a pandas DataFrame.
def get_qh_api_data(instruments_list, interval='1M', count=10):
    """
    Fetches OHLC data from the QH API for a list of instruments.

    Args:
        instruments_list (list): A list of instrument tickers (e.g., ["ES", "NQ"]).
        interval (str): The time interval for the data (e.g., '1M', '1H', '1D').
        count (int): The number of data points to retrieve.

    Returns:
        pd.DataFrame: A DataFrame with the fetched data or an empty DataFrame on failure.
    """
    # NOTE: You MUST replace 'your_access_token' with your actual authorization token.
    # The provided snippet had a long token, but it's best to handle this securely.
    api_token = "your_access_token_here"
    url = "https://qh-api.corp.hertshtengroup.com/api/v2/ohlc"
    
    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    params = {
        "instruments": ",".join(instruments_list),
        "interval": interval,
        "count": str(count)
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # The API returns a dictionary where keys are instruments.
        # We'll combine them into a single DataFrame.
        all_data = []
        for instrument, ohlc_data in data.items():
            if ohlc_data:
                df = pd.DataFrame(ohlc_data)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.set_index('datetime')
                df['instrument'] = instrument
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data)
        else:
            st.error("No data returned from the QH API.")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from QH API: {e}")
        return pd.DataFrame()


# --- Your existing data.py code starts here ---

# Configuration for Yahoo Finance Symbols
SYMBOLS = {
    'Stock Indices': {
        '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ Composite', '^FTSE': 'FTSE 100 (UK)',
        '^N225': 'Nikkei 225 (Japan)', '^GDAXI': 'DAX 30 (Germany)', '^NSEI': 'Nifty 50 (India)',
        '^BVSP': 'Bovespa (Brazil)', '^MXX': 'IPC Mexico', '^HSI': 'Hang Seng (Hong Kong)',
        '^STOXX50E': 'Euro Stoxx 50', '^SETI': 'SET Index (Thailand)', '^ATH': 'Athex Composite (Greece)',
        'FTSEMIB.MI': 'FTSE MIB (Italy)', '^WIG20': 'WIG20 (Poland)', '^FCHI': 'CAC 40 (France)',
        '^KS11': 'KOSPI (South Korea)', 'IMOEX.ME': 'MOEX (Russia)', '^GSPTSE': 'S&P/TSX (Canada)',
        '^AXJO': 'S&P/ASX 200 (Australia)', '^PSI': 'PSEi (Philippines)'
    },
    'Commodities': {
        'GC=F': 'Gold Futures', 'SI=F': 'Silver Futures', 'CL=F': 'Crude Oil Futures',
        'NG=F': 'Natural Gas Futures', 'BZ=F': 'Brent Crude Futures', 'HG=F': 'Copper Futures'
    },
    'Currencies': {
        'EURUSD=X': 'EUR/USD', 'JPY=X': 'USD/JPY', 'GBPUSD=X': 'GBP/USD',
        'AUDUSD=X': 'AUD/USD', 'CADUSD=X': 'CAD/USD', 'CHFUSD=X': 'CHF/USD'
    },
    'Major Tech Stocks': {
        'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc. (Class A)',
        'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms Inc.', 'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corp.', 'AMD': 'Advanced Micro Devices Inc.', 'NFLX': 'Netflix Inc.'
    }
}

# Configuration for QH API Tickers
QH_TICKERS = {
    "Bonds": ["TU", "YR", "FV", "TY", "TN", "ZB"],
    "Futures": ["6E", "6J", "6B", "6S", "RF", "6C"],
    "Index": ["ES", "YM", "DXY", "FXXP", "FTUK", "FDAX", "JNI", "FDXM", "FESX", "NQ", "DX"],
    "Yields": [
        "DE2YR", "DE3YR", "DE5YR", "DE10YR", "DE30YR",
        "GB2YR", "GB3YR", "GB10YR", "CA2YR", "CA3YR", "CA10YR",
        "BR2YR", "BR5YR", "BR10YR", "US2YR", "US3YR", "US5YR", "US7YR", "US10YR", "US30YR",
        "JP2YR", "JP5YR", "JP10YR"
    ]
}


# Streamlit session state initialization
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.now().date() - timedelta(days=2)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now().date() - timedelta(days=1)
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'tables'


@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_yfinance_data(symbols_dict, start, end):
    data = {}
    for category, symbols in symbols_dict.items():
        tickers = list(symbols.keys())
        df = yf.download(tickers, start=start, end=end, progress=False)
        if not df.empty:
            df.columns = [
                f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
                for col in df.columns.values
            ]
            df = df.reset_index().rename(columns={'Date': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df.set_index('date')
            data[category] = df
    return data


# Define a function to generate styled tables for displaying data
def generate_styled_table(df, title, date_col, value_col):
    st.subheader(title)
    if value_col in df.columns:
        styled_df = df.pivot_table(
            index=df.index, columns=df.index, values=value_col
        ).rename(columns={c: str(c) for c in df.index})
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.write("Data not available for this category.")


def calculate_and_display_changes(df_1, df_2, category):
    st.subheader(f"{category} Change")
    if df_1 is not None and df_2 is not None and not df_1.empty and not df_2.empty:
        try:
            df_1_close = df_1["Close"].reset_index(drop=True)
            df_2_close = df_2["Close"].reset_index(drop=True)
            changes = (
                ((df_2_close - df_1_close) / df_1_close * 100)
                .round(2)
                .to_frame(name="% Change")
            )
            changes = changes.rename(index=dict(zip(df_1["Ticker"].tolist(), df_1["Ticker"].tolist())))
            
            def color_change(val):
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f"color: {color}"

            st.dataframe(
                changes.style.applymap(color_change, subset=["% Change"]),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error calculating changes for {category}: {e}")
    else:
        st.write("Data not available for comparison.")


# Function to fetch and display QH API data
def display_qh_api_data():
    st.header("QH API Data")
    st.write("Fetching the latest data for selected instruments from the QH API.")
    
    selected_category = st.selectbox(
        "Select an Instrument Category:",
        list(QH_TICKERS.keys()),
        key="qh_category_select"
    )
    
    if selected_category:
        instruments = QH_TICKERS[selected_category]
        if instruments:
            # Fetch data using the new function
            qh_df = get_qh_api_data(instruments)
            if not qh_df.empty:
                st.subheader(f"OHLC Data for {selected_category}")
                st.dataframe(qh_df, use_container_width=True)
            else:
                st.write("Could not retrieve data for the selected instruments.")


# --- UI Layout ---
st.set_page_config(layout="wide", page_title="Market Comparison App")
st.title("Market Performance Dashboard")

# Date selection for Yahoo Finance data
date_1 = st.session_state.start_date
date_2 = st.session_state.end_date

# Sidebar for date selection (only for yfinance)
with st.sidebar:
    st.header("Date Range (for Yahoo Finance)")
    st.session_state.start_date = st.date_input("Start Date", st.session_state.start_date)
    st.session_state.end_date = st.date_input("End Date", st.session_state.end_date)
    
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("Error: End Date must be after Start Date.")


col_buttons = st.columns([1, 1, 1, 1, 1])
with col_buttons[0]:
    if st.button('Refresh Data', help="Click to fetch the latest data"):
        st.session_state.end_date = datetime.now().date() - timedelta(days=1)
        st.session_state.start_date = datetime.now().date() - timedelta(days=2)
        st.cache_data.clear()
        st.rerun()

with col_buttons[2]:
    if st.button("Show Tables", help="Display the detailed tables"):
        st.session_state.view_mode = 'tables'
        st.rerun()

with col_buttons[3]:
    if st.button("Show Heatmap", help="Display a heatmap grid of daily percentage changes"):
        st.session_state.view_mode = 'heatmap'
        st.rerun()
        
with col_buttons[4]:
    if st.button("Show QH API Data", help="Display data from the new QH API"):
        st.session_state.view_mode = 'qh_api'
        st.rerun()

# Conditional rendering based on session state
if st.session_state.view_mode == 'tables':
    st.header(f"Market Performance: {date_1.strftime('%Y-%m-%d')} vs {date_2.strftime('%Y-%m-%d')}")
    market_data_dfs = fetch_yfinance_data(SYMBOLS, date_1, date_2)
    for category in SYMBOLS.keys():
        df = market_data_dfs.get(category)
        if df is not None:
            # Assuming you want to display the 'Close' price for a given date
            generate_styled_table(df, f"{category} - Close Prices", 'date', 'Close')

elif st.session_state.view_mode == 'heatmap':
    st.header(f"Daily Percentage Change Heatmap")
    st.write("This feature is not yet fully implemented for the new data.")
    # You would need to add code here to fetch and process data for a heatmap,
    # similar to what was likely in your original data.py for yfinance data.

elif st.session_state.view_mode == 'qh_api':
    display_qh_api_data()

