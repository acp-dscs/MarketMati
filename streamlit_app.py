import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from plotly import graph_objs as go
#for PBProphet
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

# Digital Assets Dictionary of Images and extra info for user
crypto_data = [
    {"ticker": "BTC-USD", "name": "Bitcoin", "max_supply": "21,000,000",
     "description": "Often referred to as Digital Gold. A pioneer crypto using blockchain for decentralised digital currency without intermediaries.",
     "image": "https://assets.coingecko.com/coins/images/1/standard/bitcoin.png?1696501400"},
     {"ticker": "MSTR", "name": "MicroStrategy", "max_supply": "Infinite",
     "description": "Bitcoin Treasury Company (BTC) In August 2020, MicroStrategy became the first publicly traded US company to acquire and hold bitcoin on its balance sheet as a primary treasury reserve asset. ",
     "image": "https://raw.githubusercontent.com/acp-dscs/MarketMativ1/main/assets/MSTR.png"},
    {"ticker": "ETH-USD", "name": "Ethereum", "max_supply": "Infinite",
     "description": "Proof-of-Stake blockchain for dApps, scaling with Layer 2 solutions.",
     "image": "https://assets.coingecko.com/coins/images/279/standard/ethereum.png?1696501628"},
    {"ticker": "BNB-USD", "name": "Binance Chain", "max_supply": "200,000,000",
     "description": "Native Binance Smart Chain coin, reduces trading fees.",
     "image": "https://assets.coingecko.com/coins/images/825/standard/bnb-icon2_2x.png?1696501970"},
    {"ticker": "SOL-USD", "name": "Solana", "max_supply": "Infinite",
     "description": "Fast Layer 1 blockchain with smart contracts, Proof-of-History and Stake.",
     "image": "https://assets.coingecko.com/coins/images/4128/standard/solana.png?1718769756"},
    {"ticker": "XRP-USD", "name": "XRP", "max_supply": "100,000,000,000",
     "description": "Facilitates global payments via XRPL ledger for banks and providers.",
     "image": "https://assets.coingecko.com/coins/images/44/standard/xrp-symbol-white-128.png?1696501442"},
    {"ticker": "ADA-USD", "name": "Cardano", "max_supply": "45,000,000,000",
     "description": "Proof-of-Stake blockchain, supports dApps, with a multi-asset ledger and smart contracts.",
     "image": "https://assets.coingecko.com/coins/images/975/standard/cardano.png?1696502090"},
    {"ticker": "DOT-USD", "name": "Polkadot", "max_supply": "Infinite",
     "description": "Builds decentralised oracle networks for secure blockchain smart contracts.",
     "image": "https://static.coingecko.com/s/polkadot-73b0c058cae10a2f076a82dcade5cbe38601fad05d5e6211188f09eb96fa4617.gif"},
    {"ticker": "LINK-USD", "name": "Chainlink", "max_supply": "1,000,000,000",
     "description": "Layer-0 platform linking chains, pooled security, diverse protocols.",
     "image": "https://assets.coingecko.com/coins/images/877/standard/chainlink-new-logo.png?1696502009"},
    {"ticker": "MATIC-USD", "name": "Polygon", "max_supply": "10,000,000,000",
     "description": "The first well-structured, easy-to-use platform for Ethereum scaling.",
     "image": "https://assets.coingecko.com/coins/images/4713/standard/polygon.png?1698233745"},
    {"ticker": "ZEC-USD", "name": "Zcash", "max_supply": "21,000,000",
     "description": "Zcash (ZEC), based on Bitcoin's codebase, uses zk-SNARKs to offer optional anonymity through shielded transactions. As the first major application of this zero-knowledge cryptography, Zcash ensures privacy by encrypting shielded transactions while still validating them under network rules.",
     "image": "https://assets.coingecko.com/coins/images/486/standard/circle-zcash-color.png?1696501740"}
]
crypto_dict = {crypto['ticker']: crypto for crypto in crypto_data}

# MarketMati Streamlit Program Main Code
mme_url = 'https://raw.githubusercontent.com/acp-dscs/MarketMativ1/main/assets/MMEYE.png'
st.image(mme_url, use_container_width=True)

# Fetch data from Yahoo Finance
def fetch_yf_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    data = data['Close'].reset_index()
    data = data.melt(id_vars=['Date'], var_name='ticker', value_name='close_price')
    return data

# Fetch live prices and previous day's close
def fetch_live_prices(tickers):
    live_data = []
    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)
        hist_data = ticker_data.history(period="5d")  # Fetch last 5 days of data
        if len(hist_data) >= 2:  # Ensure we have at least two days of data
            live_price = hist_data['Close'].iloc[-1]  # Most recent price
            prev_close = hist_data['Close'].iloc[-2]  # Second most recent price
            percent_change = ((live_price - prev_close) / prev_close) * 100  # % Change
            live_data.append({
                "ticker": ticker,
                "current_price": live_price,
                "prev_close": prev_close,
                "percent_change": percent_change,
            })
        else:
            live_data.append({
                "ticker": ticker,
                "current_price": None,
                "prev_close": None,
                "percent_change": None,
            })
    return pd.DataFrame(live_data)

# Prepare the data
live_prices = fetch_live_prices([crypto["ticker"] for crypto in crypto_data])

# Prepare data for heatmap visualization
z = np.array([
    live_prices['percent_change'].fillna(0).tolist()  # % Change for heatmap
])

text = np.array([
    [f"{row['ticker']}<br>Price: {row['current_price']:.2f} USD<br>% Change: {row['percent_change']:.2f}%"
     if pd.notnull(row['percent_change']) else f"{row['ticker']}: Data N/A"
     for _, row in live_prices.iterrows()]
])

colors = np.array([
    ['green' if row['percent_change'] > 0 else 'red' if row['percent_change'] < 0 else 'white'
     for _, row in live_prices.iterrows()]
])

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=z,
    text=text,
    texttemplate="%{text}",
    colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
    showscale=True,  # Show color scale for better context
    zmin=-5,  # Adjust for typical percentage range
    zmax=5,
))

# Heatmap Title
st.markdown('<h1 style="color: green;">Digital Assets & MSTR</h1>', unsafe_allow_html=True)
st.write('Accurate to YFinance Market Data')
# Add title and layout adjustments
fig.update_layout(
    title="Heatmap: Expand to see - Current Price Vs Previous Day Close",
    xaxis=dict(
        title="",
        tickmode="array",
        tickvals=np.arange(len(live_prices)),
        ticktext=live_prices['ticker'].tolist(),
    ),
    yaxis=dict(
        title="",
        tickmode="array",
        tickvals=[0],
        ticktext=[""],
    ),
)

# Display in Streamlit
st.plotly_chart(fig)

# Interactive Section for User
st.markdown('<h1 style="color: green;">Cryptocurrency Deep Dive</h1>', unsafe_allow_html=True)
st.subheader('Top Picks and Analysis')
selected_ticker = st.selectbox('Cryptocurrency tickers:', [crypto['ticker'] for crypto in crypto_data])
selected_crypto = crypto_dict[selected_ticker]
st.image(selected_crypto['image'])
st.subheader(selected_crypto['name'])
st.write(f"Max Supply: {selected_crypto['max_supply']}")
st.text(selected_crypto['description'])

# Fetch historical data
historical_data = fetch_yf_data([selected_ticker], "2010-01-01", date.today().strftime('%Y-%m-%d'))
historical_data = historical_data[historical_data['ticker'] == selected_ticker]
historical_data['111SMA'] = historical_data['close_price'].rolling(window=111, min_periods=1).mean()
historical_data['350SMA'] = historical_data['close_price'].rolling(window=350, min_periods=1).mean()
historical_data['PiCycle'] = historical_data['350SMA'] * 2

# Historical Prices and Pi Cycle Chart
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['close_price'], name=f"{selected_ticker} Day Close", line=dict(color='green', width=1.5)))
    fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['111SMA'], name='111-day SMA', line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['PiCycle'], name='350-day SMA', line=dict(color='red', width=1.5)))
    fig.layout.update(title_text=f"{selected_ticker} Price with Pi Cycle Indicator", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Calculate and display with Plotly Chart monthly percentage changes, use end of month close price
columns_to_drop = ['open_price', 'high_price', 'low_price', '111SMA', '350SMA', 'PiCycle']
filtered_data_subset = historical_data.drop(columns=[col for col in columns_to_drop if col in historical_data])
monthly_last_rows = filtered_data_subset.groupby(filtered_data_subset['Date'].dt.to_period('M')).last().reset_index(drop=True)
monthly_last_rows['month_percentage_change'] = monthly_last_rows['close_price'].pct_change() * 100
def colour_neg_red(value):
    if value > 0:
        return 'green'
    elif value < 0:
        return 'red'
    else:
        return 'grey'

st.markdown('<h1 style="color: green;">Monthly Percentage Changes</h1>', unsafe_allow_html=True)

if selected_ticker:
    data_selected = monthly_last_rows[monthly_last_rows['ticker'] == selected_ticker]
    fig = go.Figure()
    for index, row in data_selected.iterrows():
        month = row['Date'].strftime('%Y-%m')
        change = row['month_percentage_change']
        if not pd.isnull(change):
            colour = colour_neg_red(change)
            fig.add_trace(go.Bar(
                x=[month],
                y=[change],
                marker_color=colour,
                name=row['Date'].strftime('%b %Y'),
                text=f"{change:.2f}%",
                hoverinfo='text'
            ))
    fig.update_layout(
        title=f'Monthly Percentage Changes for {selected_ticker}',
        xaxis_title='',
        yaxis_title='Percentage Change (%)',
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig)

# Allow user to view underlying data
if st.checkbox('Expand Monthly Percentages Data', key='checkbox_raw_monthly_last_rows'):
    st.write(monthly_last_rows)

# --- Add Annual Candlestick Section ---
# --- Add Annual Candlestick Section ---
# Hardcoded data for Bitcoin from 2008 to 2013
hardcoded_btc_data = {
    'Year': [2008, 2009, 2010, 2011, 2012, 2013],
    'Year Open': [0.001, 0.39, 0.10, 0.30, 5.27, 13.30],
    'Year High': [0.39, 1.00, 0.39, 31.91, 13.70, 1156.00],
    'Year Low': [0.001, 0.01, 0.06, 2.05, 4.41, 13.30],
    'Year Close': [0.39, 0.10, 0.30, 4.70, 13.30, 755.00],
    'Digital Asset': ['BTC-USD'] * 6
}

# Convert hardcoded data into a DataFrame for Bitcoin
btc_data_df = pd.DataFrame(hardcoded_btc_data)

# Fetch annual candlestick data for the selected crypto
def fetch_annual_candles(tickers, start_date="2010-01-01"):
    annual_data = []
    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)  # Fetch data from Yahoo Finance
        history = ticker_data.history(start=start_date, interval="1mo")  # Fetch monthly data from start date
        history.reset_index(inplace=True)
        history['Year'] = history['Date'].dt.year  # Extract year from Date
        yearly_summary = history.groupby('Year').agg(
            Open=('Open', 'first'),
            High=('High', 'max'),
            Low=('Low', 'min'),
            Close=('Close', 'last')
        ).reset_index()
        yearly_summary['ticker'] = ticker
        annual_data.append(yearly_summary)
    return pd.concat(annual_data, ignore_index=True)

# Fetch annual candlestick data for the selected tickers
tickers = [selected_ticker]  # Only show the selected ticker in the interactive section
annual_candles_data = fetch_annual_candles(tickers, start_date="2014-01-01")

# Rename and display the dataframe with relevant columns
annual_candles_data.rename(columns={
    'ticker': 'Digital Asset',
    'Open': 'Year Open',
    'High': 'Year High',
    'Low': 'Year Low',
    'Close': 'Year Close'
}, inplace=True)

# Combine the hardcoded BTC data with the fetched data only if the selected ticker is BTC-USD
if selected_ticker == 'BTC-USD':
    combined_data = pd.concat([btc_data_df, annual_candles_data], ignore_index=True)
else:
    combined_data = annual_candles_data

# Ensure 'Year' column is formatted as YYYY
combined_data['Year'] = combined_data['Year'].astype(int)

# Create a Plotly table for display with a more neutral green theme
fig = go.Figure(data=[go.Table(
    header=dict(
        values=list(combined_data.columns),
        fill_color='darkgreen',  # Darker green header for a more professional look
        font=dict(color='white', size=12),  # White text for header
        align='center'
    ),
    cells=dict(
        values=[combined_data[col] for col in combined_data.columns],
        fill_color=['#E8F5E9' if i % 2 == 0 else '#C8E6C9' for i in range(len(combined_data))],  # Subtle alternating green shades
        font=dict(color='black', size=12),  # Black text for readability
        align='center'
    )
)])

# Show the Plotly table
st.markdown('<h2 style="color: green;">Annual Data</h2>', unsafe_allow_html=True)
st.plotly_chart(fig)


#Start of FBProphet section

# New Section Header: Display logo MarketMati
mm_url = 'https://raw.githubusercontent.com/acp-dscs/MarketMati/main/assets/MM.png'
st.image(mm_url, use_container_width=True)  # Changed to use_container_width
# FBProphet Title
st.markdown('<h2 style="color: green;">Time Series Forecasting - META FB Prophet</h2>', unsafe_allow_html=True)

# Forecasting in years with slider for user, up to ten years
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
n_years = st.slider('Move the toggle to gain an insight into what the future holds for this cryptocurrency:', 1, 10)
period = n_years * 365

# Filter the historical data for the selected ticker
filtered_data = historical_data[historical_data['ticker'] == selected_ticker]

# Forecasting with Prophet
df_forecast_train = filtered_data[['Date', 'close_price']]  # Ensure 'Date' column is used instead of 'date_price'
df_forecast_train = df_forecast_train.rename(columns={"Date": "ds", "close_price": "y"})
m = Prophet()
m.fit(df_forecast_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Prevent the lower interval from becoming negative
forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], 0)
forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)

# Forecasting chart
st.write(f'**Forecasting {selected_ticker} for {n_years} year{"s" if n_years > 1 else ""}**')
fig1 = plot_plotly(m, forecast)
lgreen_line = '#00CC96'
dgreen_marker = '#006400'

# Adjust line and marker colors in the plot
for trace in fig1['data']:
    trace['line']['color'] = lgreen_line
    if 'marker' in trace:
        trace['marker']['color'] = dgreen_marker

fig1.update_layout(
    xaxis=dict(title=''),
    yaxis=dict(title='')
)
st.plotly_chart(fig1)

# Allow user to view underlying data
if st.checkbox('Expand Forecasting Data', key='checkbox_raw_data_forecast'):
    st.write(forecast)

# End of PBProphet section



# Import and display logo MarketMati images from GitHub URL
mmf_url = 'https://raw.githubusercontent.com/acp-dscs/MarketMativ1/main/assets/MarketMati.png'
st.image(mmf_url, use_container_width=True)

# Disclaimer
st.title('')
st.markdown('<h1 style="color: red;">DISCLAIMER - MarketMati</h1>', unsafe_allow_html=True)
st.subheader('IS NOT INVESTMENT ADVICE')
st.write('Use for educational purposes only. Financial investment decisions are your own.')
st.write('**CAUTION: The Digital Assets class is highly volatile.**')
st.write('If you are considering investing in Digital Assets, ensure you **ALWAYS** seek professional advice from a qualified financial advisor.')
st.write('**Credit to sources below:**')
st.write('FB Prophet - Time Series, YFinance API, CoinGecko & Philip Swift - Pi Cycle')
