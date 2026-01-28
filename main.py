import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Gold Price Forecast Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan menarik
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #FFF;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
        transform: scale(1.05);
    }
    .api-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=180)  # Cache 3 menit
def get_current_gold_price():
    """Mengambil harga emas terkini dari berbagai API gratis"""
    apis_tried = []
    
    # 1. Gold-API.com (NO KEY NEEDED - Best option)
    try:
        apis_tried.append("Gold-API.com")
        response = requests.get('https://www.gold-api.com/api/XAU/USD', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'price' in data:
                return {
                    'price': data['price'],
                    'source': 'Gold-API.com',
                    'timestamp': data.get('timestamp', datetime.now().isoformat()),
                    'success': True
                }
    except Exception as e:
        st.warning(f"Gold-API.com gagal: {str(e)}")
    
    # 2. API Ninjas (Need free API key - sign up at api-ninjas.com)
    # Uncomment and add your key if you want to use it
    # try:
    #     apis_tried.append("API-Ninjas")
    #     headers = {'X-Api-Key': 'YOUR_API_KEY_HERE'}
    #     response = requests.get('https://api.api-ninjas.com/v1/goldprice', headers=headers, timeout=5)
    #     if response.status_code == 200:
    #         data = response.json()
    #         return {
    #             'price': data['price'],
    #             'source': 'API-Ninjas',
    #             'timestamp': data.get('timestamp', datetime.now().isoformat()),
    #             'success': True
    #         }
    # except:
    #     pass
    
    # 3. Fallback ke Yahoo Finance
    try:
        apis_tried.append("Yahoo Finance")
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="1d")
        if len(hist) > 0:
            return {
                'price': hist['Close'].iloc[-1],
                'source': 'Yahoo Finance (Futures)',
                'timestamp': hist.index[-1].isoformat(),
                'success': True
            }
    except Exception as e:
        st.warning(f"Yahoo Finance gagal: {str(e)}")
    
    # 4. Last resort - Metal Price API (limited free tier)
    try:
        apis_tried.append("Free Metal API")
        response = requests.get('https://api.metals.live/v1/spot/gold', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'price': data[0]['price'] if isinstance(data, list) else data.get('price', 0),
                'source': 'Metals.Live',
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
    except:
        pass
    
    return {
        'price': None,
        'source': 'None (APIs failed)',
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'tried': apis_tried
    }

@st.cache_data(ttl=300)
def get_gold_data(period="2y"):
    """Mengambil data historis harga emas"""
    try:
        # Primary: Yahoo Finance for historical
        gold = yf.Ticker("GC=F")
        data = gold.history(period=period)
        
        if len(data) == 0:
            st.error("Tidak ada data historis")
            return None
        
        # Try to update current price with more accurate API
        current_data = get_current_gold_price()
        if current_data['success'] and current_data['price']:
            last_date = data.index[-1]
            today = pd.Timestamp.now(tz=last_date.tz)
            
            # Only update if data is from today
            if (today.date() - last_date.date()).days == 0:
                # Update last row
                data.loc[last_date, 'Close'] = current_data['price']
                data.loc[last_date, 'High'] = max(data.loc[last_date, 'High'], current_data['price'])
                data.loc[last_date, 'Low'] = min(data.loc[last_date, 'Low'], current_data['price'])
            else:
                # Add new row for today if missing
                new_row = pd.DataFrame({
                    'Open': [current_data['price']],
                    'High': [current_data['price']],
                    'Low': [current_data['price']],
                    'Close': [current_data['price']],
                    'Volume': [0],
                    'Dividends': [0],
                    'Stock Splits': [0]
                }, index=[today])
                data = pd.concat([data, new_row])
        
        return data, current_data.get('source', 'Yahoo Finance')
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_exchange_rate():
    """Mengambil kurs USD to IDR dari API gratis"""
    apis = [
        'https://api.exchangerate-api.com/v4/latest/USD',
        'https://open.er-api.com/v6/latest/USD',
        'https://api.exchangerate.host/latest?base=USD'
    ]
    
    for api_url in apis:
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'rates' in data and 'IDR' in data['rates']:
                    return data['rates']['IDR'], api_url.split('/')[2]
        except:
            continue
    
    # Fallback to Yahoo Finance
    try:
        usd_idr = yf.Ticker("USDIDR=X")
        rate = usd_idr.info.get('regularMarketPrice', 15500)
        return rate, 'Yahoo Finance'
    except:
        return 15500, 'Default'

def prepare_features(data, lookback=30):
    """Menyiapkan features untuk ML models"""
    df = data.copy()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'STD_{window}'] = df['Close'].rolling(window=window).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # Lagged features
    for i in range(1, lookback + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    
    # Volume features
    if 'Volume' in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Time features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['Quarter'] = df.index.quarter
    
    return df

def train_ensemble_model(data, forecast_days=30):
    """Melatih ensemble model untuk forecasting"""
    df = prepare_features(data)
    df = df.dropna()
    
    # Prepare train data
    feature_cols = [col for col in df.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']]
    X = df[feature_cols]
    y = df['Close']
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # Train multiple models
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, verbose=-1),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
    }
    
    predictions = {}
    scores = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_scaled)
        pred_scaled = model.predict(X_test_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        predictions[name] = pred
        
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        
        scores[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Accuracy': max(0, 100 - mape)
        }
    
    # Ensemble prediction (weighted average based on R2 score)
    weights = np.array([scores[name]['R2'] for name in models.keys()])
    weights = np.maximum(weights, 0)  # Ensure non-negative
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(models)) / len(models)
    
    ensemble_pred = np.zeros_like(predictions['XGBoost'])
    for i, name in enumerate(models.keys()):
        ensemble_pred += weights[i] * predictions[name]
    
    # Calculate ensemble metrics
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)
    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
    
    scores['Ensemble'] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': max(0, 100 - mape)
    }
    
    # Forecast future
    last_row = X.iloc[-1:].copy()
    future_predictions = []
    
    for _ in range(forecast_days):
        last_row_scaled = scaler_X.transform(last_row)
        
        # Predict with all models
        future_pred = 0
        for i, (name, model) in enumerate(models.items()):
            pred_scaled = model.predict(last_row_scaled)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            future_pred += weights[i] * pred
        
        future_predictions.append(future_pred)
        
        # Update features for next prediction
        for i in range(29, 0, -1):
            if f'Lag_{i}' in last_row.columns and f'Lag_{i+1}' in last_row.columns:
                last_row[f'Lag_{i+1}'] = last_row[f'Lag_{i}'].values[0]
        last_row['Lag_1'] = future_pred
        
        # Update moving averages (simplified)
        for window in [5, 10, 20, 50]:
            if f'MA_{window}' in last_row.columns:
                last_row[f'MA_{window}'] = future_pred * 0.7 + last_row[f'MA_{window}'].values[0] * 0.3
    
    return future_predictions, scores, models, scaler_X, scaler_y

def train_prophet_model(data, forecast_days=30):
    """Melatih Prophet model untuk forecasting"""
    df = data.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df = df[['ds', 'y']]
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Calculate accuracy on historical data
    historical = forecast[forecast['ds'].isin(df['ds'])][['ds', 'yhat']]
    actual = df['y'].values
    predicted = historical['yhat'].values
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy = max(0, 100 - mape)
    
    future_pred = forecast.tail(forecast_days)['yhat'].values
    
    return future_pred, accuracy, forecast

def train_arima_model(data, forecast_days=30):
    """Melatih ARIMA model untuk forecasting"""
    prices = data['Close'].values
    
    try:
        model = ARIMA(prices, order=(5, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_days)
        
        # Calculate in-sample accuracy
        predictions = fitted.fittedvalues
        actual = prices[1:]
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        accuracy = max(0, 100 - mape)
        
        return forecast, accuracy
    except:
        return None, 0

def create_price_chart(data, currency='USD', exchange_rate=1):
    """Membuat grafik harga emas interaktif"""
    df = data.copy()
    df['Price'] = df['Close'] * (exchange_rate if currency == 'IDR' else 1)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Harga Emas ({currency})', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'] * (exchange_rate if currency == 'IDR' else 1),
            high=df['High'] * (exchange_rate if currency == 'IDR' else 1),
            low=df['Low'] * (exchange_rate if currency == 'IDR' else 1),
            close=df['Price'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Volume
    if 'Volume' in df.columns:
        colors = ['red' if row['Close'] < row['Open'] else 'green' for idx, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text=f"Harga ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_forecast_chart(data, forecasts, currency='USD', exchange_rate=1):
    """Membuat grafik forecast"""
    fig = go.Figure()
    
    # Historical data
    historical_price = data['Close'].values * (exchange_rate if currency == 'IDR' else 1)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=historical_price,
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )
    
    # Forecast data
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecasts['Ensemble']))
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, (name, pred) in enumerate(forecasts.items()):
        forecast_price = np.array(pred) * (exchange_rate if currency == 'IDR' else 1)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast_price,
                name=f'{name} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            )
        )
    
    fig.update_layout(
        title=f'Gold Price Forecast ({currency})',
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main App
def main():
    st.title("💰 Gold Price Forecast Pro")
    st.markdown("### Advanced Gold Price Monitoring & Forecasting System")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/gold-bars.png", width=80)
        st.header("⚙️ Settings")
        
        currency = st.selectbox(
            "💱 Currency",
            ["USD", "IDR"],
            help="Pilih mata uang untuk menampilkan harga"
        )
        
        period = st.selectbox(
            "📅 Historical Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=4,
            help="Periode data historis"
        )
        
        forecast_days = st.slider(
            "🔮 Forecast Days",
            min_value=7,
            max_value=90,
            value=30,
            help="Jumlah hari untuk prediksi"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Selection")
        use_ensemble = st.checkbox("Ensemble ML", value=True)
        use_prophet = st.checkbox("Prophet", value=True)
        use_arima = st.checkbox("ARIMA", value=True)
        
        st.markdown("---")
        st.markdown("### 🌐 Free APIs Used")
        st.markdown("""
        <div style='font-size: 11px;'>
        ✅ Gold-API.com<br>
        ✅ ExchangeRate-API<br>
        ✅ Yahoo Finance<br>
        <br>
        <i>No API keys needed!</i>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Get data
    with st.spinner("Loading gold data from free APIs..."):
        result = get_gold_data(period)
        if result is not None:
            data, data_source = result
        else:
            st.error("Gagal mengambil data harga emas")
            return
        
        exchange_rate, exchange_source = get_exchange_rate()
    
    if data is None or len(data) == 0:
        st.error("Gagal mengambil data harga emas")
        return
    
    # Section 1: Current Price & Historical Data
    st.header("📈 Current Gold Price & Historical Data")
    
    # Show data sources
    col_info1, col_info2 = st.columns([3, 1])
    with col_info1:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;'>
        📡 <b>Data Sources:</b> 
        <span class='api-badge'>💰 {data_source}</span>
        <span class='api-badge'>💱 {exchange_source}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_info2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"<div style='text-align: right; color: white;'>🕐 {current_time}</div>", unsafe_allow_html=True)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
    
    if currency == 'IDR':
        display_price = current_price * exchange_rate
        display_change = change * exchange_rate
        curr_symbol = "Rp"
    else:
        display_price = current_price
        display_change = change
        curr_symbol = "$"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"Current Price ({currency})",
            value=f"{curr_symbol}{display_price:,.2f}",
            delta=f"{display_change:,.2f} ({change_pct:.2f}%)"
        )
    
    with col2:
        st.metric(
            label=f"High (24h)",
            value=f"{curr_symbol}{data['High'].iloc[-1] * (exchange_rate if currency == 'IDR' else 1):,.2f}"
        )
    
    with col3:
        st.metric(
            label=f"Low (24h)",
            value=f"{curr_symbol}{data['Low'].iloc[-1] * (exchange_rate if currency == 'IDR' else 1):,.2f}"
        )
    
    with col4:
        st.metric(
            label="USD/IDR Rate",
            value=f"Rp{exchange_rate:,.2f}"
        )
    
    # Price chart
    st.plotly_chart(
        create_price_chart(data, currency, exchange_rate),
        use_container_width=True
    )
    
    # Statistics
    with st.expander("📊 Statistical Summary"):
        col1, col2 = st.columns(2)
        
        prices = data['Close'].values * (exchange_rate if currency == 'IDR' else 1)
        
        with col1:
            st.markdown("#### Price Statistics")
            st.write(f"Mean: {curr_symbol}{np.mean(prices):,.2f}")
            st.write(f"Median: {curr_symbol}{np.median(prices):,.2f}")
            st.write(f"Std Dev: {curr_symbol}{np.std(prices):,.2f}")
            st.write(f"Min: {curr_symbol}{np.min(prices):,.2f}")
            st.write(f"Max: {curr_symbol}{np.max(prices):,.2f}")
        
        with col2:
            st.markdown("#### Returns")
            returns = data['Close'].pct_change().dropna()
            st.write(f"Daily Return Mean: {np.mean(returns)*100:.4f}%")
            st.write(f"Daily Return Std: {np.std(returns)*100:.4f}%")
            st.write(f"Volatility (Annualized): {np.std(returns)*np.sqrt(252)*100:.2f}%")
            sharpe = (np.mean(returns)/np.std(returns))*np.sqrt(252) if np.std(returns) > 0 else 0
            st.write(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Section 2: Forecast
    st.header("🔮 Price Forecast")
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Training advanced ML models... This may take a minute..."):
            forecasts = {}
            scores = {}
            
            # Ensemble Model
            if use_ensemble:
                try:
                    ensemble_pred, ensemble_scores, models, scaler_X, scaler_y = train_ensemble_model(data, forecast_days)
                    forecasts['Ensemble'] = ensemble_pred
                    scores.update(ensemble_scores)
                    st.success("✅ Ensemble model trained successfully!")
                except Exception as e:
                    st.error(f"Ensemble model error: {str(e)}")
            
            # Prophet Model
            if use_prophet:
                try:
                    prophet_pred, prophet_acc, prophet_forecast = train_prophet_model(data, forecast_days)
                    forecasts['Prophet'] = prophet_pred
                    scores['Prophet'] = {'Accuracy': prophet_acc, 'MAPE': 100 - prophet_acc}
                    st.success("✅ Prophet model trained successfully!")
                except Exception as e:
                    st.error(f"Prophet model error: {str(e)}")
            
            # ARIMA Model
            if use_arima:
                try:
                    arima_pred, arima_acc = train_arima_model(data, forecast_days)
                    if arima_pred is not None:
                        forecasts['ARIMA'] = arima_pred
                        scores['ARIMA'] = {'Accuracy': arima_acc, 'MAPE': 100 - arima_acc}
                        st.success("✅ ARIMA model trained successfully!")
                except Exception as e:
                    st.error(f"ARIMA model error: {str(e)}")
            
            if forecasts:
                st.session_state['forecasts'] = forecasts
                st.session_state['scores'] = scores
    
    # Display forecasts
    if 'forecasts' in st.session_state and st.session_state['forecasts']:
        forecasts = st.session_state['forecasts']
        scores = st.session_state['scores']
        
        # Model Performance
        st.subheader("🎯 Model Performance")
        
        perf_data = []
        for model_name, score in scores.items():
            perf_data.append({
                'Model': model_name,
                'Accuracy (%)': f"{score.get('Accuracy', 0):.2f}",
                'MAPE (%)': f"{score.get('MAPE', 0):.4f}",
                'R² Score': f"{score.get('R2', 0):.4f}" if 'R2' in score else 'N/A',
                'RMSE': f"{score.get('RMSE', 0):.2f}" if 'RMSE' in score else 'N/A'
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # Best model
        best_model = max(scores.items(), key=lambda x: x[1].get('Accuracy', 0))
        st.info(f"🏆 Best Model: **{best_model[0]}** with {best_model[1]['Accuracy']:.2f}% accuracy")
        
        # Forecast chart
        st.plotly_chart(
            create_forecast_chart(data, forecasts, currency, exchange_rate),
            use_container_width=True
        )
        
        # Forecast table
        st.subheader("📋 Detailed Forecast")
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d')
        })
        
        for name, pred in forecasts.items():
            forecast_price = np.array(pred) * (exchange_rate if currency == 'IDR' else 1)
            forecast_df[f'{name} ({currency})'] = [f"{curr_symbol}{p:,.2f}" for p in forecast_price]
        
        st.dataframe(forecast_df, use_container_width=True, height=400)
        
        # Download forecast
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"gold_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Price prediction summary
        st.subheader("💡 Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        
        ensemble_pred = forecasts.get('Ensemble', forecasts[list(forecasts.keys())[0]])
        final_pred = ensemble_pred[-1] * (exchange_rate if currency == 'IDR' else 1)
        avg_pred = np.mean(ensemble_pred) * (exchange_rate if currency == 'IDR' else 1)
        max_pred = np.max(ensemble_pred) * (exchange_rate if currency == 'IDR' else 1)
        min_pred = np.min(ensemble_pred) * (exchange_rate if currency == 'IDR' else 1)
        
        with col1:
            st.metric(
                f"Predicted Price (Day {forecast_days})",
                f"{curr_symbol}{final_pred:,.2f}",
                f"{((final_pred - display_price) / display_price * 100):.2f}%"
            )
        
        with col2:
            st.metric(
                "Avg Forecast Price",
                f"{curr_symbol}{avg_pred:,.2f}"
            )
        
        with col3:
            st.metric(
                "Price Range",
                f"{curr_symbol}{min_pred:,.2f} - {curr_symbol}{max_pred:,.2f}"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>💰 <b>Gold Price Forecast Pro</b> - Powered by Advanced Machine Learning & Free APIs</p>
        <p><small>🌐 APIs: Gold-API.com, ExchangeRate-API, Yahoo Finance (No Keys Required!)</small></p>
        <p><small>🤖 Models: XGBoost, LightGBM, Prophet, ARIMA, Random Forest, Gradient Boosting & Ensemble</small></p>
        <p><small>⚠️ Disclaimer: Forecasts are for informational purposes only. Not financial advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
