import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import joblib
import hopsworks  
# ✅ Import the function from your Python file



# ============================================================
# 📱 PAGE CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title="SkyCast - AQI Prediction",
    page_icon="🌤️",
    layout="centered",  # Changed from "wide" to "centered" for proper responsive fit
    initial_sidebar_state="collapsed"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

st.markdown("""
<style>
    /* ============ ROOT VARIABLES & FULL-SCREEN ============ */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --accent-primary: #5b6fdb;
        --accent-secondary: #6b5d9e;
    }
    
    /* Dark mode variables */
    html[data-theme="dark"] {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --text-primary: #f3f4f6;
        --text-secondary: #d1d5db;
        --border-color: #374151;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
    }
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Streamlit container adjustments */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 20px !important;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Hide sidebar & decorations */
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    
    /* ============ HEADER STYLING ============ */
    .header {
        background: linear-gradient(135deg, #5b6fdb 0%, #6b5d9e 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(91, 111, 219, 0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
    }
    
    .header p {
        margin: 4px 0 0 0;
        font-size: 13px;
        opacity: 0.9;
    }
    
    /* Theme toggle button */
    .theme-toggle-btn {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 8px 14px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.2s ease;
        white-space: nowrap;
    }
    
    .theme-toggle-btn:hover {
        background: rgba(255, 255, 255, 0.25);
        border-color: rgba(255, 255, 255, 0.5);
        transform: scale(1.08);
    }
    
    /* ============ LOCATION & TIME ============ */
    .location-time {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding: 0 4px;
    }
    
    .location-time h2 {
        margin: 0;
        color: var(--text-primary);
        font-size: 16px;
        font-weight: 600;
    }
    
    /* ============ AQI DISPLAY ============ */
    .aqi-display {
        background: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .aqi-display:hover {
        transform: scale(1.04) translateY(-5px);
        box-shadow: 0 12px 28px rgba(91, 111, 219, 0.25);
        border-color: #5b6fdb;
    }
    
    .aqi-number {
        font-size: 56px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #5b6fdb, #6b5d9e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .aqi-status {
        font-size: 16px;
        font-weight: 700;
        margin: 10px 0 0 0;
        color: var(--text-primary);
    }
    
    /* ============ INFO CARDS ============ */
    .info-card {
        background: linear-gradient(135deg, #5b6fdb 0%, #6b5d9e 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 4px 12px rgba(91, 111, 219, 0.2);
        cursor: pointer;
    }
    
    .info-card:hover {
        transform: scale(1.08) translateY(-6px);
        box-shadow: 0 14px 32px rgba(91, 111, 219, 0.35);
    }
    
    .info-card-label {
        font-size: 12px;
        font-weight: 700;
        opacity: 0.92;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .info-card-value {
        font-size: 32px;
        font-weight: 800;
        margin: 6px 0;
    }
    
    .info-card-subtitle {
        font-size: 11px;
        opacity: 0.85;
        margin-top: 4px;
    }
    
    /* ============ HEALTH RECOMMENDATIONS ============ */
    .health-rec-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 12px;
        padding: 18px;
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .health-rec-box:hover {
        transform: scale(1.05) translateY(-5px);
        box-shadow: 0 14px 28px rgba(16, 185, 129, 0.3);
    }
    
    .health-rec-title {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.95;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .health-rec-item {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        padding: 8px 12px;
        margin: 7px 0;
        border-left: 3px solid rgba(255, 255, 255, 0.7);
        font-size: 12px;
        line-height: 1.4;
        transition: all 0.2s ease;
    }
    
    .health-rec-item:hover {
        background: rgba(255, 255, 255, 0.25);
        border-left-color: white;
    }
    
    /* ============ FORECAST CARDS ============ */
    .forecast-card {
        background: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
    }
    
    .forecast-card:hover {
        transform: scale(1.07) translateY(-6px);
        border-color: #5b6fdb;
        box-shadow: 0 12px 24px rgba(91, 111, 219, 0.25);
    }
    
    .forecast-card-today {
        background: linear-gradient(135deg, #5b6fdb15 0%, #6b5d9e15 100%);
        border: 3px solid #5b6fdb;
        box-shadow: 0 6px 16px rgba(91, 111, 219, 0.2);
    }
    
    .forecast-card-today:hover {
        transform: scale(1.09) translateY(-7px);
        box-shadow: 0 16px 32px rgba(91, 111, 219, 0.3);
    }
    
    .forecast-day {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    
    .forecast-date {
        font-size: 12px;
        color: var(--text-secondary);
        margin-bottom: 12px;
    }
    
    .forecast-aqi {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    
    .forecast-aqi-item {
        flex: 1;
    }
    
    .forecast-aqi-label {
        font-size: 10px;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 3px;
    }
    
    .forecast-aqi-value {
        font-size: 20px;
        font-weight: 800;
        background: linear-gradient(135deg, #5b6fdb, #6b5d9e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .forecast-divider {
        width: 1px;
        height: 32px;
        background: var(--border-color);
        margin: 0 10px;
    }
    
    /* ============ CHART & SECTION TITLES ============ */
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 16px 0 12px 0;
    }
    
    .chart-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 14px;
        margin-top: 8px;
    }
    
    /* ============ DIVIDERS ============ */
    hr {
        border: none;
        height: 1px;
        background: var(--border-color);
        margin: 14px 0;
    }
    
    /* ============ FOOTER ============ */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 12px;
        padding: 16px;
        margin-top: 12px;
    }
    
    /* ============ RESPONSIVE ============ */
    @media (max-width: 768px) {
        .header { padding: 14px 16px; font-size: 20px; }
        .header h1 { font-size: 22px; }
        .aqi-number { font-size: 48px; }
        .info-card-value { font-size: 28px; }
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.dark_mode:
    st.markdown('<script>document.documentElement.setAttribute("data-theme", "dark");</script>', unsafe_allow_html=True)




import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# ===============================
# 🔹 Load latest predictions CSV
# ===============================
import pandas as pd
from datetime import datetime

CITY = "Karachi, Pakistan"
LAT, LON = 24.8607, 67.0011
current_time = datetime.now().strftime("%I:%M %p")

# csv_file = "latest_predictions.csv"
# Hopsworks login
project = hopsworks.login(api_key_secret_name="HOPSWORKS_API_KEY")
dataset_api = project.get_dataset_api()

# Download latest CSV to a DataFrame
try:
    latest_csv_path = dataset_api.download("latest_predictions.csv", "aqi_predictions", overwrite=True)
    forecast_df = pd.read_csv(latest_csv_path)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
except Exception as e:
    print(f"❌ Error downloading CSV: {e}")
    forecast_df = pd.DataFrame(columns=[
        "date","day_name","aqi_min","aqi_max","temperature","humidity","wind_speed"
    ])

# df = pd.read_csv(latest_csv_path)

# # Read CSV safely
# try:
#     forecast_df = pd.read_csv(csv_file)
#     forecast_df["date"] = pd.to_datetime(forecast_df["date"])
# except FileNotFoundError:
#     print(f"❌ {csv_file} not found! Run KHI_AQI.py first.")
#     forecast_df = pd.DataFrame(columns=[
#         "date","day_name","aqi_min","aqi_max","temperature","humidity","windspeed"
#     ])

# ===============================
# 🔹 Current values for front-end
# ===============================
if not forecast_df.empty:
    # Assuming first row is 'Today'
    today_forecast = forecast_df.iloc[0]

    current_aqi = int((today_forecast["aqi_min"] + today_forecast["aqi_max"]) / 2)
    current_temp = int(today_forecast["temperature"])
    current_humidity = int(today_forecast["humidity"])
    current_wind = int(today_forecast["wind_speed"])
else:
    # Default values if CSV empty
    current_aqi = 0
    current_temp = 0
    current_humidity = 0
    current_wind = 0

# ===============================
# 🔹 Optional: full 3-days forecast
# ===============================
forecast_list = forecast_df.to_dict(orient="records")

# Now your app.py front-end can use:
# current_aqi, current_temp, current_humidity, current_wind
# forecast_list (3-days forecast)


# # ============================================================
# # 🌍 DATA
# # ============================================================
# df_weather = pd.read_csv("karachi_weather_5hourly.csv")
# df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])

# # Load the trained Random Forest model
# model_file = "best_random_forest.pkl"
# rf_model = joblib.load(model_file)

# CITY = "Karachi, Pakistan"
# LAT, LON = 24.8607, 67.0011

# # ============================================================
# # 🔹 Derived features
# # ============================================================
# df_weather["hour"] = df_weather["timestamp"].dt.hour
# df_weather["day"] = df_weather["timestamp"].dt.day
# df_weather["month"] = df_weather["timestamp"].dt.month
# df_weather["weekday"] = df_weather["timestamp"].dt.weekday
# df_weather["is_weekend"] = (df_weather["weekday"] >= 5).astype(int)
# df_weather["aqi_change"] = df_weather["aqi"].diff().fillna(0)
# df_weather["aqi_change_rate"] = df_weather["aqi"].pct_change(fill_method=None)\
#     .replace([np.inf, -np.inf], 0).clip(-10,10).fillna(0)

# # ============================================================
# # 🔹 Feature columns as used in model
# # ============================================================
# feature_cols = [
#     "pm2_5",
#     "pm10",
#     "temperature",
#     "humidity",
#     "wind_speed",
#     "aqi_change",
#     "aqi_change_rate",
#     "hour",
#     "day",
#     "month",
#     "weekday",
#     "is_weekend"
# ]

# # ============================================================
# # 🔹 Current weather
# # ============================================================
# last_row = df_weather.iloc[-1:].copy()
# recent_3 = df_weather.tail(3)

# # Current averages
# current_temp = int(recent_3["temperature"].mean())
# current_humidity = int(recent_3["humidity"].mean())
# current_wind = int(recent_3["wind_speed"].mean())

# # Fill last row for prediction
# last_row["temperature"] = current_temp
# last_row["humidity"] = current_humidity
# last_row["wind_speed"] = current_wind

# # Predict current AQI
# X_current = last_row[feature_cols].values.reshape(1, -1)
# current_aqi = int(rf_model.predict(X_current)[0])

# # AQI status function
# def aqi_status(aqi):
#     if aqi <= 50:
#         return "Good"
#     elif aqi <= 100:
#         return "Moderate"
#     elif aqi <= 150:
#         return "Unhealthy for Sensitive"
#     elif aqi <= 200:
#         return "Unhealthy"
#     elif aqi <= 300:
#         return "Very Unhealthy"
#     else:
#         return "Hazardous"

# current_aqi_status = aqi_status(current_aqi)
# current_time = datetime.now().strftime("%I:%M %p")

# # ============================================================
# # 🔹 Forecast function
# # ============================================================
# def get_forecast(df_capped, model_path=model_file):
#     rf_model = joblib.load(model_path)
    
#     # Ensure datetime & sort
#     df_capped["timestamp"] = pd.to_datetime(df_capped["timestamp"])
#     df_capped = df_capped.sort_values("timestamp").reset_index(drop=True)

#     # Derived features
#     df_capped["hour"] = df_capped["timestamp"].dt.hour
#     df_capped["day"] = df_capped["timestamp"].dt.day
#     df_capped["month"] = df_capped["timestamp"].dt.month
#     df_capped["weekday"] = df_capped["timestamp"].dt.weekday
#     df_capped["is_weekend"] = (df_capped["weekday"] >= 5).astype(int)
#     df_capped["aqi_change"] = df_capped["aqi"].diff().fillna(0)
#     df_capped["aqi_change_rate"] = df_capped["aqi"].pct_change(fill_method=None)\
#         .replace([np.inf, -np.inf], 0).clip(-10,10).fillna(0)

#     # Last row
#     last_row = df_capped.iloc[-1:].copy()
#     recent_3days = df_capped.tail(3)
#     last_row["temperature"] = int(recent_3days["temperature"].mean())
#     last_row["humidity"] = int(recent_3days["humidity"].mean())
#     last_row["wind_speed"] = int(recent_3days["wind_speed"].mean())

#     forecast_dates = [datetime.now() + timedelta(days=i) for i in range(3)]
#     day_names = ["Today", "Tomorrow", "Day After"]

#     forecast_data = []
#     for i, date in enumerate(forecast_dates):
#         future_row = last_row.copy()
#         future_row["timestamp"] = date
#         future_row["hour"] = date.hour
#         future_row["day"] = date.day
#         future_row["month"] = date.month
#         future_row["weekday"] = date.weekday()
#         future_row["is_weekend"] = int(future_row["weekday"].iloc[0] >= 5)

#         X_future = future_row[feature_cols].values.reshape(1, -1)
#         predicted_aqi = int(rf_model.predict(X_future)[0])

#         forecast_data.append({
#             "date": date,
#             "day_name": day_names[i],
#             "aqi_min": max(predicted_aqi - 5, 0),
#             "aqi_max": predicted_aqi + 5,
#             "temperature": int(recent_3days["temperature"].mean()),
#             "humidity": int(recent_3days["humidity"].mean()),
#             "wind_speed": int(recent_3days["wind_speed"].mean())
#         })

#     return forecast_data


# forecast_data = get_forecast(df_weather, model_path=model_file)  # returns list of dicts with keys: date, day_name, aqi_min, aqi_max, temperature, humidity, windspeed


# current_time = datetime.now().strftime("%I:%M %p")
# current_aqi = 85
# current_aqi_status = "Moderate"
# current_temp = 28
# current_humidity = 62
# current_wind = 12

# forecast_dates = [
#     datetime.now(),
#     datetime.now() + timedelta(days=1),
#     datetime.now() + timedelta(days=2)
# ]

# forecast_data = [
#     {"date": forecast_dates[0], "aqi_min": 78, "aqi_max": 92, "day_name": "Today"},
#     {"date": forecast_dates[1], "aqi_min": 72, "aqi_max": 88, "day_name": "Tomorrow"},
#     {"date": forecast_dates[2], "aqi_min": 75, "aqi_max": 95, "day_name": "Day After"}
# ]

# ============================================================
# 💚 HEALTH RECOMMENDATIONS
# ============================================================
def get_health_recommendations(aqi):
    if aqi <= 50:
        return {
            "status": "Good",
            "recommendations": [
                "✓ Air quality is satisfactory",
                "✓ Enjoy outdoor activities freely",
                "✓ No restrictions for any group"
            ]
        }
    elif aqi <= 100:
        return {
            "status": "Moderate",
            "recommendations": [
                "⚠ Sensitive people should limit outdoor time",
                "⚠ Consider wearing N95 mask",
                "⚠ Take breaks during outdoor exercise"
            ]
        }
    elif aqi <= 150:
        return {
            "status": "Unhealthy for Sensitive",
            "recommendations": [
                "⚠ Sensitive groups should avoid outdoors",
                "⚠ Wear quality N95 mask if going out",
                "⚠ Reduce outdoor exercise intensity"
            ]
        }
    else:
        return {
            "status": "Hazardous",
            "recommendations": [
                "🚫 Avoid outdoor activities",
                "🚫 Stay indoors with air purifier",
                "🚫 Wear N95 if absolutely necessary"
            ]
        }

# ============================================================
# 🎨 RENDER PAGE
# ============================================================

st.markdown(f"""
<div class="header">
    <div class="header-left">
        <h1>🌤️ SkyCast</h1>
        <p>Air Quality Index Prediction for Karachi</p>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([10, 1])
with col2:
    if st.button("☀️/🌙", key="theme_toggle", help="Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Location & Time
col_loc, col_time = st.columns([3, 1])
with col_loc:
    st.markdown(f"<div class='location-time'><h2>📍 {CITY}</h2></div>", unsafe_allow_html=True)
with col_time:
    st.markdown(f"<div class='location-time' style='text-align: right;'><h2>⏰ {current_time}</h2></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# Current AQI + Health Recommendations
# ============================================================
col_aqi, col_health = st.columns([1.2, 1], gap="medium")

health_info = get_health_recommendations(current_aqi)

with col_aqi:
    st.markdown(f"""
    <div class="aqi-display">
        <p class="aqi-number">{current_aqi}</p>
        <p class="aqi-status">{health_info["status"]}</p>
    </div>
    """, unsafe_allow_html=True)

with col_health:
    rec_items = "".join([
        f"<div class='health-rec-item'>{rec}</div>"
        for rec in health_info["recommendations"]
    ])
    st.markdown(f"""
    <div class="health-rec-box">
        <div class="health-rec-title">Health Recommendations</div>
        {rec_items}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# Current Weather Cards
# ============================================================
st.markdown("<h3 class='section-title'>Current Weather</h3>", unsafe_allow_html=True)

col_temp, col_humid, col_wind = st.columns(3, gap="medium")

with col_temp:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-label">Temperature</div>
        <div class="info-card-value">{current_temp}°C</div>
        <div class="info-card-subtitle">Partly Cloudy</div>
    </div>
    """, unsafe_allow_html=True)

with col_humid:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-label">Humidity</div>
        <div class="info-card-value">{current_humidity}%</div>
        <div class="info-card-subtitle">Moderate</div>
    </div>
    """, unsafe_allow_html=True)

with col_wind:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-label">Wind Speed</div>
        <div class="info-card-value">{current_wind}<span style='font-size: 18px;'> km/h</span></div>
        <div class="info-card-subtitle">Light Breeze</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 3-Day Forecast
# ============================================================
st.markdown("<h3 class='section-title'>3-Day AQI Forecast</h3>", unsafe_allow_html=True)

forecast_cols = st.columns(3, gap="medium")

for idx, (col, forecast) in enumerate(zip(forecast_cols, forecast_list)):
    with col:
        card_class = "forecast-card forecast-card-today" if idx == 0 else "forecast-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div class="forecast-day">{forecast["day_name"]}</div>
            <div class="forecast-date">{forecast["date"].strftime("%b %d, %Y")}</div>
            <div class="forecast-aqi">
                <div class="forecast-aqi-item">
                    <div class="forecast-aqi-label">Min</div>
                    <div class="forecast-aqi-value">{forecast["aqi_min"]}</div>
                </div>
                <div class="forecast-divider"></div>
                <div class="forecast-aqi-item">
                    <div class="forecast-aqi-label">Max</div>
                    <div class="forecast-aqi-value">{forecast["aqi_max"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# AQI Trend Chart
# ============================================================
st.markdown("<h3 class='section-title'>AQI Trend Over Time</h3>", unsafe_allow_html=True)

trend_hours = [datetime.now() - timedelta(hours=6+i) for i in range(7)]
trend_aqi = [72, 75, 78, 82, 85, 88, 85]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=trend_hours,
    y=trend_aqi,
    mode='lines+markers',
    fill='tozeroy',
    line=dict(color='#5b6fdb', width=3),
    marker=dict(size=8, color='#6b5d9e'),
    fillcolor='rgba(91, 111, 219, 0.1)',
    name='AQI Level',
    hovertemplate='<b>%{x|%I:%M %p}</b><br>AQI: %{y}<extra></extra>'
))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="AQI Level",
    hovermode='x unified',
    template='plotly_white',
    height=300,
    margin=dict(l=40, r=20, t=20, b=40),
    font=dict(family='Segoe UI, sans-serif', size=11, color='#374151'),
    paper_bgcolor='white',
    plot_bgcolor='#f9fafb',
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e5e7eb'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown(f"""
<div class="footer">
    <p>🌍 SkyCast - Real-time Air Quality Predictions</p>
    <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)




