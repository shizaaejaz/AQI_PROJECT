#!/usr/bin/env python
# coding: utf-8

# In[381]:


import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from requests. exceptions import ConnectionError, Timeout
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import subprocess


# In[382]:


# ========== CONFIG ==========
API_KEY = "27c47edf7c5e593798ca2af3c580aeae"
LAT, LON = 24.8607, 67.0011
CITY_ID = 1174872
CSV_PATH = "karachi_weather_5hourly.csv"
SLEEP_SECONDS = 2       # delay between calls
RETRY_DELAY = 5         # wait time before retry
MAX_RETRIES = 3


# In[383]:


# RETRY HANDLER
# ==============================
def fetch_with_retry(url, params, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Handle transient errors gracefully with retries"""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200 and r.json().get("list"):
                return r.json()["list"][0]
            else:
                print(f"⚠ API error {r.status_code} (attempt {attempt})")
        except (ConnectionError, Timeout) as e:
            print(f"⚠ Connection issue: {e} — retrying in {delay}s (attempt {attempt})")
            time.sleep(delay)
    print("❌ Failed after max retries.")
    return None


# In[384]:


# ==============================
# FETCH FUNCTIONS
# ==============================
def fetch_weather_at(timestamp):
    url = "http://history.openweathermap.org/data/2.5/history/city"
    params = {
        "id": CITY_ID,
        "type": "hour",
        "start": int(timestamp.timestamp()),
        "end": int((timestamp + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    return fetch_with_retry(url, params)


# In[385]:


def fetch_pollution_at(timestamp):
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT,
        "lon": LON,
        "start": int(timestamp.timestamp()),
        "end": int((timestamp + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    return fetch_with_retry(url, params)


# In[386]:


def build_record(timestamp):
    weather = fetch_weather_at(timestamp)
    pollution = fetch_pollution_at(timestamp)
    time.sleep(SLEEP_SECONDS)

    if not weather or not pollution:
        print(f"⏭ Skipped {timestamp}, incomplete data.")
        return None

    comp = pollution["components"]
    record = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": pollution["main"]["aqi"],
        "pm2_5": comp["pm2_5"],
        "pm10": comp["pm10"],
        "temperature": round(weather["main"]["temp"] - 273.15, 2),
        "humidity": weather["main"]["humidity"],
        "wind_speed": weather["wind"]["speed"]
    }
    return record


# In[387]:


# ==============================
# SAVE AND RESUME LOGIC
# ==============================
def get_last_timestamp():
    """Find last saved timestamp to resume safely."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if "timestamp" in df.columns and not df.empty:
            return pd.to_datetime(df["timestamp"]).max()
    return datetime.now() - timedelta(days=365)

def save_records(records):
    """Append new records and drop duplicates safely."""
    if not records:
        return
    df_new = pd.DataFrame(records)
    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
    else:
        df = df_new
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Saved {len(df)} total records to {CSV_PATH}")


# In[388]:


# # ==============================
# # MAIN COLLECTOR
# # ==============================
# def collect_5hourly_data():
#     now = datetime.now() - timedelta(hours=2)
#     start = get_last_timestamp() + timedelta(hours=5)
#     records = []
#     count = 0

#     while start < now:
#         print(f"📦 Fetching data for {start}")
#         record = build_record(start)
#         if record:
#             records.append(record)
#             count += 1
#         if len(records) > 0::  
#             save_records(records)
#             df_new = pd.DataFrame(records)
#             if not df_new.empty:
#                 karachi_fg.insert(df_new)
#                 print(f"📤 Inserted {len(df_new)} new records into Feature Store.")

#             records = []
#         start += timedelta(hours=5)

#     save_records(records)
#     print("🎉 Data collection completed successfully!")
#     print("DEBUG: start =", start, "| now =", now)


# ==============================
# MAIN COLLECTOR
# ==============================
def collect_5hourly_data():
    now = datetime.now() - timedelta(hours=2)
    start = get_last_timestamp() + timedelta(hours=5)
    records = []

    if (datetime.now() - get_last_timestamp()).total_seconds() < 5*3600:
        print("✅ Data is up-to-date, no fetch needed.")
        return

    while start < now:
        print(f"📦 Fetching data for {start}")
        record = build_record(start)

        if record:
            records.append(record)

            # ✅ Save and insert immediately after each record batch
            save_records(records)
            # df_new = pd.DataFrame(records)

            # if not df_new.empty:
            #     karachi_fg.insert(df_new)
            #     print(f"📤 Inserted {len(df_new)} new records into Feature Store.")

            records = []  # clear after inserting

        start += timedelta(hours=5)

    print("🎉 Data collection completed successfully!")
    print("DEBUG: start =", start, "| now =", now)



# In[389]:


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    collect_5hourly_data()


# In[390]:


df = pd.read_csv("karachi_weather_5hourly.csv")


# In[546]:


# ==============================
# ✅ Convert to US EPA AQI scale using PM2.5 and PM10
# ==============================
import numpy as np

def pm25_to_us_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= pm25 <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (pm25 - Clow) + Ilow
    return 500

def pm10_to_us_aqi(pm10):
    breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ]
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= pm10 <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (pm10 - Clow) + Ilow
    return 500

# Apply to your dataset
df["us_aqi_pm25"] = df["pm2_5"].apply(pm25_to_us_aqi)
df["us_aqi_pm10"] = df["pm10"].apply(pm10_to_us_aqi)

# ✅ Final AQI = max of PM2.5-based and PM10-based AQI (as per EPA definition)
df["aqi"] = df[["us_aqi_pm25", "us_aqi_pm10"]].max(axis=1)
df.drop(columns=["us_aqi_pm25", "us_aqi_pm10"], inplace=True)

# Optional: round for neatness
df["aqi"] = df["aqi"].round(1)

print("✅ Converted AQI values to EPA scale successfully!")
print(df[["pm2_5", "pm10", "aqi"]].head())


# In[547]:


df.info()


# In[548]:


df.describe()


# In[549]:


df.isnull().sum()


# In[550]:


df.duplicated().sum()


# In[551]:


print(df["aqi"].describe())


# In[552]:


df["aqi"].value_counts()


# In[553]:


df["pm10"].value_counts()


# In[554]:


df["pm2_5"].value_counts()


# In[555]:


df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# In[556]:


# --- Time-based features ---
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)


# In[557]:


# ==============================
# AQI already computed using EPA scaling
# ==============================

# --- Derived features based on AQI ---
df["aqi_change"] = df["aqi"].diff().fillna(0)

df["aqi_change_rate"] = df["aqi"].pct_change().replace([np.inf, -np.inf], 0)
df["aqi_change_rate"] = df["aqi_change_rate"].clip(-10, 10)

# ✅ Check
print(df[["pm2_5", "pm10", "aqi", "aqi_change", "aqi_change_rate"]].head(10))


# In[558]:


df


# In[559]:


# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()


# In[560]:


# -------- Plot full heatmap --------
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("📊 Correlation Heatmap of AQI and Weather Features", fontsize=14, pad=12)
plt.show()


# In[561]:


# -------- Show correlation of each feature with AQI --------
aqi_corr = corr_matrix["aqi"].sort_values(ascending=False)
print("🔍 Correlation of each feature with AQI:\n")
print(aqi_corr)


# In[562]:


# Optional: visualize the top 10 correlated features with AQI
plt.figure(figsize=(8, 4))
sns.barplot(x=aqi_corr.values[:10], y=aqi_corr.index[:10], palette="coolwarm")
plt.title("Top 10 Features Most Correlated with AQI", fontsize=13, pad=10)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.show()


# In[563]:


num_cols = ['temperature', 'humidity', 'wind_speed', 'pm2_5', 'pm10', 'aqi']

outlier_indices = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_indices[col] = len(outliers)

    print(f"{col}: {len(outliers)} outliers")


# In[564]:


plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 5, i)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(col)
plt.tight_layout()
plt.show()


# In[595]:


import numpy as np
import pandas as pd

def cap_outliers_epa_safe(df, numeric_cols, pm_cols=["pm2_5", "pm10"]):
    capped_df = df.copy()
    summary = []

    # --- IQR-based capping for other numeric columns ---
    for col in numeric_cols:
        if col in pm_cols:
            continue  # skip PM columns for IQR capping
        before_min = df[col].min()
        before_max = df[col].max()

        Q1 = capped_df[col].quantile(0.25)
        Q3 = capped_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        capped_df[col] = np.where(
            capped_df[col] < lower, lower,
            np.where(capped_df[col] > upper, upper, capped_df[col])
        )

        after_min = capped_df[col].min()
        after_max = capped_df[col].max()

        summary.append({
            "Column": col,
            "Before Min": round(before_min, 2),
            "Before Max": round(before_max, 2),
            "After Min": round(after_min, 2),
            "After Max": round(after_max, 2),
            "Lower Cap": round(lower, 2),
            "Upper Cap": round(upper, 2)
        })

    # --- EPA-safe clipping for PM columns ---
    capped_df["pm2_5"] = capped_df["pm2_5"].clip(0, 500.4)
    capped_df["pm10"]  = capped_df["pm10"].clip(0, 604)

    summary.append({
        "Column": "pm2_5",
        "Before Min": df["pm2_5"].min(),
        "Before Max": df["pm2_5"].max(),
        "After Min": capped_df["pm2_5"].min(),
        "After Max": capped_df["pm2_5"].max(),
        "Lower Cap": 0,
        "Upper Cap": 500.4
    })

    summary.append({
        "Column": "pm10",
        "Before Min": df["pm10"].min(),
        "Before Max": df["pm10"].max(),
        "After Min": capped_df["pm10"].min(),
        "After Max": capped_df["pm10"].max(),
        "Lower Cap": 0,
        "Upper Cap": 604
    })

    print("✅ Outlier Capping Summary (IQR + EPA-safe):")
    print(pd.DataFrame(summary))

    return capped_df

# --- Example usage ---
numeric_cols = ['temperature', 'wind_speed', 'pm2_5', 'pm10']
df_capped = cap_outliers_epa_safe(df, numeric_cols)

# --- Recalculate AQI after capping ---
df_capped["us_aqi_pm25"] = df_capped["pm2_5"].apply(pm25_to_us_aqi)
df_capped["us_aqi_pm10"] = df_capped["pm10"].apply(pm10_to_us_aqi)
df_capped["aqi"] = df_capped[["us_aqi_pm25", "us_aqi_pm10"]].max(axis=1).round(1)
df_capped.drop(columns=["us_aqi_pm25", "us_aqi_pm10"], inplace=True)

print("✅ AQI recalculated after capping.")
print(df_capped["aqi"].describe())
df_capped["aqi"] = df_capped["aqi"].astype("float64")


# In[575]:


# # ✅ Recalculate AQI using capped PM2.5 and PM10 (US EPA scale)
# df_capped["us_aqi_pm25"] = df_capped["pm2_5"].apply(pm25_to_us_aqi)
# df_capped["us_aqi_pm10"] = df_capped["pm10"].apply(pm10_to_us_aqi)
# df_capped["aqi"] = df_capped[["us_aqi_pm25", "us_aqi_pm10"]].max(axis=1)
# df_capped.drop(columns=["us_aqi_pm25", "us_aqi_pm10"], inplace=True)
# df_capped["aqi"] = df_capped["aqi"].round(1)

# print("✅ AQI recalculated after capping.")
# print(df_capped["aqi"].describe())


# In[576]:


df_capped["aqi"].value_counts()


# In[577]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot Before Outlier Capping")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()


# In[578]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df_capped[numeric_cols])
plt.title("Boxplot After Outlier Capping")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()


# In[579]:


#The extreme values that got capped were very few or very close to the normal range.


# In[580]:


# get_ipython().system('pip install hopsworks --upgrade')
subprocess.run(["pip", "install", "--upgrade", "hopsworks"])

# In[609]:


import hopsworks

import os
import hopsworks

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
       
fs = project.get_feature_store()    # get your feature store handle


# In[610]:


import pandas as pd
df = pd.read_csv("karachi_weather_5hourly.csv", parse_dates=[0], index_col=False)   # or your latest processed file


# In[636]:


# --- Derived time features ---
df_capped["timestamp"] = pd.to_datetime(df_capped["timestamp"],errors="coerce")
df_capped["hour"] = df_capped["timestamp"].dt.hour
df_capped["day"] = df_capped["timestamp"].dt.day
df_capped["month"] = df_capped["timestamp"].dt.month
df_capped["weekday"] = df_capped["timestamp"].dt.weekday
df_capped["is_weekend"] = (df_capped["weekday"] >= 5).astype(int)

# --- AQI change features ---
df_capped["aqi_change"] = df_capped["aqi"].diff().fillna(0).astype("float64")
df_capped["aqi_change_rate"] = df_capped["aqi"].pct_change().replace([np.inf, -np.inf], 0).clip(-10, 10).astype("float64")
df_capped["aqi_change_rate"] = df_capped["aqi_change_rate"].fillna(0)



# In[637]:


df_capped["timestamp"] = pd.to_datetime(df_capped["timestamp"])
int_cols = ["hour", "day", "month", "weekday", "is_weekend"]
# df_capped[int_cols] = df_capped[int_cols].astype(int)
# Replace infinities with 0
df_capped.replace([np.inf, -np.inf], 0, inplace=True)

# Fill NaNs in int columns before conversion
df_capped[int_cols] = df_capped[int_cols].fillna(0).astype(int)


# In[743]:


df_capped
# Make sure all column names are lower-case and have underscores
df_capped.columns = [c.lower().replace(" ", "_") for c in df_capped.columns]
fg_cols = ["timestamp","aqi","pm2_5","pm10","temperature","humidity","wind_speed",
           "hour","day","month","weekday","is_weekend","aqi_change","aqi_change_rate"]
df_capped["humidity"] = df_capped["humidity"].fillna(0).astype(int)
numeric_cols = ["temperature","wind_speed","pm2_5","pm10","aqi","aqi_change","aqi_change_rate"]
for col in numeric_cols:
    df_capped[col] = df_capped[col].fillna(0).astype(float)

# Ensure only these columns exist in df_capped
df_capped = df_capped[fg_cols]


# In[638]:


try:
    karachi_fg = fs.get_feature_group("karachi_aqi_features", version=1)
    karachi_fg.delete()
    print("🗑️ Old Feature Group deleted.")
except:
    print("ℹ️ No existing Feature Group found. Creating new one.")
print("⏳ Before get_or_create_feature_group")
# 🆕 Recreate and insert updated data
karachi_fg = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=1,
    primary_key=["timestamp"],
    description="Karachi AQI features"
)
print("Feature group ready:")

# # In[639]:


# # 4️⃣ Insert processed + capped data
# karachi_fg.insert(df_capped)
# print("✅ New processed data inserted successfully into Feature Store.")


# # In[640]:


# karachi_fg.update_feature_description("timestamp", "Datetime of recorded observation")
# karachi_fg.update_feature_description("temperature", "Temperature in Celsius")
# karachi_fg.update_feature_description("humidity", "Relative humidity (%)")
# karachi_fg.update_feature_description("pm2_5", "Concentration of fine particulate matter (PM2.5, µg/m³)")
# karachi_fg.update_feature_description("pm10", "Concentration of coarse particulate matter (PM10, µg/m³)")
# karachi_fg.update_feature_description("aqi", "Air Quality Index value (1–5 scale)")
# karachi_fg.update_feature_description("wind_speed", "Wind speed in meters per second (m/s)")
# karachi_fg.update_feature_description("hour", "Hour of the day (0–23)")
# karachi_fg.update_feature_description("day", "Day of the month (1–31)")
# karachi_fg.update_feature_description("month", "Month number (1–12)")
# karachi_fg.update_feature_description("weekday", "Day of the week (0=Monday, 6=Sunday)")
# karachi_fg.update_feature_description("is_weekend", "Weekend indicator (1=Saturday/Sunday, 0=Weekday)")
# karachi_fg.update_feature_description("aqi_change", "Change in AQI compared to previous reading")
# karachi_fg.update_feature_description("aqi_change_rate", "Rate of AQI change compared to previous reading")


# In[641]:

# df_capped = df_capped.iloc[:-1]
# print(df_capped)


# In[642]:


# !pip install confluent-kafka


# In[643]:


df_capped.info()


# In[677]:


# Step 1: Connect to Feature Store
import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Step 2: Get feature group
karachi_fg = fs.get_feature_group("karachi_aqi_features", version=1)

# # Step 3: Fetch all data for training
# data = karachi_fg.read()

# Step 4: Split into features (X) and target (y)
X = df_capped.drop(columns=["aqi", "timestamp"])
y = df_capped["aqi"]


# In[678]:


df_capped


# In[660]:


X.shape,y.shape


# In[679]:


df_capped["aqi"].describe()


# In[680]:


df_capped["aqi"].value_counts().sort_index()


# In[681]:


from sklearn.model_selection import train_test_split

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% test data
    random_state=42      # fixed seed for reproducibility
)

print(" Data split complete:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# **NON_PARAMETRIC ALGORITHMS**

# In[682]:


df_capped["aqi"].value_counts()


# In[715]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# ============================
# 🔹 Scale features
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================
# 🔹 Hyperparameters
# ============================
hyperparameters = [
    {'n_estimators': 100, 'max_depth': 10,  'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': 15,  'min_samples_split': 5},
    {'n_estimators': 300, 'max_depth': 20,  'min_samples_split': 3},
    {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 4},
]

lambda_weight = 1.0   # Weight for train-test RMSE gap
results = []
best_score = float('inf')
best_rf = None

# ============================
# 🔹 Train and evaluate models
# ============================
for i, params in enumerate(hyperparameters):
    print(f"\n============================")
    print(f"🌲 Random Forest - Config {i+1}")
    print(f"Hyperparameters: {params}")
    print(f"============================")

    rf = RandomForestRegressor(**params, random_state=42)
    rf.fit(X_train_scaled, y_train)

    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred  = rf.predict(X_test_scaled)

    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    test_rmse  = np.sqrt(metrics.mean_squared_error(y_test,  y_test_pred))
    train_r2   = metrics.r2_score(y_train, y_train_pred)
    test_r2    = metrics.r2_score(y_test,  y_test_pred)
    rmse_gap   = abs(train_rmse - test_rmse)
    r2_gap     = abs(train_r2 - test_r2)
    combined_score = test_rmse + lambda_weight * rmse_gap

    print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
    print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
    print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

    results.append({
        'Config': f'RF_{i+1}',
        'Hyperparameters': params,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'RMSE_Gap': rmse_gap,
        'R2_Gap': r2_gap,
        'Combined_Score': combined_score
    })

    # 🟢 Track the best RF model based on combined score
    if combined_score < best_score:
        best_score = combined_score
        best_rf = rf

# ============================
# 🔹 Convert results to DataFrame
# ============================
results_df = pd.DataFrame(results)
print("\n📊 Diagnostic Summary (Random Forest):")
print(results_df)

# ============================
# 🔹 Save the best RF model
# ============================
joblib.dump(best_rf, 'best_random_forest.pkl')
print("\nBest Random Forest model saved as 'best_random_forest.pkl'")

# ============================
# 🔹 Plot diagnostics
# ============================
plt.figure(figsize=(10,6))
plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'],
                 color='lightcoral', alpha=0.2)
plt.title("Random Forest — RMSE Comparison")
plt.ylabel("RMSE")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'],
                 color='lightblue', alpha=0.2)
plt.title("Random Forest — R² Comparison")
plt.ylabel("R²")
plt.legend()
plt.tight_layout()
plt.show()


# Config 4 model is best because it gives the highest test accuracy with the smallest gap between training and testing errors, meaning it generalizes well without overfitting.

# In[714]:


# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import joblib

# # ============================
# # 🔹 Scale features
# # ============================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ============================
# # 🔹 Hyperparameters to try
# # ============================
# hyperparameters = [
#     {'n_neighbors': 3},
#     {'n_neighbors': 5},
#     {'n_neighbors': 7},
#     {'n_neighbors': 9}
# ]

# lambda_weight = 1.0   # Weight for train-test RMSE gap
# results = []
# best_score = float('inf')
# best_knn = None

# # ============================
# # 🔹 Train and evaluate models
# # ============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"🌟 KNN Regressor - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     knn = KNeighborsRegressor(**params)
#     knn.fit(X_train_scaled, y_train)

#     y_train_pred = knn.predict(X_train_scaled)
#     y_test_pred  = knn.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test,  y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test,  y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     results.append({
#         'Config': f'KNN_{i+1}',
#         'Hyperparameters': params,
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

#     # 🟢 Track the best KNN model based on combined score
#     if combined_score < best_score:
#         best_score = combined_score
#         best_knn = knn

# # ============================
# # 🔹 Convert results to DataFrame
# # ============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (KNN):")
# print(results_df)

# # ============================
# # 🔹 Save the best KNN model
# # ============================
# joblib.dump(best_knn, 'best_knn.pkl')
# print("\nBest KNN model saved as 'best_knn.pkl'")

# # ============================
# # 🔹 Plot diagnostics
# # ============================
# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'],
#                  color='lightcoral', alpha=0.2)
# plt.title("KNN — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'],
#                  color='lightblue', alpha=0.2)
# plt.title("KNN — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()


# # we choose knn2 as lower rmse gap and good generalization

# # In[699]:


# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import joblib

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ============================
# # 🔹 Hyperparameters
# # ============================
# hyperparameters = [
#     {'n_estimators': 100, 'max_depth': 10},
#     {'n_estimators': 200, 'max_depth': 15},
#     {'n_estimators': 300, 'max_depth': None},
#     {'n_estimators': 150, 'max_depth': 20},
# ]

# lambda_weight = 1.0   # Weight for train-test RMSE gap
# results = []
# best_score = float('inf')
# best_model = None

# # ============================
# # 🔹 Train and evaluate models
# # ============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"🌳 Extra Trees Regressor - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     model = ExtraTreesRegressor(**params, random_state=42)
#     model.fit(X_train_scaled, y_train)

#     y_train_pred = model.predict(X_train_scaled)
#     y_test_pred  = model.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test, y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     results.append({
#         'Config': f'ET_{i+1}',
#         'Hyperparameters': params,
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

#     # 🟢 Track the best Extra Trees model (based on combined score)
#     if combined_score < best_score:
#         best_score = combined_score
#         best_model = model

# # ============================
# # 🔹 Convert results to DataFrame
# # ============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (Extra Trees):")
# print(results_df)

# # ============================
# # 🔹 Save the best Extra Trees model
# # ============================
# joblib.dump(best_model, 'best_extra_trees.pkl')
# print("\nBest Extra Trees model saved as 'best_extra_trees.pkl'")

# # ============================
# # 🔹 Plot diagnostics
# # ============================
# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'],
#                  color='lightcoral', alpha=0.2)
# plt.title("Extra Trees — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'],
#                  color='lightblue', alpha=0.2)
# plt.title("Extra Trees — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Copy results for later use
# et_results = results_df.copy()


# # Config 3,  demonstrates the best generalization by having the smallest $\text{R}^2$ Gap  and the smallest RMSE Gap .The other configurations (ET_2, ET_3, ET_4) all show signs of severe overfitting, despite achieving very high Test $\text{R}^2$ scores.

# # In[716]:


# import xgboost as xgb
# from sklearn import metrics
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib  # for saving the model

# from sklearn.preprocessing import StandardScaler


# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ============================
# # 🔹 Hyperparameters
# # ============================
# hyperparameters = [
#     {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
#     {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7},
#     {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 10},
#     {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8},
# ]

# lambda_weight = 1.0   # Weight for train-test RMSE gap
# results = []
# best_score = float('inf')
# best_model = None

# # ============================
# # 🔹 Train and evaluate models
# # ============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"⚡ XGBoost Regressor - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     model = xgb.XGBRegressor(**params, random_state=42)
#     model.fit(X_train_scaled, y_train)

#     y_train_pred = model.predict(X_train_scaled)
#     y_test_pred  = model.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test, y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     results.append({
#         'Config': f'XGB_{i+1}',
#         'Hyperparameters': params,
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

#     # 🟢 Track the best XGBoost model (based on combined score)
#     if combined_score < best_score:
#         best_score = combined_score
#         best_model = model

# # ============================
# # 🔹 Convert results to DataFrame
# # ============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (XGBoost):")
# print(results_df)

# # ============================
# # 🔹 Save the best XGBoost model
# # ============================
# joblib.dump(best_model, 'best_xgboost.pkl')
# print("\nBest XGBoost model saved as 'best_xgboost.pkl'")

# # ============================
# # 🔹 Plot diagnostics
# # ============================
# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'],
#                  color='lightcoral', alpha=0.2)
# plt.title("XGBoost — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'],
#                  color='lightblue', alpha=0.2)
# plt.title("XGBoost — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Copy results for later use
# xgb_results = results_df.copy()


# # he best model is Config 2

# # In[703]:


# from sklearn.svm import SVR
# from sklearn import metrics
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib  # for saving the model
# from sklearn.preprocessing import StandardScaler



# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ============================
# # 🔹 Hyperparameters
# # ============================
# hyperparameters = [
#     {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1},
#     {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2},
#     {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'epsilon': 0.1},
#     {'kernel': 'sigmoid', 'C': 5.0, 'epsilon': 0.2}
# ]

# lambda_weight = 1.0   # Weight for train-test RMSE gap
# results = []
# best_score = float('inf')
# best_model = None

# # ============================
# # 🔹 Train and evaluate models
# # ============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"💠 SVR Regressor - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     model = SVR(**params)
#     model.fit(X_train_scaled, y_train)

#     y_train_pred = model.predict(X_train_scaled)
#     y_test_pred  = model.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test, y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     results.append({
#         'Config': f'SVR_{i+1}',
#         'Hyperparameters': params,
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

#     # 🟢 Track the best SVR model (based on combined score)
#     if combined_score < best_score:
#         best_score = combined_score
#         best_model = model

# # ============================
# # 🔹 Convert results to DataFrame
# # ============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (SVR):")
# print(results_df)

# # ============================
# # 🔹 Save the best SVR model
# # ============================
# joblib.dump(best_model, 'best_svr.pkl')
# print("\nBest SVR model saved as 'best_svr.pkl'")

# # ============================
# # 🔹 Plot diagnostics
# # ============================
# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'],
#                  color='lightcoral', alpha=0.2)
# plt.title("SVR — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'],
#                  color='lightblue', alpha=0.2)
# plt.title("SVR — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Copy results for later use
# svr_results = results_df.copy()


# # config 1

# # **Parametric Algorithms**

# # In[717]:


# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import joblib

# # ==============================
# # 🔹 Scale the features
# # ==============================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ==============================
# # 🔹 Linear Regression Model
# # ==============================
# lambda_weight = 1.0   # Weight for train-test RMSE gap
# best_score = float('inf')
# best_lr = None
# results = []

# # For Linear Regression, usually only one config
# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)

# y_train_pred = lr.predict(X_train_scaled)
# y_test_pred  = lr.predict(X_test_scaled)

# train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
# test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
# train_r2   = metrics.r2_score(y_train, y_train_pred)
# test_r2    = metrics.r2_score(y_test, y_test_pred)
# rmse_gap   = abs(train_rmse - test_rmse)
# r2_gap     = abs(train_r2 - test_r2)
# combined_score = test_rmse + lambda_weight * rmse_gap

# # Track best model (though just one here)
# if combined_score < best_score:
#     best_score = combined_score
#     best_lr = lr

# print("\n============================")
# print("📈 Linear Regression Summary")
# print("============================")
# print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
# print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
# print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

# # ==============================
# # 🔹 Store results
# # ==============================
# results.append({
#     'Config': 'LR_1',
#     'Train_RMSE': train_rmse,
#     'Test_RMSE': test_rmse,
#     'Train_R2': train_r2,
#     'Test_R2': test_r2,
#     'RMSE_Gap': rmse_gap,
#     'R2_Gap': r2_gap,
#     'Combined_Score': combined_score
# })

# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (Linear Regression):")
# print(results_df)

# # ==============================
# # 🔹 Save best model
# # ==============================
# joblib.dump(best_lr, 'best_linear_regression.pkl')
# print("\nBest Linear Regression model saved as 'best_linear_regression.pkl'")

# # ==============================
# # 📉 Diagnostic Plots
# # ==============================
# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'], color='lightcoral', alpha=0.2)
# plt.title("Linear Regression — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'], color='lightblue', alpha=0.2)
# plt.title("Linear Regression — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# lr_results = results_df.copy()


# # 

# # In[689]:


# df["aqi"].describe()


# # In[705]:


# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import joblib

# # ==============================
# # 🔹 Scale features
# # ==============================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ==============================
# # 🔹 Ridge Regression Hyperparameters
# # ==============================
# hyperparameters = [
#     {'alpha': 0.1},
#     {'alpha': 1.0},
#     {'alpha': 10.0},
#     {'alpha': 50.0}
# ]

# lambda_weight = 1.0
# best_score = float('inf')
# best_ridge = None
# results = []

# # ==============================
# # 🔹 Train and Evaluate
# # ==============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"📈 Ridge Regression - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     ridge = Ridge(**params)
#     ridge.fit(X_train_scaled, y_train)

#     y_train_pred = ridge.predict(X_train_scaled)
#     y_test_pred  = ridge.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test, y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     # Track best Ridge model
#     if combined_score < best_score:
#         best_score = combined_score
#         best_ridge = ridge

#     results.append({
#         'Config': f'Ridge_{i+1}',
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

# # ==============================
# # 🔹 Convert results to DataFrame
# # ==============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (Ridge Regression):")
# print(results_df)

# # ==============================
# # 🔹 Save the best Ridge model
# # ==============================
# joblib.dump(best_ridge, 'best_ridge.pkl')
# print("\nBest Ridge model saved as 'best_ridge.pkl'")

# # ==============================
# # 📉 Diagnostic Plots
# # ==============================
# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'], color='lightcoral', alpha=0.2)
# plt.title("Ridge Regression — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'], color='lightblue', alpha=0.2)
# plt.title("Ridge Regression — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# ridge_results = results_df.copy()


# # config 2

# # In[718]:


# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Lasso
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import joblib

# # ==============================
# # 🔹 Scale features
# # ==============================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ==============================
# # 🔹 Lasso Regression Hyperparameters
# # ==============================
# hyperparameters = [
#     {'alpha': 0.001},
#     {'alpha': 0.01},
#     {'alpha': 0.1},
#     {'alpha': 1.0}
# ]

# lambda_weight = 1.0
# best_score = float('inf')
# best_lasso = None
# results = []

# # ==============================
# # 🔹 Train and Evaluate
# # ==============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"📈 Lasso Regression - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     lasso = Lasso(**params, max_iter=5000)
#     lasso.fit(X_train_scaled, y_train)

#     y_train_pred = lasso.predict(X_train_scaled)
#     y_test_pred  = lasso.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test, y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     # Track best Lasso model
#     if combined_score < best_score:
#         best_score = combined_score
#         best_lasso = lasso

#     results.append({
#         'Config': f'Lasso_{i+1}',
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

# # ==============================
# # 🔹 Convert results to DataFrame
# # ==============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (Lasso Regression):")
# print(results_df)

# # ==============================
# # 🔹 Save the best Lasso model
# # ==============================
# joblib.dump(best_lasso, 'best_lasso.pkl')
# print("\nBest Lasso model saved as 'best_lasso.pkl'")

# # ==============================
# # 📉 Diagnostic Plots
# # ==============================
# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'], color='lightcoral', alpha=0.2)
# plt.title("Lasso Regression — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'], color='lightblue', alpha=0.2)
# plt.title("Lasso Regression — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# lasso_results = results_df.copy()


# # config 1

# # In[719]:


# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import ElasticNet
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import joblib

# # ==============================
# # 🔹 Scale features
# # ==============================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ==============================
# # 🔹 ElasticNet Hyperparameters
# # ==============================
# hyperparameters = [
#     {'alpha': 0.01, 'l1_ratio': 0.2},
#     {'alpha': 0.1,  'l1_ratio': 0.5},
#     {'alpha': 1.0,  'l1_ratio': 0.7},
#     {'alpha': 10.0, 'l1_ratio': 0.9}
# ]

# lambda_weight = 1.0
# best_score = float('inf')
# best_enet = None
# results = []

# # ==============================
# # 🔹 Train and Evaluate
# # ==============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"📈 Elastic Net Regression - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     enet = ElasticNet(**params, max_iter=5000)
#     enet.fit(X_train_scaled, y_train)

#     y_train_pred = enet.predict(X_train_scaled)
#     y_test_pred  = enet.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test, y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     # Track best ElasticNet model
#     if combined_score < best_score:
#         best_score = combined_score
#         best_enet = enet

#     results.append({
#         'Config': f'ENet_{i+1}',
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

# # ==============================
# # 🔹 Convert results to DataFrame
# # ==============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (Elastic Net):")
# print(results_df)

# # ==============================
# # 🔹 Save the best ElasticNet model
# # ==============================
# joblib.dump(best_enet, 'best_elasticnet.pkl')
# print("\nBest ElasticNet model saved as 'best_elasticnet.pkl'")

# # ==============================
# # 📉 Diagnostic Plots
# # ==============================
# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'], color='lightcoral', alpha=0.2)
# plt.title("Elastic Net Regression — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'], color='lightblue', alpha=0.2)
# plt.title("Elastic Net Regression — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# enet_results = results_df.copy()


# # config 1

# # In[720]:


# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import joblib

# # ==============================
# # 🔹 Scale input features
# # ==============================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# # ==============================
# # 🔹 ANN Hyperparameters
# # ==============================
# hyperparameters = [
#     {'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'alpha': 0.001},
#     {'hidden_layer_sizes': (128, 64, 32), 'activation': 'relu', 'alpha': 0.0001},
#     {'hidden_layer_sizes': (64, 64, 64), 'activation': 'tanh', 'alpha': 0.0005}
# ]

# lambda_weight = 1.0
# best_score = float('inf')
# best_ann = None
# results = []

# # ==============================
# # 🔹 Train and Evaluate
# # ==============================
# for i, params in enumerate(hyperparameters):
#     print(f"\n============================")
#     print(f"🧠 ANN Regressor - Config {i+1}")
#     print(f"Hyperparameters: {params}")
#     print(f"============================")

#     ann = MLPRegressor(
#         **params,
#         solver='adam',
#         max_iter=1000,
#         random_state=42
#     )

#     ann.fit(X_train_scaled, y_train)

#     y_train_pred = ann.predict(X_train_scaled)
#     y_test_pred  = ann.predict(X_test_scaled)

#     train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
#     test_rmse  = np.sqrt(metrics.mean_squared_error(y_test,  y_test_pred))
#     train_r2   = metrics.r2_score(y_train, y_train_pred)
#     test_r2    = metrics.r2_score(y_test,  y_test_pred)
#     rmse_gap   = abs(train_rmse - test_rmse)
#     r2_gap     = abs(train_r2 - test_r2)
#     combined_score = test_rmse + lambda_weight * rmse_gap

#     print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
#     print(f"Train R²  : {train_r2:.3f} | Test R² : {test_r2:.3f}")
#     print(f"Combined Score (RMSE + λ*Gap): {combined_score:.3f}")

#     # Track the best ANN model
#     if combined_score < best_score:
#         best_score = combined_score
#         best_ann = ann

#     results.append({
#         'Config': f'ANN_{i+1}',
#         'Train_RMSE': train_rmse,
#         'Test_RMSE': test_rmse,
#         'Train_R2': train_r2,
#         'Test_R2': test_r2,
#         'RMSE_Gap': rmse_gap,
#         'R2_Gap': r2_gap,
#         'Combined_Score': combined_score
#     })

# # ==============================
# # 🔹 Convert results to DataFrame
# # ==============================
# results_df = pd.DataFrame(results)
# print("\n📊 Diagnostic Summary (ANN):")
# print(results_df)

# # ==============================
# # 🔹 Save the best ANN model
# # ==============================
# joblib.dump(best_ann, 'best_ann.pkl')
# print("\nBest ANN model saved as 'best_ann.pkl'")

# # ==============================
# # 📉 Diagnostic Plots
# # ==============================
# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_RMSE'], marker='o', label='Train RMSE')
# plt.plot(results_df['Config'], results_df['Test_RMSE'], marker='s', label='Test RMSE')
# plt.fill_between(results_df['Config'], results_df['Train_RMSE'], results_df['Test_RMSE'], color='lightcoral', alpha=0.2)
# plt.title("ANN — RMSE Comparison")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(results_df['Config'], results_df['Train_R2'], marker='o', label='Train R²')
# plt.plot(results_df['Config'], results_df['Test_R2'], marker='s', label='Test R²')
# plt.fill_between(results_df['Config'], results_df['Train_R2'], results_df['Test_R2'], color='lightblue', alpha=0.2)
# plt.title("ANN — R² Comparison")
# plt.ylabel("R²")
# plt.legend()
# plt.tight_layout()
# plt.show()

# ann_results = results_df.copy()


# The best model is **Config 2

# In[722]:


# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import matplotlib.pyplot as plt

# # ==========================================
# # 🔹 Step 1: Ensure timestamp & sort
# # ==========================================
# df_capped["timestamp"] = pd.to_datetime(df_capped["timestamp"])
# df_capped = df_capped.sort_values("timestamp").reset_index(drop=True)

# # Keep last row for features
# last_known = df_capped.iloc[-1:].copy()

# # ==========================================
# # 🔹 Step 2: Generate next 3 future dates (daily)
# # ==========================================
# future_dates = [last_known["timestamp"].iloc[0] + timedelta(days=i) for i in range(1, 4)]

# # Duplicate last row for features
# future_df = pd.concat([last_known]*3, ignore_index=True)
# future_df["timestamp"] = future_dates

# # ==========================================
# # 🔹 Step 3: Generate time-based features (same as training)
# # ==========================================
# future_df["hour"] = future_df["timestamp"].dt.hour
# future_df["day"] = future_df["timestamp"].dt.day
# future_df["month"] = future_df["timestamp"].dt.month
# future_df["weekday"] = future_df["timestamp"].dt.weekday
# future_df["is_weekend"] = (future_df["weekday"] >= 5).astype(int)

# # Keep AQI-change placeholders if your model used them
# future_df["aqi_change"] = 0
# future_df["aqi_change_rate"] = 0

# # ==========================================
# # 🔹 Step 4: Prepare feature matrix (same columns & scaling as train)
# # ==========================================
# X_future = future_df.drop(columns=["aqi", "timestamp"], errors="ignore")
# X_future = X_future[X_train.columns]  # ensure exact same order
# X_future_scaled = scaler.transform(X_future)

# # ==========================================
# # 🔹 Step 5: Predict with all models
# # ==========================================
# models_all = {
#     "LinearRegression": lr,
#     "Ridge": ridge,
#     "Lasso": lasso,
#     "ElasticNet": enet,
#     "RandomForest": best_rf,
#     "ExtraTrees": best_etr,
#     "XGBoost": best_xgbr,
#     "KNN": best_knn,
#     "SVR": best_svr,
#     "ANN": best_ann
# }

# future_predictions = {}
# for name, model in models_all.items():
#     try:
#         preds = model.predict(X_future_scaled)
#         future_predictions[name] = preds
#     except Exception as e:
#         print(f"⚠️ Could not predict with {name}: {e}")

# future_results = pd.DataFrame(future_predictions, index=future_dates)
# future_results.index.name = "Future Timestamp"

# # ==========================================
# # 🔹 Step 6: Display predictions table
# # ==========================================
# print("📈 AQI Prediction for Next 3 Days (All Models):")
# display(future_results.round(2))

# # ==========================================
# # 🔹 Step 7: Plot past vs predicted future
# # ==========================================
# plt.figure(figsize=(12,6))

# # Plot last 10 actual AQI readings
# plt.plot(df_capped["timestamp"].tail(10), df_capped["aqi"].tail(10), label="Past AQI", color="black", linewidth=2, marker="o")

# # Plot predictions for next 3 days
# for col in future_results.columns:
#     plt.plot(future_results.index, future_results[col], linestyle="--", marker="s", alpha=0.8, label=col)

# plt.title("📈 AQI Trend — Past vs Next 3 Days Prediction")
# plt.xlabel("Timestamp")
# plt.ylabel("AQI Level")
# plt.legend(ncol=3, fontsize=8)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

# # ==========================================
# # 🔹 Step 8: Prediction summary
# # ==========================================
# print("📊 Prediction Summary Statistics (Next 3 Days):")
# display(future_results.describe().T.round(2))


# In[723]:


df_capped.tail(5)


# In[740]:


import joblib

# Save the best Random Forest model locally
model_file = "best_random_forest.pkl"
joblib.dump(best_rf, model_file)
print("✅ Best Random Forest model saved locally as 'best_random_forest.pkl'")
print(df_capped.head(5))
print("above rows and below rows")
print(df_capped.tail(5))
# In[741]:

# ==========================================
# 🔹 Streamlit Prediction Function
# ==========================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

def get_forecast(df_capped, model_path="best_random_forest.pkl",save_csv=True):
    """
    Generate AQI forecast for next 3 days using saved Random Forest model.
    Temperature, humidity, windspeed are averaged from last 3 days.
    
    Parameters:
    - df_capped: pd.DataFrame containing historical 5-hourly data with columns
                 ['timestamp','aqi','temperature','humidity','windspeed', ...]
    - model_path: path to saved Random Forest model
    
    Returns:
    - forecast_data: list of dicts with keys ['date','day_name','aqi_min','aqi_max',
                                              'temperature','humidity','windspeed']
    """

    # 🔹 Load saved model
    rf_model = joblib.load(model_path)

    # 🔹 Ensure timestamp is datetime & sort
    df_capped["timestamp"] = pd.to_datetime(df_capped["timestamp"])
    df_capped = df_capped.sort_values("timestamp").reset_index(drop=True)

    # 🔹 Last row for features
    last_row = df_capped.iloc[-1:].copy()

    # 🔹 Compute last 3 days average for temp, humidity, windspeed
    three_days_ago = df_capped["timestamp"].max() - pd.Timedelta(days=3)
    recent_3days = df_capped[df_capped["timestamp"] > three_days_ago]

    avg_temp = int(recent_3days["temperature"].mean())
    avg_humidity = int(recent_3days["humidity"].mean())
    avg_wind = int(recent_3days["wind_speed"].mean())

    # 🔹 Generate forecast for next 3 days
    forecast_dates = [datetime.now() + timedelta(days=i) for i in range(3)]
    day_names = ["Today", "Tomorrow", "Day After"]

    forecast_data = []
    for i, date in enumerate(forecast_dates):
        # Copy last known row & update timestamp
        future_row = last_row.copy()
        future_row["timestamp"] = date

        # Prepare features (drop aqi & timestamp)
        X_future = future_row.drop(columns=["aqi", "timestamp"], errors="ignore").values.reshape(1, -1)
        predicted_aqi = int(rf_model.predict(X_future)[0])

        forecast_data.append({
            "date": date,
            "day_name": day_names[i],
            "aqi_min": max(predicted_aqi - 5, 0),
            "aqi_max": predicted_aqi + 5,
            "temperature": avg_temp,
            "humidity": avg_humidity,
            "wind_speed": avg_wind
        })
        # 🔹 Convert to DataFrame
    forecast_df = pd.DataFrame(forecast_data)

    # 🔹 Save CSV if needed
    if save_csv:
        forecast_df.to_csv("latest_predictions.csv", index=False)
        print("✅ Forecast saved to 'latest_predictions.csv'")

    return forecast_data

forecast_data = get_forecast(df_capped, model_path="best_random_forest.pkl", save_csv=True)
print("✅ latest_predictions.csv is ready for app.py")
   
