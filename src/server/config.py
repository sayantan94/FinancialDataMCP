import os

# === Securely Load Twelve Data API Key ===
try:
    with open("/etc/twelve_data_api_key.txt", "r") as f:
        TWELVE_DATA_API_KEY: str = f.read().strip()
except FileNotFoundError:
    TWELVE_DATA_API_KEY: str = ""
    print("Warning: API key file not found at /etc/twelve_data_api_key.txt")
except Exception as e:
    TWELVE_DATA_API_KEY: str = ""
    print(f"Warning: Failed to read API key: {e}")

# === Server Settings ===
FININSIGHT_HOST: str = "0.0.0.0"
FININSIGHT_PORT: int = 8100

# === Twelve Data API Settings ===
TWELVE_DATA_BASE_URL: str = "https://api.twelvedata.com"
TWELVE_DATA_OUTPUT_SIZE: int = 5000


# === 1-Minute Timeframe (Scalping / 0DTE Setup) ===
VP_1M_BARS_INTERVAL: str = "1min"
VP_1M_LOOKBACK_HOURS: int = 8       # Covers full trading day with buffer
TA_1M_INDICATOR_INTERVAL: str = "5min"

# === 5-Minute Timeframe (Intraday / 2-3DTE Setup) ===
VP_5M_BARS_INTERVAL: str = "5min"
VP_5M_LOOKBACK_DAYS: int = 5        # Recent week structure
TA_5M_INDICATOR_INTERVAL: str = "15min"

# === Daily Timeframe (Swing Context) ===
VP_1D_BARS_INTERVAL: str = "1h"     # 1h bars over several months
VP_1D_LOOKBACK_MONTHS: int = 3      # Recent quarter structure
TA_1D_INDICATOR_INTERVAL: str = "1day"