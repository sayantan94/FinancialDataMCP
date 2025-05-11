import requests
import pandas as pd
from typing import Optional, Dict, Any, List, Union

from src.server.config import TWELVE_DATA_BASE_URL, TWELVE_DATA_API_KEY, TWELVE_DATA_OUTPUT_SIZE


def _make_twelvedata_request(
        endpoint: str,
        params: Dict[str, Any],
        symbol: Optional[str] = None # Add symbol for better error reporting
) -> Optional[Dict[str, Any]]:
    """Helper function to make SYNCHRONOUS requests to Twelve Data API and handle common errors."""
    url = f"{TWELVE_DATA_BASE_URL}/{endpoint}"
    all_params = params.copy()
    all_params["apikey"] = TWELVE_DATA_API_KEY
    all_params["format"] = "json" # Always request JSON

    if not TWELVE_DATA_API_KEY:
        print(f"Error: TWELVE_DATA_API_KEY is not set in config. Cannot request {endpoint}.")
        return None # Indicate critical configuration error

    try:
        response = requests.get(url, params=all_params)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Twelve Data specific error handling (check 'status' and 'code')
        if data.get("status") == "error":
            error_message = data.get('message', 'Unknown API error')
            error_code = data.get('code', 'N/A')
            symbol_info = symbol if symbol else params.get('symbol', 'N/A')
            print(f"Twelve Data API error ({endpoint}) for symbol {symbol_info} [Code: {error_code}]: {error_message}")
            # Return None for API-reported errors that indicate the data is genuinely unavailable
            # Add other critical codes if needed
            if error_code in [400, 401, 403]:
                return None
            else:
                # For other errors (e.g., rate limit, internal server error at Twelve Data)
                # Treat as fetch failure for simplicity
                return None

        # Handle successful response but with no data values
        if "values" not in data or not data["values"]:
            pass # Let the calling function check for data["values"]

        return data # Return the full data dictionary

    except requests.exceptions.RequestException as e:
        symbol_info = symbol if symbol else params.get('symbol', 'N/A')
        print(f"Network/Request error fetching {endpoint} from Twelve Data for symbol {symbol_info}: {e}")
        return None
    except Exception as e:
        symbol_info = symbol if symbol else params.get('symbol', 'N/A')
        print(f"An unexpected error occurred processing Twelve Data response for {endpoint} ({symbol_info}): {e}")
        return None


# --- Fetching Time Series (Needed for Volume Profile Calculation) ---
def fetch_time_series(
        symbol: str,
        interval: str,
        outputsize: int = TWELVE_DATA_OUTPUT_SIZE,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetches time series data (OHLCV bars) from Twelve Data API (Synchronous)."""
    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    data = _make_twelvedata_request("time_series", params, symbol=symbol)

    if data is None:
        return None

    if "values" not in data or not data["values"]:
        print(f"No time series data received for {symbol} with interval {interval} in specified range.")
        return pd.DataFrame() # Return empty DataFrame for no data found


    df = pd.DataFrame(data["values"])
    df = df.iloc[::-1].reset_index(drop=True) # Reverse to chronological order

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)

    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    return df

# --- Fetching Pre-calculated TA Indicators ---
def fetch_indicator(
        indicator_name: str, # e.g., "SMA", "RSI", etc.
        symbol: str,
        interval: str,
        outputsize: int = TWELVE_DATA_OUTPUT_SIZE,
        params: Optional[Dict[str, Any]] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches a specific pre-calculated TA indicator from Twelve Data (Synchronous).
    Returns None on critical fetch error, [] on no data points, or a list of dicts.
    Ensures returned list only contains dictionaries.
    """
    endpoint = indicator_name.lower()
    all_params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
    }
    if params:
        all_params.update(params)

    data = _make_twelvedata_request(endpoint, all_params, symbol=symbol)
    #print(data)

    if data is None:
        return None

    if "values" not in data or not data["values"]:
        print(f"No indicator data received for {indicator_name} on {symbol} ({interval}).")
        return []

    values = data["values"]

    if not all(isinstance(item, dict) for item in values):
        print(f"Warning: Received unexpected non-dict items in '{indicator_name}' values for {symbol} ({interval}). Returning empty list.")
        return []

    return values[::-1] # Reverse to chronological order (oldest first)


# --- Specific helper functions for common indicators (Synchronous Wrappers) ---
# UPDATED: Check for indicator-specific keys
def fetch_sma(symbol: str, interval: str, time_period: int, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:
    data = fetch_indicator("SMA", symbol, interval, outputsize, {"time_period": time_period})
    if data is None: return None # Fetch failed
    if not data: return [] # No data points returned

    # Check the last item for the specific 'sma' key
    last_item = data[-1]
    if not isinstance(last_item, dict) or 'sma' not in last_item or last_item.get('sma') is None:
        print(f"Warning: Last SMA data item for {symbol} ({interval}) missing or invalid 'sma' key. Returning empty list.")
        return []
    # Ensure all items in the list have the 'sma' key if the last one does
    # (Could add check here or rely on the consumer to handle potential missing keys in older items if needed)
    return data


def fetch_ema(symbol: str, interval: str, time_period: int, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:
    data = fetch_indicator("EMA", symbol, interval, outputsize, {"time_period": time_period})
    if data is None: return None
    if not data: return []

    # Check for the specific 'ema' key
    last_item = data[-1]
    if not isinstance(last_item, dict) or 'ema' not in last_item or last_item.get('ema') is None:
        print(f"Warning: Last EMA data item for {symbol} ({interval}) missing or invalid 'ema' key. Returning empty list.")
        return []
    return data


def fetch_rsi(symbol: str, interval: str, time_period: int = 14, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:
    data = fetch_indicator("RSI", symbol, interval, outputsize, {"time_period": time_period})
    if data is None: return None
    if not data: return []

    # Check for the specific 'rsi' key (confirmed by your sample)
    last_item = data[-1]
    if not isinstance(last_item, dict) or 'rsi' not in last_item or last_item.get('rsi') is None:
        print(f"Warning: Last RSI data item for {symbol} ({interval}) missing or invalid 'rsi' key. Returning empty list.")
        return []
    return data

def fetch_stoch(symbol: str, interval: str, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:

    print("stoch params")
    data = fetch_indicator("STOCH", symbol, interval, outputsize, {})
    #print(data)
    if data is None: return None
    if not data: return []

    # Check for specific 'k' and 'd' keys
    last_item = data[-1]
    if not isinstance(last_item, dict) or 'slow_k' not in last_item or 'slow_d' not in last_item or last_item.get('slow_k') is None or last_item.get('slow_d') is None:
        print(f"Warning: Last STOCH data item for {symbol} ({interval}) missing or invalid 'slow_k' or 'slow_d'. Returning empty list.")
        return []
    return data


def fetch_macd(symbol: str, interval: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:
    params = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
    data = fetch_indicator("MACD", symbol, interval, outputsize, params)
    if data is None: return None
    if not data: return []

    # Check for specific 'macd', 'hist', 'signal' keys.
    last_item = data[-1]
    # Require at least one of the primary keys to be present and not None
    if not isinstance(last_item, dict) or not any(last_item.get(key) is not None for key in ['macd', 'hist', 'signal']):
        print(f"Warning: Last MACD data item for {symbol} ({interval}) missing or invalid expected keys (macd, hist, signal). Returning empty list.")
        return []
    # Note: Twelve Data might return 'value' instead of 'macd' for the MACD line sometimes.
    # The API endpoint's processor will handle this using .get().
    return data


def fetch_atr(symbol: str, interval: str, time_period: int = 14, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:
    data = fetch_indicator("ATR", symbol, interval, outputsize, {"time_period": time_period})
    if data is None: return None
    if not data: return []

    # Check for the specific 'atr' key
    last_item = data[-1]
    if not isinstance(last_item, dict) or 'atr' not in last_item or last_item.get('atr') is None:
        print(f"Warning: Last ATR data item for {symbol} ({interval}) missing or invalid 'atr' key. Returning empty list.")
        return []
    return data

def fetch_vwap(symbol: str, interval: str, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[List[Dict[str, Any]]]:
    data = fetch_indicator("VWAP", symbol, interval, outputsize)
    if data is None: return None
    if not data: return []

    # Check for the specific 'vwap' key
    last_item = data[-1]
    if not isinstance(last_item, dict) or 'vwap' not in last_item or last_item.get('vwap') is None:
        print(f"Warning: Last VWAP data item for {symbol} ({interval}) missing or invalid 'vwap' key. Returning empty list.")
        return []
    return data


# TODO : Update this method to get GEX and DEX
def fetch_options_chain(symbol: str, expiry_date: str, outputsize: int = TWELVE_DATA_OUTPUT_SIZE) -> Optional[Dict[str, Any]]: # Options chain structure is different
    """Placeholder for fetching options chain data (Synchronous)."""
    print(f"Fetching options chain for {symbol} expiring {expiry_date} - NOT IMPLEMENTED YET")
    data = _make_twelvedata_request("options_chain", {"symbol": symbol, "date": expiry_date}, symbol=symbol)
    if data is None: return None # Fetch failed


    if data and 'calls' in data and 'puts' in data and isinstance(data['calls'], list) and isinstance(data['puts'], list):
        # Basic structure check passed
        return data # Return the raw data structure
    else:
        print(f"Warning: Options chain data for {symbol} expiring {expiry_date} missing expected structure (calls/puts lists). Returning None.")
        return None


# Example usage (for testing the fetcher functions individually)
if __name__ == "__main__":
    # Ensure you have a .env file with TWELVE_DATA_API_KEY set up at the project root
    # Run this file with `python src/server/data/twelvedata_fetcher.py`

    test_symbol = "AAPL"
    test_invalid_symbol = "INVALIDSYMBOL123" # Example invalid symbol
    test_interval_intraday = "15min"
    test_interval_daily = "1day"
    test_invalid_interval = "1sec" # Example invalid interval

    print(f"--- Testing Synchronous Fetcher for {test_symbol} ---")

    print(f"\nFetching Time Series ({test_interval_intraday}):")
    ts_df = fetch_time_series(symbol=test_symbol, interval=test_interval_intraday, outputsize=10)
    if ts_df is not None:
        print(ts_df)
    else:
        print("Failed to fetch time series (returned None).")
    if ts_df is not None and ts_df.empty:
        print("Time series fetch successful, but returned empty DataFrame.")


    print(f"\nFetching SMA 50 ({test_interval_daily}):")
    sma_data = fetch_sma(symbol=test_symbol, interval=test_interval_daily, time_period=50, outputsize=5)
    if sma_data is not None:
        print(sma_data)
    else:
        print("Failed to fetch SMA (returned None).")
    if sma_data is not None and not sma_data: # Check if list is empty after validation
        print("SMA fetch successful, but returned empty list after validation.")


    print(f"\nFetching MACD ({test_interval_intraday}):")
    macd_data = fetch_macd(symbol=test_symbol, interval=test_interval_intraday, outputsize=5)
    if macd_data is not None:
        print(macd_data)
    else:
        print("Failed to fetch MACD (returned None).")
    if macd_data is not None and not macd_data:
        print("MACD fetch successful, but returned empty list after validation.")

    print(f"\nFetching STOCH ({test_interval_intraday}):")
    stoch_data = fetch_stoch(symbol=test_symbol, interval=test_interval_intraday, outputsize=5)
    if stoch_data is not None:
        print(stoch_data)
    else:
        print("Failed to fetch STOCH (returned None).")
    if stoch_data is not None and not stoch_data:
        print("STOCH fetch successful, but returned empty list after validation.")


    print(f"\nFetching Options Chain (Placeholder):")
    fetch_options_chain(symbol=test_symbol, expiry_date="2024-12-31") # Use a valid future date if testing

    # --- Test error handling ---
    print("\n--- Testing Fetcher Error Handling ---")

    print(f"\nFetching Time Series (Invalid Symbol: {test_invalid_symbol}):")
    ts_invalid_df = fetch_time_series(symbol=test_invalid_symbol, interval=test_interval_intraday, outputsize=10)
    if ts_invalid_df is None:
        print("Correctly returned None for invalid symbol (critical error).")
    elif ts_invalid_df is not None and ts_invalid_df.empty: # Check if df is not None AND empty
        print("Correctly returned empty DataFrame for invalid symbol (no data).")
    else:
        print("Unexpectedly fetched data for invalid symbol.")

    print(f"\nFetching SMA (Invalid Interval: {test_invalid_interval}):")
    sma_invalid_interval = fetch_sma(symbol=test_symbol, interval=test_invalid_interval, time_period=14, outputsize=10)
    if sma_invalid_interval is None:
        print("Correctly returned None for invalid interval (critical error).")
    elif not sma_invalid_interval: # Check if the list is empty
        print("Correctly returned empty list for invalid interval (no data after validation).")
    else:
        print("Unexpectedly fetched data for invalid interval.")