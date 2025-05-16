from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta
import concurrent.futures

from src.server.config import TA_1M_INDICATOR_INTERVAL, TA_5M_INDICATOR_INTERVAL, TA_1D_INDICATOR_INTERVAL
from src.server.data import twelvedata_fetcher
from src.server.compute import technical_analysis  # Import our custom technical analysis calculations

router = APIRouter()

@router.get("/technical_analysis/{symbol}")
def get_technical_analysis(
        symbol: str
):
    """
    Retrieves pre-calculated Technical Analysis indicators from Twelve Data
    for default timeframes (1m, 5m, 1d) in parallel.
    Handles potential fetch errors and partial data.
    """
    # Define the default timeframes and their settings for Technical Analysis
    timeframes_config = {
        "1m": {
            "ta_indicator_interval": TA_1M_INDICATOR_INTERVAL,
        },
        "5m": {
            "ta_indicator_interval": TA_5M_INDICATOR_INTERVAL,
        },
        "1d": {
            "ta_indicator_interval": TA_1D_INDICATOR_INTERVAL,
        }
    }

    # TODO: If holiday rollback to last trading day
    response_time_utc = datetime.utcnow()
    # If it's Saturday (5) or Sunday (6), roll back to last Friday
    if response_time_utc.weekday() == 5:  # Saturday
        response_time_utc -= timedelta(days=1)
    elif response_time_utc.weekday() == 6:  # Sunday
        response_time_utc -= timedelta(days=2)

    timeframe_technical_analysis: Dict[str, Any] = {}
    overall_status = "success" # Can be "success", "partial_success", "error"
    overall_message = "Technical Analysis data processed."

    def fetch_and_process_indicator(fetch_func, category, key, transform_func, *args):
        try:
            result = fetch_func(*args)
            if result is None:
                return "error", f"{fetch_func.__name__} fetch failed.", None
            elif result:
                try:
                    value = transform_func(result)
                    return "success", "", (category, key, value)
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Parse/Access error {fetch_func.__name__} for {args}: {str(e)}")
                    return "partial_success", f"{fetch_func.__name__} parse/access failed.", None
            return "warning", f"No {fetch_func.__name__} data available.", None
        except Exception as e:
            print(f"Exception in {fetch_func.__name__} for {args}: {str(e)}")
            return "error", f"{fetch_func.__name__} exception: {str(e)}", None

    def calculate_and_process_indicator(calc_func, category, key, *args):
        """Helper function for calculated indicators (like Ichimoku) rather than fetched ones"""
        try:
            result = calc_func(*args)
            if result is None:
                return "error", f"{calc_func.__name__} calculation failed.", None
            return "success", "", (category, key, result)
        except Exception as e:
            print(f"Exception in {calc_func.__name__} for {args}: {str(e)}")
            return "error", f"{calc_func.__name__} exception: {str(e)}", None

    # Process a single timeframe with parallel indicator fetching
    def process_timeframe(timeframe_key, ta_indicator_interval, symbol):
        # --- Initialize TA data structure ---
        ta_data: Dict[str, Any] = {
            "moving_averages": {},
            "oscillators": {},
            "volatility": {},
            "vwap": {},
            "ichimoku": {}
        }

        # Status for this specific timeframe: "success", "partial_success", "error", "warning"
        frame_status = "success"
        frame_message = "Data fetched."

        # First, fetch time series data for custom calculations (like Ichimoku)
        time_series_df = None
        try:
            # Fetch enough data for the longest indicator period (Ichimoku needs at least 52 periods)
            # Using a larger outputsize to ensure we have enough data for calculation
            time_series_df = twelvedata_fetcher.fetch_time_series(
                symbol=symbol,
                interval=ta_indicator_interval,
                outputsize=100  # Enough data for Ichimoku calculations
            )
            if time_series_df is None or time_series_df.empty:
                print(f"Warning: Failed to fetch time series data for {symbol} ({ta_indicator_interval})")
                frame_status = "partial_success"
                frame_message = f"Failed to fetch time series data for custom calculations."
        except Exception as e:
            print(f"Error fetching time series for {symbol} ({ta_indicator_interval}): {str(e)}")
            frame_status = "partial_success"
            frame_message = f"Error fetching time series: {str(e)}"

        # --- Define indicator tasks ---
        indicator_tasks = [
            # SMA 50
            (
                twelvedata_fetcher.fetch_sma,
                "moving_averages",
                "sma_50",
                lambda result: float(result[-1].get('sma')),
                symbol, ta_indicator_interval, 50
            ),
            # EMA 20
            (
                twelvedata_fetcher.fetch_ema,
                "moving_averages",
                "ema_20",
                lambda result: float(result[-1].get('ema')),
                symbol, ta_indicator_interval, 20
            ),
            # RSI 14
            (
                twelvedata_fetcher.fetch_rsi,
                "oscillators",
                "rsi_14",
                lambda result: float(result[-1].get('rsi')),
                symbol, ta_indicator_interval, 14
            ),
            # STOCH
            (
                twelvedata_fetcher.fetch_stoch,
                "oscillators",
                "stoch",
                lambda result: {
                    "slow_k": float(result[-1].get('slow_k', 0.0)),
                    "slow_d": float(result[-1].get('slow_d', 0.0))
                },
                symbol, ta_indicator_interval
            ),
            # MACD
            (
                twelvedata_fetcher.fetch_macd,
                "oscillators",
                "macd",
                lambda result: {
                    "line": float(result[-1].get('macd', result[-1].get('value', 0.0))),
                    "histogram": float(result[-1].get('hist', 0.0)),
                    "signal": float(result[-1].get('signal', 0.0))
                },
                symbol, ta_indicator_interval
            ),
            # ATR 14
            (
                twelvedata_fetcher.fetch_atr,
                "volatility",
                "atr_14",
                lambda result: float(result[-1].get('atr')),
                symbol, ta_indicator_interval, 14
            ),
            # VWAP
            (
                twelvedata_fetcher.fetch_vwap,
                "vwap",
                None,
                lambda result: {
                    "value": float(result[-1].get('vwap', 0.0)),
                    "upper_band_1sd": None,
                    "lower_band_1sd": None
                },
                symbol, ta_indicator_interval
            )
        ]

        # --- Execute all indicator tasks in parallel ---
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as indicator_executor:
            # Submit all tasks
            futures = [indicator_executor.submit(fetch_and_process_indicator, *task) for task in indicator_tasks]

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                status, message, data = future.result()
                results.append((status, message))

                # Update ta_data with the fetched result if available
                if data:
                    category, key, value = data
                    if key is None:  # Special case for VWAP
                        ta_data[category] = value
                    else:
                        ta_data[category][key] = value

        # Calculate Ichimoku Cloud if we have time series data
        if time_series_df is not None and not time_series_df.empty:
            try:
                ichimoku_data = technical_analysis.calculate_ichimoku(time_series_df)
                if ichimoku_data:
                    ta_data["ichimoku"] = ichimoku_data
                    results.append(("success", ""))
                else:
                    results.append(("warning", "Ichimoku calculation returned no data."))
            except Exception as e:
                print(f"Error calculating Ichimoku for {symbol} ({ta_indicator_interval}): {str(e)}")
                results.append(("partial_success", f"Ichimoku calculation error: {str(e)}"))

        # Determine frame status and message based on results
        error_messages = [msg for status, msg in results if msg]
        has_error = any(status == "error" for status, msg in results)
        has_partial = any(status == "partial_success" for status, msg in results)

        if has_error:
            frame_status = "error"
            frame_message = " ".join(error_messages)
        elif has_partial:
            frame_status = "partial_success"
            frame_message = "Some indicators failed. " + " ".join(error_messages)

        # Determine final status and message for this timeframe
        if not any(category for category in ta_data.values()):
            # If all TA categories are still empty after trying to populate them
            frame_status = "warning" # Not a fetch error, but no data available
            frame_message = "No Technical Analysis data available for this timeframe/interval."
            ta_data = {} # Ensure empty dict if no data
        elif frame_status == "success":
            # If at least one indicator was successfully fetched and parsed and no errors were flagged
            frame_message = "All indicators fetched and parsed successfully."

        result = {
            "calculation_context": {
                "ta_indicator_interval": ta_indicator_interval,
            },
            "technical_analysis": ta_data,
            "status": frame_status,
            "message": frame_message
        }

        return {"result": result, "status": frame_status}

    # Process all timeframes in parallel using concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(timeframes_config)) as executor:
        future_to_timeframe = {}

        # Submit a task for each timeframe
        for timeframe_key, tf_settings in timeframes_config.items():
            ta_indicator_interval = tf_settings["ta_indicator_interval"]

            # Create a task to process this timeframe
            future = executor.submit(
                process_timeframe,
                timeframe_key,
                ta_indicator_interval,
                symbol
            )
            future_to_timeframe[future] = timeframe_key

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_timeframe):
            timeframe_key = future_to_timeframe[future]
            frame_result = future.result()
            timeframe_technical_analysis[timeframe_key] = frame_result["result"]

            # Update overall status
            frame_status = frame_result["status"]
            if frame_status == "error":
                overall_status = "error" # If any frame has a critical error, the overall is error
            elif frame_status == "partial_success" and overall_status == "success":
                overall_status = "partial_success" # Downgrade overall if at least one is partial

    # --- Construct Final Response ---
    if overall_status == "error":
        overall_message = "Critical errors occurred fetching data for one or more timeframes. Check timeframe statuses."
    elif overall_status == "partial_success":
        overall_message = "Technical Analysis data generated for some timeframes or some indicators failed. Check individual statuses."
    elif overall_status == "success" and not timeframe_technical_analysis:
        overall_status = "warning" # Or "error" if you require data for at least one frame
        overall_message = "No Technical Analysis data processed for any timeframe."
    elif overall_status == "success":
        overall_message = "All timeframes processed successfully."

    response_data: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp_utc": response_time_utc.isoformat() + 'Z',
        "status": overall_status,
        "message": overall_message,
        "timeframe_technical_analysis": timeframe_technical_analysis
    }

    return response_data