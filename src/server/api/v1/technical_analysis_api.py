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
            "ichimoku": {},
            "volume_indicators": {},
            "trend_strength": {},
            "divergences": {}
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
            else:
                try:
                    volume_indicators = technical_analysis.calculate_volume_momentum_indicators(time_series_df)
                    if volume_indicators:
                        ta_data["volume_indicators"] = volume_indicators
                except Exception as e:
                    print(f"Warning: Failed to calculate volume indicators for {symbol} ({ta_indicator_interval}): {str(e)}")

                try:
                    trend_strength = technical_analysis.calculate_trend_strength(time_series_df)
                    if trend_strength:
                        ta_data["trend_strength"] = trend_strength
                except Exception as e:
                    print(f"Warning: Failed to calculate trend strength for {symbol} ({ta_indicator_interval}): {str(e)}")

                try:
                    divergences = technical_analysis.detect_divergences(time_series_df)
                    if divergences:
                        ta_data["divergences"] = divergences
                except Exception as e:
                    print(f"Warning: Failed to detect divergences for {symbol} ({ta_indicator_interval}): {str(e)}")

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
            ),
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

        try:
            # Compile factors that suggest continued upside despite overbought readings
            uptrend_continuation_factors = []

            # Check volume indicators (if available)
            if "volume_indicators" in ta_data and ta_data["volume_indicators"]:
                vol_indicators = ta_data["volume_indicators"]

                # Check if OBV is positive (bullish)
                if "obv" in vol_indicators and vol_indicators["obv"] > 0:
                    uptrend_continuation_factors.append("positive_obv")

                # Check if CMF is positive (bullish)
                if "cmf" in vol_indicators and vol_indicators["cmf"] > 0:
                    uptrend_continuation_factors.append("positive_cmf")

                # Check buying pressure
                if "buying_pressure" in vol_indicators and vol_indicators["buying_pressure"].get("is_bullish_volume", False):
                    uptrend_continuation_factors.append("bullish_volume_pattern")

                # Check large volume bars
                if "large_volume_bars" in vol_indicators:
                    bullish_bars = [bar for bar in vol_indicators["large_volume_bars"] if bar["type"] == "bullish"]
                    if bullish_bars:
                        uptrend_continuation_factors.append("bullish_volume_spikes")

            # Check trend strength
            if "trend_strength" in ta_data and ta_data["trend_strength"]:
                ts = ta_data["trend_strength"]

                # Strong ADX suggests trend likely to continue
                if "adx" in ts and ts["adx"] > 25:
                    uptrend_continuation_factors.append("strong_adx")

                # Positive EMA slope suggests uptrend
                if "ema20_slope" in ts and ts["ema20_slope"] > 0:
                    uptrend_continuation_factors.append("rising_ema")

                # Price well above moving average suggests strong trend
                if "price_distance_from_sma50" in ts and ts["price_distance_from_sma50"] > 3:
                    uptrend_continuation_factors.append("price_above_ma")

            # Check divergences
            if "divergences" in ta_data and ta_data["divergences"]:
                if ta_data["divergences"].get("bullish_divergence", False):
                    uptrend_continuation_factors.append("bullish_divergence")

            # Check ichimoku (if available)
            if "ichimoku" in ta_data and ta_data["ichimoku"]:
                if ta_data["ichimoku"].get("cloud_status") == "bullish":
                    uptrend_continuation_factors.append("bullish_ichimoku")

            # Overall MACD condition
            if "oscillators" in ta_data and "macd" in ta_data["oscillators"]:
                macd = ta_data["oscillators"]["macd"]
                if macd.get("line", 0) > 0 and macd.get("histogram", 0) > 0:
                    uptrend_continuation_factors.append("positive_macd")

            # Calculate uptrend probability based on factors
            uptrend_factor_count = len(uptrend_continuation_factors)
            max_possible_factors = 9  # Total possible factors we check
            uptrend_probability = min(uptrend_factor_count / max_possible_factors * 100, 100)

            # Add to response
            ta_data["continuation_analysis"] = {
                "uptrend_continuation_probability": uptrend_probability,
                "uptrend_continuation_factors": uptrend_continuation_factors,
                "overbought_may_continue": uptrend_probability > 60
            }

        except Exception as e:
            print(f"Error calculating uptrend continuation probability: {str(e)}")

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

    supporting_timeframes = []
    for timeframe_key, timeframe_data in timeframe_technical_analysis.items():
        if "technical_analysis" in timeframe_data and "continuation_analysis" in timeframe_data["technical_analysis"]:
            continuation_analysis = timeframe_data["technical_analysis"]["continuation_analysis"]
            if continuation_analysis.get("overbought_may_continue", False):
                supporting_timeframes.append(timeframe_key)

    # Calculate confidence level
    confidence_level = "high" if len(supporting_timeframes) >= 2 else "medium" if len(supporting_timeframes) == 1 else "low"

    # Identify shared factors across timeframes
    shared_factors = []
    if supporting_timeframes:
        # Get factors from first supporting timeframe as baseline
        baseline_timeframe = supporting_timeframes[0]
        baseline_factors = timeframe_technical_analysis[baseline_timeframe]["technical_analysis"]["continuation_analysis"].get("uptrend_continuation_factors", [])

        # Check if factors appear in all supporting timeframes
        for factor in baseline_factors:
            factor_in_all = True
            for tf in supporting_timeframes[1:]:
                tf_factors = timeframe_technical_analysis[tf]["technical_analysis"]["continuation_analysis"].get("uptrend_continuation_factors", [])
                if factor not in tf_factors:
                    factor_in_all = False
                    break

            if factor_in_all and factor not in shared_factors:
                shared_factors.append(factor)

    consolidated_analysis = {
        "supports_continued_upside": len(supporting_timeframes) > 0,
        "supporting_timeframes": supporting_timeframes,
        "confidence": confidence_level,
        "common_factors": shared_factors
    }

    response_data: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp_utc": response_time_utc.isoformat() + 'Z',
        "status": overall_status,
        "message": overall_message,
        "timeframe_technical_analysis": timeframe_technical_analysis,
        "consolidated_analysis": consolidated_analysis
    }

    return response_data