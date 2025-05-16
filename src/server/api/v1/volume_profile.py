from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from src.server.config import VP_1M_BARS_INTERVAL, VP_1M_LOOKBACK_HOURS, VP_5M_BARS_INTERVAL, VP_5M_LOOKBACK_DAYS, \
    VP_1D_BARS_INTERVAL, VP_1D_LOOKBACK_MONTHS, TWELVE_DATA_OUTPUT_SIZE
from src.server.data import twelvedata_fetcher
from src.server.compute import technical_analysis

router = APIRouter()

@router.get("/volume_profile/{symbol}")
async def get_volume_profile(
        symbol: str
):
    """
    Retrieves calculated Volume Profile structure for default timeframes (1m, 5m, 1d).
    Enhanced with volume dynamics analysis.
    """
    # Define the default timeframes and their settings for Volume Profile
    timeframes_config = {
        "1m": {
            "vp_bars_interval": VP_1M_BARS_INTERVAL,
            "vp_lookback_td": timedelta(hours=VP_1M_LOOKBACK_HOURS),
        },
        "5m": {
            "vp_bars_interval": VP_5M_BARS_INTERVAL,
            "vp_lookback_td": timedelta(days=VP_5M_LOOKBACK_DAYS),
        },
        "1d": {
            "vp_bars_interval": VP_1D_BARS_INTERVAL,
            "vp_lookback_td": timedelta(days=VP_1D_LOOKBACK_MONTHS * 30.44), # Approx days for months
        }
    }

    response_time_utc = datetime.utcnow()
    # If it's Saturday (5) or Sunday (6), roll back to last Friday
    if response_time_utc.weekday() == 5:  # Saturday
        response_time_utc -= timedelta(days=1)
    elif response_time_utc.weekday() == 6:  # Sunday
        response_time_utc -= timedelta(days=2)
    timeframe_volume_profile: Dict[str, Any] = {}
    overall_status = "success"
    overall_message = "Volume Profile data processed."

    # Process each default timeframe
    for timeframe_key, tf_settings in timeframes_config.items():
        vp_bars_interval = tf_settings["vp_bars_interval"]
        vp_lookback_td = tf_settings["vp_lookback_td"]

        # Determine date range for fetching VP bars
        start_date_utc = response_time_utc - vp_lookback_td
        start_date_str = start_date_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = response_time_utc.strftime("%Y-%m-%d %H:%M:%S")

        # --- 1. Fetch Raw Bar Data for Volume Profile Calculation ---
        df_vp = twelvedata_fetcher.fetch_time_series(
            symbol=symbol,
            interval=vp_bars_interval,
            start_date=start_date_str,
            end_date=end_date_str,
            outputsize=TWELVE_DATA_OUTPUT_SIZE # Handle pagination for >5000 bars
        )

        volume_profile_structure = {} # Default empty
        volume_dynamics = {}

        if df_vp is None: # Critical fetch error
            print(f"Error fetching VP bars for {symbol} / {timeframe_key}. Calculation skipped.")
            volume_profile_structure["error"] = "Failed to fetch time series data."
            overall_status = "partial_success" if overall_status == "success" else overall_status
        elif df_vp.empty: # Twelve Data returned no data
            print(f"No VP bar data found for {symbol} / {timeframe_key}. Calculation skipped.")
            volume_profile_structure["message"] = "No time series data for this period."
        else:
            # --- 2. Perform Local Volume Profile Calculation ---
            vp_result = technical_analysis.calculate_volume_profile(df_vp, price_precision=2)
            if vp_result is None:
                volume_profile_structure["error"] = "Volume Profile calculation failed."
                print(f"Warning: Volume Profile calculation failed for {symbol} / {timeframe_key}.")
                overall_status = "partial_success" if overall_status == "success" else overall_status
            else:
                volume_profile_structure = vp_result

                try:
                    # Calculate volume change over time
                    recent_period = min(5, len(df_vp))
                    if recent_period > 1:
                        recent_volume = df_vp['volume'].tail(recent_period).sum()
                        previous_period = min(recent_period, len(df_vp) - recent_period)
                        if previous_period > 0:
                            previous_volume = df_vp['volume'].iloc[-(recent_period+previous_period):-recent_period].sum()
                            if previous_volume > 0:
                                volume_change_pct = (recent_volume / previous_volume - 1) * 100
                                volume_dynamics["volume_change_pct"] = float(volume_change_pct)
                                volume_dynamics["volume_trend"] = "increasing" if volume_change_pct > 0 else "decreasing"

                    # Calculate average volume
                    avg_volume = df_vp['volume'].mean()
                    if not pd.isna(avg_volume) and avg_volume > 0:
                        volume_dynamics["average_volume"] = float(avg_volume)

                        # Calculate latest volume relative to average
                        if len(df_vp) > 0:
                            latest_volume = df_vp['volume'].iloc[-1]
                            volume_ratio = latest_volume / avg_volume
                            volume_dynamics["latest_volume_ratio"] = float(volume_ratio)
                            volume_dynamics["latest_volume_description"] = "above_average" if volume_ratio > 1.2 else "below_average" if volume_ratio < 0.8 else "average"

                    # Calculate up/down volume
                    df_vp['price_change'] = df_vp['close'].diff()
                    up_volume = df_vp.loc[df_vp['price_change'] > 0, 'volume'].sum()
                    down_volume = df_vp.loc[df_vp['price_change'] < 0, 'volume'].sum()
                    total_volume = df_vp['volume'].sum()

                    if total_volume > 0:
                        up_volume_ratio = up_volume / total_volume
                        down_volume_ratio = down_volume / total_volume

                        volume_dynamics["up_volume_ratio"] = float(up_volume_ratio)
                        volume_dynamics["down_volume_ratio"] = float(down_volume_ratio)
                        volume_dynamics["volume_bias"] = "bullish" if up_volume_ratio > down_volume_ratio else "bearish"

                    # Detect large volume bars
                    large_volume_bars = technical_analysis.identify_large_volume_bars(df_vp)
                    if large_volume_bars:
                        volume_dynamics["large_volume_bars"] = large_volume_bars

                        # Count bullish vs bearish large volume bars
                        bullish_large_bars = [bar for bar in large_volume_bars if bar["type"] == "bullish"]
                        bearish_large_bars = [bar for bar in large_volume_bars if bar["type"] == "bearish"]

                        volume_dynamics["bullish_large_bars_count"] = len(bullish_large_bars)
                        volume_dynamics["bearish_large_bars_count"] = len(bearish_large_bars)

                    # Calculate volume momentum indicators if enough data
                    volume_momentum = technical_analysis.calculate_volume_momentum_indicators(df_vp)
                    if volume_momentum:
                        # Extract key values for volume dynamics summary
                        if "obv" in volume_momentum:
                            volume_dynamics["obv"] = volume_momentum["obv"]
                        if "cmf" in volume_momentum:
                            volume_dynamics["cmf"] = volume_momentum["cmf"]
                        if "buying_pressure" in volume_momentum and "is_bullish_volume" in volume_momentum["buying_pressure"]:
                            volume_dynamics["is_bullish_volume"] = volume_momentum["buying_pressure"]["is_bullish_volume"]

                    # Evaluate if volume supports continued upside
                    supports_upside_factors = []

                    # Check volume trend
                    if volume_dynamics.get("volume_trend") == "increasing":
                        supports_upside_factors.append("increasing_volume")

                    # Check volume bias
                    if volume_dynamics.get("volume_bias") == "bullish":
                        supports_upside_factors.append("bullish_volume_bias")

                    # Check latest volume
                    if volume_dynamics.get("latest_volume_description") == "above_average":
                        supports_upside_factors.append("above_average_volume")

                    # Check large volume bars
                    if volume_dynamics.get("bullish_large_bars_count", 0) > volume_dynamics.get("bearish_large_bars_count", 0):
                        supports_upside_factors.append("bullish_volume_spikes")

                    # Check volume indicators
                    if volume_dynamics.get("obv", 0) > 0:
                        supports_upside_factors.append("positive_obv")

                    if volume_dynamics.get("cmf", 0) > 0:
                        supports_upside_factors.append("positive_cmf")

                    if volume_dynamics.get("is_bullish_volume", False):
                        supports_upside_factors.append("buying_pressure")

                    # Calculate overall probability
                    upside_factor_count = len(supports_upside_factors)
                    max_possible_factors = 7  # Total possible factors we check
                    upside_probability = min(upside_factor_count / max_possible_factors * 100, 100)

                    # Add summary to volume dynamics
                    volume_dynamics["supports_upside_factors"] = supports_upside_factors
                    volume_dynamics["volume_supports_upside_probability"] = upside_probability
                    volume_dynamics["volume_indicates_continued_upside"] = upside_probability > 50

                except Exception as e:
                    print(f"Error calculating volume dynamics: {str(e)}")
                    volume_dynamics["error"] = f"Volume dynamics calculation error: {str(e)}"


        # --- 3. Store Results for this Timeframe ---
        timeframe_volume_profile[timeframe_key] = {
            "calculation_context": {
                "vp_bars_interval": vp_bars_interval,
                "vp_lookback_description": f"Last ~{int(vp_lookback_td.total_seconds() / 3600)} hours" if vp_bars_interval == "1min" else f"Last ~{vp_lookback_td.days} days",
            },
            "volume_profile_structure": volume_profile_structure,
            "volume_dynamics": volume_dynamics
        }

        # If calculation failed completely for this timeframe, note it
        if "error" in volume_profile_structure or "message" in volume_profile_structure and not volume_profile_structure.get("point_of_control"):
            timeframe_volume_profile[timeframe_key]["status"] = "warning"
            timeframe_volume_profile[timeframe_key]["message"] = volume_profile_structure.get("error", volume_profile_structure.get("message", "No data or calculation failed."))
            if "error" in volume_profile_structure:
                del volume_profile_structure["error"] # Clean up the error detail in the VP structure itself


    # --- 4. Construct Final Response ---
    if overall_status == "partial_success":
        overall_message = "Volume Profile data generated for some timeframes, check individual statuses."
    elif overall_status == "success" and not timeframe_volume_profile:
        overall_status = "error"
        overall_message = "No Volume Profile data processed for any timeframe."

    # Add consolidated analysis across all timeframes
    supporting_timeframes = []
    for timeframe, data in timeframe_volume_profile.items():
        if "upside_analysis" in data and data["upside_analysis"].get("supports_continued_upside", False):
            supporting_timeframes.append(timeframe)

    consolidated_analysis = {
        "supports_continued_upside": len(supporting_timeframes) > 0,
        "supporting_timeframes": supporting_timeframes,
        "confidence": "high" if len(supporting_timeframes) >= 2 else "medium" if len(supporting_timeframes) == 1 else "low"
    }

    response_data: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp_utc": response_time_utc.isoformat() + 'Z',
        "status": overall_status,
        "message": overall_message,
        "timeframe_volume_profile": timeframe_volume_profile,
        "consolidated_analysis": consolidated_analysis
    }

    return response_data
