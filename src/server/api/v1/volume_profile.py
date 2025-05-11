from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta

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


        # --- 3. Store Results for this Timeframe ---
        timeframe_volume_profile[timeframe_key] = {
            "calculation_context": {
                "vp_bars_interval": vp_bars_interval,
                "vp_lookback_description": f"Last ~{int(vp_lookback_td.total_seconds() / 3600)} hours" if vp_bars_interval == "1min" else f"Last ~{vp_lookback_td.days} days",
            },
            "volume_profile_structure": volume_profile_structure
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


    response_data: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp_utc": response_time_utc.isoformat() + 'Z',
        "status": overall_status,
        "message": overall_message,
        "timeframe_volume_profile": timeframe_volume_profile
    }

    return response_data