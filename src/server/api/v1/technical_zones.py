from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from server.config import VP_1M_BARS_INTERVAL, VP_1M_LOOKBACK_HOURS, TA_1M_INDICATOR_INTERVAL, VP_5M_BARS_INTERVAL, \
    VP_5M_LOOKBACK_DAYS, TA_5M_INDICATOR_INTERVAL, VP_1D_BARS_INTERVAL, VP_1D_LOOKBACK_MONTHS, TA_1D_INDICATOR_INTERVAL, \
    TWELVE_DATA_OUTPUT_SIZE, PRICE_BANDING_WIDTH
from src.server.data import twelvedata_fetcher
from src.server.compute import technical_analysis

router = APIRouter()

@router.get("/technical_zones/{symbol}")
def get_technical_zones(
        symbol: str
):
    """
    Retrieves technical support and resistance zones derived from Volume Profile
    and other technical methods for default granular timeframes (1m, 5m).
    Omits confidence scores and provides zones for each timeframe bucket.
    """
    # Define the default timeframes and their specific data acquisition/calculation settings
    # using values from settings.py
    timeframes_config = {
        "1m": {
            "vp_bars_interval": VP_1M_BARS_INTERVAL, # e.g., '1min'
            "vp_lookback_hours": VP_1M_LOOKBACK_HOURS, # e.g., 10 hours
            "ta_indicator_interval":TA_1M_INDICATOR_INTERVAL # e.g., '5min' for ATR/etc.
        },
        "5m": {
            "vp_bars_interval": VP_5M_BARS_INTERVAL, # e.g., '5min'
            "vp_lookback_days": VP_5M_LOOKBACK_DAYS, # e.g., 5 days
            "ta_indicator_interval": TA_5M_INDICATOR_INTERVAL # e.g., '15min' for ATR/etc.
        },

        "1d": {
            "vp_bars_interval": VP_1D_BARS_INTERVAL, # e.g., '1h' or '1day'
            "vp_lookback_months": VP_1D_LOOKBACK_MONTHS, # e.g., 6 months
            "ta_indicator_interval": TA_1D_INDICATOR_INTERVAL # e.g., '1day' for ATR/PDH/PDL
        }
    }

    response_time_utc = datetime.utcnow()
    # Adjust timestamp to a trading day if on a weekend, for consistent date range calculation
    # This assumes the lookback should end on the most recent trading day's close/current time.
    adjusted_timestamp_utc = response_time_utc
    if adjusted_timestamp_utc.weekday() == 5: # Saturday
        adjusted_timestamp_utc -= timedelta(days=1)
    elif adjusted_timestamp_utc.weekday() == 6: # Sunday
        adjusted_timestamp_utc -= timedelta(days=2)

    timeframe_zones_data: Dict[str, Any] = {} # Dictionary to hold results for each timeframe
    overall_status = "success"
    overall_message = "Technical zones processed."


    # Process each configured timeframe
    for timeframe_key, tf_settings in timeframes_config.items():
        vp_bars_interval = tf_settings.get("vp_bars_interval") # Use .get for safety
        ta_indicator_interval = tf_settings.get("ta_indicator_interval")

        # Determine lookback duration based on timeframe config
        if "vp_lookback_hours" in tf_settings:
            vp_lookback_td = timedelta(hours=tf_settings["vp_lookback_hours"])
            lookback_description = f"Last ~{tf_settings['vp_lookback_hours']} hours"
        elif "vp_lookback_days" in tf_settings:
            vp_lookback_td = timedelta(days=tf_settings["vp_lookback_days"])
            lookback_description = f"Last ~{tf_settings['vp_lookback_days']} trading days"
        elif "vp_lookback_months" in tf_settings:
            # Approximate days for months
            vp_lookback_td = timedelta(days=tf_settings["vp_lookback_months"] * 30.44)
            lookback_description = f"Last ~{tf_settings['vp_lookback_months']} months"
        else:
            print(f"Error: No valid lookback defined for timeframe {timeframe_key}. Skipping.")
            overall_status = "partial_success" if overall_status == "success" else overall_status
            continue # Skip this timeframe

        # Determine date range for fetching VP bars using the adjusted timestamp
        start_date_utc = adjusted_timestamp_utc - vp_lookback_td
        start_date_str = start_date_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = adjusted_timestamp_utc.strftime("%Y-%m-%d %H:%M:%S")


        # List to hold zones identified for this specific timeframe
        technical_zones_list: List[Dict[str, Any]] = []
        frame_status = "success"
        frame_message = "Zones generated."
        frame_fetch_failed = False # Flag for critical fetch errors in this frame

        # --- 1. Fetch Raw Bar Data for Volume Profile Calculation ---
        df_vp = twelvedata_fetcher.fetch_time_series(
            symbol=symbol,
            interval=vp_bars_interval,
            start_date=start_date_str,
            end_date=end_date_str,
            outputsize=TWELVE_DATA_OUTPUT_SIZE # Handle pagination if needed
        )

        volume_profile_structure = None # Default to None
        if df_vp is None: # Critical fetch error
            print(f"Error fetching VP bars for {symbol} / {timeframe_key}. VP calculation skipped.")
            frame_fetch_failed = True
        elif df_vp.empty: # Twelve Data returned no data for the period
            print(f"No VP bar data found for {symbol} / {timeframe_key}. VP calculation skipped.")
            # This is a warning, not error, for this frame
            frame_status = "warning" if frame_status == "success" else frame_status
            frame_message = "No VP data available." # Update message
        else:
            # --- 2. Perform Local Volume Profile Calculation and Identify Zones ---
            vp_result = technical_analysis.calculate_volume_profile(df_vp, price_precision=2) # Assume 2 decimals

            if vp_result is None:
                print(f"Warning: Volume Profile calculation failed for {symbol} / {timeframe_key}. No VP zones.")
                frame_status = "warning" if frame_status == "success" else frame_status
                frame_message = "VP calculation failed."
            else:
                volume_profile_structure = vp_result # Store the calculated VP structure

                # Add VP zones if calculation was successful
                if volume_profile_structure.get("point_of_control") is not None: # Check if POC was successfully calculated
                    technical_zones_list.append({
                        "type": "NEUTRAL",
                        "name": f"{timeframe_key.upper()} POC", # Use timeframe key in name
                        "level": volume_profile_structure["point_of_control"],
                        "source": f"Volume Profile ({vp_bars_interval} Bars)"
                    })
                if volume_profile_structure.get("value_area_high") is not None:
                    technical_zones_list.append({
                        "type": "RESISTANCE",
                        "name": f"{timeframe_key.upper()} VAH",
                        "range_start": volume_profile_structure["value_area_high"],
                        "range_end": volume_profile_structure["value_area_high"] + (PRICE_BANDING_WIDTH / 2), # Approx range end
                        "source": f"Volume Profile ({vp_bars_interval} Bars)"
                    })
                if volume_profile_structure.get("value_area_low") is not None:
                    technical_zones_list.append({
                        "type": "SUPPORT",
                        "name": f"{timeframe_key.upper()} VAL",
                        "range_start": volume_profile_structure["value_area_low"] - (PRICE_BANDING_WIDTH / 2), # Approx range start
                        "range_end": volume_profile_structure["value_area_low"],
                        "source": f"Volume Profile ({vp_bars_interval} Bars)"
                    })
                # Add HVNs/LVNs if your calculate_volume_profile returns them
                for hvn in volume_profile_structure.get("high_volume_nodes", []):
                    technical_zones_list.append({
                        "type": "RESISTANCE" if hvn["start"] > volume_profile_structure.get("point_of_control", -1) else "SUPPORT", # Simple type logic
                        "name": f"{timeframe_key.upper()} HVN {hvn['start']}",
                        "range_start": hvn["start"],
                        "range_end": hvn["end"],
                        "source": f"Volume Profile ({vp_bars_interval} Bars)"
                    })
                for lvn in volume_profile_structure.get("low_volume_nodes", []):
                    technical_zones_list.append({
                        "type": "NEUTRAL", # LVNs are often acceleration points, not S/R
                        "name": f"{timeframe_key.upper()} LVN {lvn['start']}",
                        "range_start": lvn["start"],
                        "range_end": lvn["end"],
                        "source": f"Volume Profile ({vp_bars_interval} Bars)"
                    })


        # --- 3. Fetch Standard TA Indicators needed for other zones (like ATR) ---
        # Fetch ATR using the specified TA interval for this timeframe
        atr_data = twelvedata_fetcher.fetch_atr(symbol, ta_indicator_interval, 14) # Assuming 14-period ATR

        atr_value = None
        if atr_data is None:
            print(f"Error fetching ATR for {symbol}/{timeframe_key}.")
            frame_fetch_failed = True
            # ATR fetch failed, skip ATR based zones for this frame
        elif not atr_data:
            print(f"No ATR data points returned for {symbol}/{timeframe_key}. Skipping ATR zones.")
            # No data is not a critical error, but means we can't make ATR zones
        else: # ATR data was fetched and not empty
            try:
                # Get the latest ATR value
                atr_value = float(atr_data[-1].get('atr')) # Use .get('atr') and float conversion
                if atr_value is None: # Handle if 'atr' key was missing or None despite list not being empty
                    print(f"Warning: Last ATR data item for {symbol}/{timeframe_key} missing or invalid 'atr' value. Skipping ATR zones.")
                    atr_value = None # Ensure it's None if parsing failed
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Parse error ATR for {symbol}/{timeframe_key}: {e}. Skipping ATR zones.")
                frame_status = "partial_success" if frame_status == "success" else frame_status
                frame_message = "Partial data fetched (ATR error)."
                atr_value = None # Ensure it's None if parsing failed


        # --- 4. Identify other zones (e.g., ATR extensions, Previous Day High/Low) ---
        current_price = None # You might need current price for some zone calculations
        # fetch underlying price if needed (e.g., for ATR extensions relative to current price)
        # current_price_data = twelvedata_fetcher.fetch_underlying_price(symbol) # Example fetcher call
        # if current_price_data: current_price = float(current_price_data.get('price'))

        # Example: ATR Extensions (using a fixed base like last close from VP bars or current price)
        if atr_value is not None and df_vp is not None and not df_vp.empty:
            # Use the last close price from VP bars as a base for ATR extensions
            base_price = float(df_vp['close'].iloc[-1])
            if base_price is not None:
                # Add 1x ATR extensions as potential targets/boundaries
                technical_zones_list.append({
                    "type": "TARGET_UPSIDE",
                    "name": f"{timeframe_key.upper()} +1 ATR",
                    "level": round(base_price + atr_value, 2), # Round to 2 decimals
                    "source": f"ATR Calculation ({ta_indicator_interval})"
                })
                technical_zones_list.append({
                    "type": "TARGET_DOWNSIDE",
                    "name": f"{timeframe_key.upper()} -1 ATR",
                    "level": round(base_price - atr_value, 2),
                    "source": f"ATR Calculation ({ta_indicator_interval})"
                })


        # Example: Previous Day High/Low (PDH/PDL) - Requires daily bar data
        # This zone type is only relevant for timeframes >= day (like 5m or 1m looking back across days)
        # We need to fetch the *previous* day's bar data specifically.
        if timeframe_key in ["1m", "5m"]: # These timeframes benefit from daily levels
            # Fetch the previous day's bar
            prev_day_start = adjusted_timestamp_utc - timedelta(days=1)
            # Need to find the *actual* previous trading day if weekend/holiday logic is complex
            # For simplicity, fetch 2 days prior and get the last daily bar
            prev_day_df = twelvedata_fetcher.fetch_time_series(
                symbol=symbol,
                interval="1day",
                start_date=(adjusted_timestamp_utc - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                end_date=adjusted_timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"),
                outputsize=2 # Just need the last couple of daily bars
            )
            if prev_day_df is not None and not prev_day_df.empty and len(prev_day_df) > 1:
                # The second to last bar is likely the previous trading day
                prev_day_bar = prev_day_df.iloc[-2] # -1 is today, -2 is yesterday
                pdh = float(prev_day_bar.get('high'))
                pdl = float(prev_day_bar.get('low'))
                if pdh is not None:
                    technical_zones_list.append({
                        "type": "RESISTANCE",
                        "name": "Previous Day High",
                        "level": round(pdh, 2),
                        "source": "Price Action (Daily Bar)"
                    })
                if pdl is not None:
                    technical_zones_list.append({
                        "type": "SUPPORT",
                        "name": "Previous Day Low",
                        "level": round(pdl, 2),
                        "source": "Price Action (Daily Bar)"
                    })
            elif prev_day_df is None:
                print(f"Error fetching previous day data for {symbol}/{timeframe_key}. Skipping PDH/PDL.")
                frame_fetch_failed = True
            else:
                print(f"No previous day data found for {symbol}/{timeframe_key}. Skipping PDH/PDL.")
                # Not a critical error, just means no data available

        # --- Store Results for this Timeframe ---
        # If any critical fetch error occurred for this frame, mark its status as error
        if frame_fetch_failed:
            final_frame_status = "error"
            final_frame_message = "Critical fetch error for one or more data sources."
            # Clear zones list if critical error? Or return partial? Let's return partial data.
            # technical_zones_list = [] # Uncomment to clear zones on critical error
        elif not technical_zones_list:
            # If no zones were generated at all for this timeframe
            final_frame_status = "warning"
            final_frame_message = "No technical zones could be generated for this timeframe."
        else:
            # Some zones were generated, check if there were partial errors (e.g., parse errors)
            final_frame_status = frame_status # 'success' or 'partial_success'
            if frame_status == "partial_success":
                final_frame_message = f"Some zones generated, but errors occurred ({frame_message})"
            else:
                final_frame_message = "Zones generated successfully."


        timeframe_zones_data[timeframe_key] = {
            "calculation_context": {
                "bars_interval": vp_bars_interval,
                "lookback_description": lookback_description,
                "ta_indicator_interval": ta_indicator_interval,
                # Add data freshness if possible
                # "data_freshness_utc": "..."
            },
            "technical_zones": technical_zones_list, # Contains the zones found for this timeframe
            "status": final_frame_status,
            "message": final_frame_message
        }

        # Update overall status
        if final_frame_status == "error":
            overall_status = "error" # If any frame has a critical error, the overall is error
        elif final_frame_status in ["partial_success", "warning"] and overall_status == "success":
            overall_status = "partial_success" # Downgrade overall if at least one is partial or warning


    # --- 5. Construct Final Response ---
    if overall_status == "error":
        overall_message = "Critical errors occurred fetching data for one or more timeframes. Check timeframe statuses."
        # Consider returning a 500 HTTPException here instead of 200 with error status
        # raise HTTPException(status_code=500, detail=overall_message)
    elif overall_status == "partial_success":
        overall_message = "Technical zones generated for some timeframes or some data failed. Check individual statuses."
    elif overall_status == "success" and not timeframe_zones_data:
        # Should not happen if timeframes_config is not empty, but safety check
        overall_status = "error" # Or warning
        overall_message = "No technical zones processed for any timeframe."
    elif overall_status == "success":
        overall_message = "All timeframes processed successfully."


    response_data: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp_utc": response_time_utc.isoformat() + 'Z',
        "status": overall_status,
        "message": overall_message,
        "timeframe_zones": timeframe_zones_data
    }

    return response_data