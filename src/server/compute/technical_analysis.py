import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Note: This file contains only calculations that are NOT available directly
# from the Twelve Data API. Volume Profile is the primary example here.
# Standard TA (SMA, RSI, etc.) is fetched via twelvedata_fetcher.py


def calculate_volume_profile(
        df: pd.DataFrame,
        price_precision: int = 2 # Decimal places for price banding
) -> Optional[Dict[str, Any]]:
    """
    Calculates key Volume Profile levels (POC, VAH, VAL) from historical
    price/volume DataFrame fetched from Twelve Data's time series endpoint.
    Assumes DataFrame has 'high', 'low', 'close', and 'volume' columns
    as provided by Twelve Data. Returns standard Python types for JSON serialization.
    """
    # Ensure required columns exist and DataFrame is not empty after cleaning
    if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        print("Volume Profile calculation skipped: Input DataFrame is empty or lacks required columns.")
        return None
    # Filter out rows with any NaN in critical columns (price/volume) before calculation
    df_cleaned = df.dropna(subset=['high', 'low', 'close', 'volume']).copy()
    if df_cleaned.empty:
        print("Volume Profile calculation skipped: Input DataFrame is empty after cleaning NaN values.")
        return None


    # Determine price range, handling potential NaNs after cleaning
    min_price = df_cleaned[['low', 'close']].min().min()
    max_price = df_cleaned[['high', 'close']].max().max()

    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        # Handle cases with no valid price data range (e.g., all prices are the same, or NaNs dominated)
        print(f"Volume Profile calculation skipped: Invalid price range ({min_price}-{max_price}).")
        return None

    # Determine bin width based on price precision (e.g., 0.01 for precision 2)
    bin_width = 10 ** -price_precision
    # Create bins that span the entire price range, ensuring max price is included
    # Add a small epsilon to the max price to ensure the highest price falls into a bin
    bins = np.arange(min_price, max_price + bin_width * 1.001, bin_width)

    # Assign each bar's volume to a price bin. Using typical price is common.
    df_cleaned['typical_price'] = (df_cleaned['high'] + df_cleaned['low'] + df_cleaned['close']) / 3
    # Use pd.cut for binning, which is robust. Labels are the left edge of the bin.
    df_cleaned['price_bin'] = pd.cut(df_cleaned['typical_price'], bins=bins, labels=bins[:-1], include_lowest=True, right=True)

    # Sum volume per bin
    volume_by_price_bin = df_cleaned.groupby('price_bin', observed=True)['volume'].sum().reset_index()
    volume_by_price_bin = volume_by_price_bin.rename(columns={'price_bin': 'price', 'volume': 'total_volume'})

    # pd.cut labels are interval objects or float intervals, convert to float for sorting and comparison
    if not volume_by_price_bin.empty and isinstance(volume_by_price_bin['price'].iloc[0], pd.Interval):
        volume_by_price_bin['price'] = volume_by_price_bin['price'].apply(lambda x: x.left if pd.notna(x) else np.nan).round(price_precision)
    else: # Should be float if labels=bins[:-1]
        volume_by_price_bin['price'] = volume_by_price_bin['price'].astype(float).round(price_precision)


    # Filter out bins with zero volume and sort by price ascending
    volume_by_price_bin = volume_by_price_bin[volume_by_price_bin['total_volume'] > 0].dropna(subset=['price']).sort_values('price').reset_index(drop=True)

    if volume_by_price_bin.empty:
        print("Volume Profile calculation skipped: No volume traded in any bin after cleaning and binning.")
        return None

    # --- Calculate POC ---
    total_volume_sum = volume_by_price_bin['total_volume'].sum()
    # Explicitly convert total_volume_sum to standard Python int/float
    total_volume_sum_py = int(total_volume_sum) if pd.api.types.is_integer_dtype(volume_by_price_bin['total_volume']) else float(total_volume_sum)

    if total_volume_sum_py == 0:
        poc_level = round((min_price + max_price) / 2, price_precision) # Default POC if no volume
        print("Volume Profile calculation: Total volume sum is zero.")
        return {"point_of_control": float(poc_level), "value_area_high": float(poc_level), "value_area_low": float(poc_level), "total_volume": 0} # Ensure float conversion for levels


    # Find the index of the row with the maximum volume
    poc_row_index = volume_by_price_bin['total_volume'].idxmax()
    poc_level = round(volume_by_price_bin.loc[poc_row_index, 'price'], price_precision)


    # --- Calculate VAH/VAL (Value Area High/Low) ---
    # Value Area typically covers 70% of total volume
    value_area_volume_target = total_volume_sum * 0.70

    # Start from POC's price bin index in the *price-sorted* dataframe
    current_volume_in_va = volume_by_price_bin.loc[poc_row_index, 'total_volume']
    value_area_price_indices = {poc_row_index} # Use a set to track indices within VA
    up_idx = poc_row_index + 1
    down_idx = poc_row_index - 1

    # Expand outwards from POC price bin until 70% of volume is covered
    while current_volume_in_va < value_area_volume_target and (up_idx < len(volume_by_price_bin) or down_idx >= 0):
        can_go_up = up_idx < len(volume_by_price_bin)
        can_go_down = down_idx >= 0

        vol_up = volume_by_price_bin.loc[up_idx, 'total_volume'] if can_go_up else -1
        vol_down = volume_by_price_bin.loc[down_idx, 'total_volume'] if can_go_down else -1

        if vol_up > vol_down:
            if current_volume_in_va + vol_up <= total_volume_sum * 1.01:
                current_volume_in_va += vol_up
                value_area_price_indices.add(up_idx)
                up_idx += 1
            elif can_go_down and current_volume_in_va + vol_down <= total_volume_sum * 1.01:
                current_volume_in_va += vol_down
                value_area_price_indices.add(down_idx)
                down_idx -= 1
            else:
                break
        elif vol_down > -1:
            if current_volume_in_va + vol_down <= total_volume_sum * 1.01:
                current_volume_in_va += vol_down
                value_area_price_indices.add(down_idx)
                down_idx -= 1
            elif can_go_up and current_volume_in_va + vol_up <= total_volume_sum * 1.01:
                current_volume_in_va += vol_up
                value_area_price_indices.add(up_idx)
                up_idx += 1
            else:
                break
        else:
            break

    # Get the price levels corresponding to the indices in the VA set
    value_area_price_levels = [volume_by_price_bin.loc[idx, 'price'] for idx in value_area_price_indices]

    # Ensure min/max calculation is safe and convert to float
    vah_level = round(max(value_area_price_levels), price_precision) if value_area_price_levels else poc_level
    val_level = round(min(value_area_price_levels), price_precision) if value_area_price_levels else poc_level


    # --- Identify HVNs/LVNs (Simple Placeholder for now) ---
    # TODO: this is for future and is not needed now
    high_volume_nodes = []
    low_volume_nodes = []

    # --- Return standard Python types ---
    return {
        "point_of_control": float(poc_level), # Ensure float conversion
        "value_area_high": float(vah_level), # Ensure float conversion
        "value_area_low": float(val_level),  # Ensure float conversion
        "total_volume": total_volume_sum_py, # Use the standard Python type
        # "high_volume_nodes": high_volume_nodes, # Already list of dicts
        # "low_volume_nodes": low_volume_nodes   # Already list of dicts
    }


def calculate_fibonacci_levels(
        df: pd.DataFrame,
        price_precision: int = 2
) -> List[Dict[str, Any]]:
    """
    Calculates standard Fibonacci retracement and extension levels
    based on the highest high and lowest low within the provided DataFrame.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        price_precision: Decimal places for rounding prices.

    Returns:
        List of dictionaries representing the Fibonacci zones.
    """
    fib_zones: List[Dict[str, Any]] = []

    if df is None or df.empty or not all(col in df.columns for col in ['high', 'low']):
        print("Fibonacci calculation skipped: Input DataFrame is empty or lacks required columns (high, low).")
        return fib_zones # Return empty list

    # Find the highest high and lowest low within the DataFrame
    highest_high = df['high'].max()
    lowest_low = df['low'].min()

    # Ensure valid prices and a price swing exists
    if pd.isna(highest_high) or pd.isna(lowest_low) or highest_high <= lowest_low:
        print(f"Fibonacci calculation skipped: Invalid price range ({lowest_low}-{highest_high}).")
        return fib_zones # Return empty list

    price_range = highest_high - lowest_low

    # Define standard Fibonacci ratios
    # Retracements: 0.382, 0.5, 0.618
    # Extensions (Common): 1.0, 1.618
    retracement_ratios = [0.618, 0.5, 0.382] # Order from higher to lower
    extension_ratios = [1.0, 1.618] # Order from 100% upwards

    # Determine swing direction for retracements
    # If the last close is higher than the first open, assume an overall uptrend correction (draw Fibs from Low to High)
    # If the last close is lower than the first open, assume an overall downtrend correction (draw Fibs from High to Low)
    # This is a simple heuristic, more advanced methods use specific swing points.
    # Let's use the direction of the overall move in the DF
    first_price = df['open'].iloc[0]
    last_price = df['close'].iloc[-1]
    is_uptrend_correction = last_price > first_price

    # Calculate Retracement Levels
    for ratio in retracement_ratios:
        if is_uptrend_correction:
            # Price moved up, correcting down. Fibs from Low to High. Levels are Low + (Range * Ratio)
            level = lowest_low + (price_range * ratio)
            fib_zones.append({
                "type": "SUPPORT", # Retracements in uptrend are support
                "name": f"Fib Retracement {ratio*100:.1f}%",
                "level": round(level, price_precision),
                "source": "Fibonacci (Calculated)"
            })
        else:
            # Price moved down, correcting up. Fibs from High to Low. Levels are High - (Range * Ratio)
            level = highest_high - (price_range * ratio)
            fib_zones.append({
                "type": "RESISTANCE", # Retracements in downtrend are resistance
                "name": f"Fib Retracement {ratio*100:.1f}%",
                "level": round(level, price_precision),
                "source": "Fibonacci (Calculated)"
            })

    # Sort retracements by price for cleaner output (highest to lowest for RESISTANCE, lowest to highest for SUPPORT)
    fib_zones.sort(key=lambda x: x['level'], reverse=True if not is_uptrend_correction else False)


    # Calculate Extension Levels (typically targets beyond the initial move)
    # Simplest way: project from the High/Low of the move.
    # A common way to use extensions is from the start of the move, to the end, and then the retracement point.
    # Let's calculate from the endpoints of the move (High/Low) relative to the starting point.

    # Using the High-Low swing endpoints for extensions
    # Extension levels = High + (Range * Ratio) for uptrend continuation (Low->High move followed by potential break higher)
    # Extension levels = Low - (Range * Ratio) for downtrend continuation (High->Low move followed by potential break lower)

    # For simplicity, let's calculate extensions as projections from the 'end' of the swing
    # based on the magnitude of the swing.
    # If move was Low -> High: Levels = High + (Range * Ratio)
    # If move was High -> Low: Levels = Low - (Range * Ratio)

    swing_magnitude = abs(highest_high - lowest_low)

    for ratio in extension_ratios:
        if is_uptrend_correction: # Low -> High move
            level = highest_high + (swing_magnitude * ratio)
            fib_zones.append({
                "type": "TARGET_UPSIDE",
                "name": f"Fib Extension {ratio:.3f}",
                "level": round(level, price_precision),
                "source": "Fibonacci (Calculated)"
            })
        else: # High -> Low move
            level = lowest_low - (swing_magnitude * ratio)
            fib_zones.append({
                "type": "TARGET_DOWNSIDE",
                "name": f"Fib Extension {ratio:.3f}",
                "level": round(level, price_precision),
                "source": "Fibonacci (Calculated)"
            })

    # Sort extensions by price
    fib_zones.sort(key=lambda x: x['level'], reverse=True if is_uptrend_correction else False)


    return fib_zones

# Example usage (for testing calculation logic outside the API)
if __name__ == "__main__":
    # Ensure you have numpy and pandas installed
    # Create a dummy DataFrame resembling Twelve Data output
    data = {
        'datetime': pd.to_datetime(['2023-10-27 09:30', '2023-10-27 09:31', '2023-10-27 09:32', '2023-10-27 09:33', '2023-10-27 09:34']),
        'open': [420.1, 420.5, 420.8, 420.6, 421.0],
        'high': [420.6, 420.9, 421.2, 420.9, 421.5],
        'low': [420.0, 420.4, 420.7, 420.5, 420.9],
        'close': [420.5, 420.8, 420.6, 421.0, 421.4],
        'volume': [1000, 1500, 800, 1200, 2000] # Note volume varies
    }
    dummy_df = pd.DataFrame(data)

    vp_result = calculate_volume_profile(dummy_df, price_precision=1) # Use precision 1 for this tiny sample
    print("Volume Profile Result (Precision 1):")
    print(vp_result)
    print(f"Type of total_volume: {type(vp_result['total_volume']) if vp_result else 'N/A'}")
    if vp_result:
        for key, val in vp_result.items():
            print(f"Type of {key}: {type(val)}")


    # Example with higher precision
    vp_result_high_prec = calculate_volume_profile(dummy_df, price_precision=2) # Use precision 2
    print("\nVolume Profile Result (Precision 2):")
    print(vp_result_high_prec)
    print(f"Type of total_volume: {type(vp_result_high_prec['total_volume']) if vp_result_high_prec else 'N/A'}")

    # Example with no volume
    no_volume_data = {
        'datetime': pd.to_datetime(['2023-10-27 09:30']),
        'open': [100], 'high': [100], 'low': [100], 'close': [100], 'volume': [0]
    }
    dummy_df_no_vol = pd.DataFrame(no_volume_data)
    vp_result_no_vol = calculate_volume_profile(dummy_df_no_vol)
    print("\nVolume Profile (No Volume):")
    print(vp_result_no_vol)

    # Example with only one price point
    one_point_data = {
        'datetime': pd.to_datetime(['2023-10-27 09:30']),
        'open': [100], 'high': [100], 'low': [100], 'close': [100], 'volume': [1000]
    }
    dummy_df_one_point = pd.DataFrame(one_point_data)
    vp_result_one_point = calculate_volume_profile(dummy_df_one_point)
    print("\nVolume Profile (One Point):")
    print(vp_result_one_point)

    # Example with potential NaN values in data
    nan_data = {
        'datetime': pd.to_datetime(['2023-10-27 09:30', '2023-10-27 09:31', '2023-10-27 09:32']),
        'open': [420.1, np.nan, 420.8],
        'high': [420.6, 420.9, 421.2],
        'low': [420.0, 420.4, np.nan],
        'close': [420.5, 420.8, 420.6],
        'volume': [1000, np.nan, 800]
    }
    dummy_df_nan = pd.DataFrame(nan_data)
    vp_result_nan = calculate_volume_profile(dummy_df_nan)
    print("\nVolume Profile (with NaNs):")
    print(vp_result_nan)