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

def calculate_ichimoku(
        df: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
        price_precision: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Calculates Ichimoku Cloud components from historical price data.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        tenkan_period: Period for Tenkan-sen (Conversion Line), default 9
        kijun_period: Period for Kijun-sen (Base Line), default 26
        senkou_span_b_period: Period for Senkou Span B (Leading Span B), default 52
        price_precision: Decimal places for rounding prices

    Returns:
        Dictionary with Ichimoku components or None if calculation fails
    """
    # Ensure required columns exist and DataFrame has sufficient data
    if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
        print("Ichimoku calculation skipped: Input DataFrame is empty or lacks required columns.")
        return None

    # Need at least the longest period for calculation
    min_required_periods = max(tenkan_period, kijun_period, senkou_span_b_period)
    if len(df) < min_required_periods:
        print(f"Ichimoku calculation skipped: Input DataFrame has insufficient data ({len(df)} rows, need {min_required_periods}).")
        return None

    # Function to calculate a donchian channel midpoint (average of highest high and lowest low over n periods)
    def donchian_midpoint(high_series, low_series, period):
        highest_high = high_series.rolling(window=period).max()
        lowest_low = low_series.rolling(window=period).min()
        return (highest_high + lowest_low) / 2

    # Calculate Tenkan-sen (Conversion Line) - 9-period midpoint
    df['tenkan_sen'] = donchian_midpoint(df['high'], df['low'], tenkan_period)

    # Calculate Kijun-sen (Base Line) - 26-period midpoint
    df['kijun_sen'] = donchian_midpoint(df['high'], df['low'], kijun_period)

    # Calculate Senkou Span A (Leading Span A) - Average of Tenkan-sen and Kijun-sen, shifted forward 26 periods
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)

    # Calculate Senkou Span B (Leading Span B) - 52-period midpoint, shifted forward 26 periods
    df['senkou_span_b'] = donchian_midpoint(df['high'], df['low'], senkou_span_b_period).shift(kijun_period)

    # Calculate Chikou Span (Lagging Span) - Current close, shifted backwards 26 periods
    df['chikou_span'] = df['close'].shift(-kijun_period)

    # Get the most recent values (last row)
    last_row = df.iloc[-1]

    # Determine cloud status (above/below/in cloud)
    current_close = last_row['close']
    # Check if Senkou Span A and B values are both available for the latest bar
    if pd.notna(last_row['senkou_span_a']) and pd.notna(last_row['senkou_span_b']):
        # Define upper and lower bounds of cloud
        cloud_top = max(last_row['senkou_span_a'], last_row['senkou_span_b'])
        cloud_bottom = min(last_row['senkou_span_a'], last_row['senkou_span_b'])

        if current_close > cloud_top:
            cloud_status = "bullish"  # Price above cloud
        elif current_close < cloud_bottom:
            cloud_status = "bearish"  # Price below cloud
        else:
            cloud_status = "neutral"  # Price inside cloud
    else:
        cloud_status = "unknown"  # Not enough data to determine

    # Check if we have valid values for all components
    components = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
    valid_components = all(pd.notna(last_row[component]) for component in components)

    if not valid_components:
        print("Ichimoku calculation warning: Some components have NaN values.")

    # Package results with proper formatting
    ichimoku_data = {
        "tenkan_sen": round(float(last_row['tenkan_sen']), price_precision) if pd.notna(last_row['tenkan_sen']) else None,
        "kijun_sen": round(float(last_row['kijun_sen']), price_precision) if pd.notna(last_row['kijun_sen']) else None,
        "senkou_span_a": round(float(last_row['senkou_span_a']), price_precision) if pd.notna(last_row['senkou_span_a']) else None,
        "senkou_span_b": round(float(last_row['senkou_span_b']), price_precision) if pd.notna(last_row['senkou_span_b']) else None,
        "chikou_span": round(float(last_row['chikou_span']), price_precision) if pd.notna(last_row['chikou_span']) else None,
        "cloud_status": cloud_status
    }

    return ichimoku_data

def calculate_on_balance_volume(df: pd.DataFrame) -> float:
    """
    Calculates On-Balance Volume (OBV) from historical price/volume data.
    OBV adds volume on up days and subtracts volume on down days.

    Args:
        df: DataFrame with 'close' and 'volume' columns

    Returns:
        Current OBV value as float
    """
    if df is None or df.empty or not all(col in df.columns for col in ['close', 'volume']):
        print("OBV calculation skipped: Input DataFrame is empty or lacks required columns.")
        return None

    # Calculate daily price changes
    df = df.copy()
    df['price_change'] = df['close'].diff()

    # Initialize OBV with first row's volume
    obv = [0]

    # Calculate OBV for each row after the first
    for i in range(1, len(df)):
        if df['price_change'].iloc[i] > 0:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['price_change'].iloc[i] < 0:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])

    df['obv'] = obv

    # Return latest OBV value
    return float(df['obv'].iloc[-1])

def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> float:
    """
    Calculates Chaikin Money Flow (CMF) which measures buying and selling pressure.
    Positive CMF indicates buying pressure, negative indicates selling pressure.

    Args:
        df: DataFrame with 'high', 'low', 'close' and 'volume' columns
        period: Lookback period for calculation, default 20

    Returns:
        Current CMF value as float
    """
    if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        print("CMF calculation skipped: Input DataFrame is empty or lacks required columns.")
        return None

    if len(df) < period:
        print(f"CMF calculation skipped: Input DataFrame has insufficient data ({len(df)} rows, need {period}).")
        return None

    df = df.copy()

    # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
    df['high_low_range'] = df['high'] - df['low']
    df.loc[df['high_low_range'] == 0, 'high_low_range'] = 0.001  # Avoid division by zero

    df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / df['high_low_range']

    # Money Flow Volume: Money Flow Multiplier * Volume
    df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']

    # Chaikin Money Flow: Sum of Money Flow Volume over period / Sum of Volume over period
    df['cmf'] = df['money_flow_volume'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

    # Return latest CMF value
    return float(df['cmf'].iloc[-1]) if pd.notna(df['cmf'].iloc[-1]) else 0.0

def calculate_volume_price_trend(df: pd.DataFrame) -> float:
    """
    Calculates Volume Price Trend (VPT), a volume-based indicator that shows
    the balance between demand and supply.

    Args:
        df: DataFrame with 'close' and 'volume' columns

    Returns:
        Current VPT value as float
    """
    if df is None or df.empty or not all(col in df.columns for col in ['close', 'volume']):
        print("VPT calculation skipped: Input DataFrame is empty or lacks required columns.")
        return None

    df = df.copy()

    # Calculate percentage price change
    df['price_change_pct'] = df['close'].pct_change()

    # Initialize VPT with first non-NaN row's volume
    vpt = [0]

    # Calculate VPT for each row after the first
    for i in range(1, len(df)):
        if pd.notna(df['price_change_pct'].iloc[i]):
            vpt.append(vpt[-1] + df['volume'].iloc[i] * df['price_change_pct'].iloc[i])
        else:
            vpt.append(vpt[-1])

    df['vpt'] = vpt

    # Return latest VPT value
    return float(df['vpt'].iloc[-1])

def calculate_buying_pressure(df: pd.DataFrame, lookback: int = 10) -> Dict[str, float]:
    """
    Analyzes volume distribution on up vs down bars to determine buying pressure.

    Args:
        df: DataFrame with 'close' and 'volume' columns
        lookback: Number of bars to analyze for trend

    Returns:
        Dictionary with buying pressure metrics
    """
    if df is None or df.empty or not all(col in df.columns for col in ['close', 'volume']):
        print("Buying pressure calculation skipped: Input DataFrame is empty or lacks required columns.")
        return None

    if len(df) < lookback:
        print(f"Buying pressure calculation skipped: Insufficient data ({len(df)} rows, need {lookback}).")
        return None

    df = df.copy()
    df['price_change'] = df['close'].diff()

    # Analyze only the most recent bars within lookback period
    recent_df = df.tail(lookback)

    # Calculate up and down volume
    up_volume = recent_df.loc[recent_df['price_change'] > 0, 'volume'].sum()
    down_volume = recent_df.loc[recent_df['price_change'] < 0, 'volume'].sum()
    total_volume = recent_df['volume'].sum()

    # Calculate volume ratios
    up_volume_ratio = up_volume / total_volume if total_volume > 0 else 0
    down_volume_ratio = down_volume / total_volume if total_volume > 0 else 0

    # Calculate average volume on up vs down days
    avg_up_volume = recent_df.loc[recent_df['price_change'] > 0, 'volume'].mean() if len(recent_df.loc[recent_df['price_change'] > 0]) > 0 else 0
    avg_down_volume = recent_df.loc[recent_df['price_change'] < 0, 'volume'].mean() if len(recent_df.loc[recent_df['price_change'] < 0]) > 0 else 0

    # Volume strength ratio (avg volume on up days / avg volume on down days)
    volume_strength_ratio = avg_up_volume / avg_down_volume if avg_down_volume > 0 else float('inf')

    return {
        "up_volume_ratio": float(up_volume_ratio),
        "down_volume_ratio": float(down_volume_ratio),
        "volume_strength_ratio": float(volume_strength_ratio),
        "is_bullish_volume": bool(up_volume_ratio > 0.6 or volume_strength_ratio > 1.5)
    }

def identify_large_volume_bars(df: pd.DataFrame, threshold: float = 1.5) -> List[Dict[str, Any]]:
    """
    Identifies unusually large volume bars that often precede continuation moves.

    Args:
        df: DataFrame with 'volume' and 'close' columns
        threshold: Multiple of average volume to consider 'large'

    Returns:
        List of dictionaries with details of large volume bars
    """
    if df is None or df.empty or not all(col in df.columns for col in ['close', 'volume']):
        print("Large volume bar detection skipped: Input DataFrame is empty or lacks required columns.")
        return []

    if len(df) < 5:  # Need at least a few bars to establish average
        print(f"Large volume bar detection skipped: Insufficient data ({len(df)} rows).")
        return []

    df = df.copy()

    # Calculate average volume (excluding the current bar)
    avg_volume = df['volume'].iloc[:-1].mean()

    # Identify large volume bars
    large_bars = []
    lookback = min(10, len(df))  # Look at the most recent 10 bars

    for i in range(len(df) - lookback, len(df)):
        if df['volume'].iloc[i] > avg_volume * threshold:
            # Determine if it's a bullish or bearish volume bar
            price_change = df['close'].iloc[i] - df['close'].iloc[i-1] if i > 0 else 0
            bar_type = "bullish" if price_change > 0 else "bearish"

            large_bars.append({
                "bar_index": i,
                "datetime": df.index[i].isoformat() if isinstance(df.index, pd.DatetimeIndex) else str(i),
                "volume": float(df['volume'].iloc[i]),
                "volume_ratio": float(df['volume'].iloc[i] / avg_volume),
                "type": bar_type
            })

    return large_bars

def calculate_volume_momentum_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive function to calculate multiple volume-based momentum indicators.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with volume momentum indicators
    """
    if df is None or df.empty:
        return {}

    result = {}

    # Calculate OBV
    obv = calculate_on_balance_volume(df)
    if obv is not None:
        result["obv"] = obv

    # Calculate CMF
    cmf = calculate_chaikin_money_flow(df)
    if cmf is not None:
        result["cmf"] = cmf

    # Calculate VPT
    vpt = calculate_volume_price_trend(df)
    if vpt is not None:
        result["vpt"] = vpt

    # Calculate Buying Pressure
    buying_pressure = calculate_buying_pressure(df)
    if buying_pressure is not None:
        result["buying_pressure"] = buying_pressure

    # Identify Large Volume Bars
    large_volume_bars = identify_large_volume_bars(df)
    if large_volume_bars:
        result["large_volume_bars"] = large_volume_bars

    return result

def calculate_trend_strength(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates various trend strength metrics.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with trend strength metrics
    """
    if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
        return {}

    result = {}

    # Calculate ADX (Average Directional Index)
    # This is a simplified calculation - consider using a library for production
    try:
        # Make a deep copy of the dataframe to avoid SettingWithCopyWarning
        temp_df = df.copy(deep=True)

        # Calculate True Range
        temp_df['tr1'] = abs(temp_df['high'] - temp_df['low'])
        temp_df['tr2'] = abs(temp_df['high'] - temp_df['close'].shift(1))
        temp_df['tr3'] = abs(temp_df['low'] - temp_df['close'].shift(1))
        temp_df['tr'] = temp_df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Initialize +DM and -DM columns with zeros
        temp_df['plus_dm'] = 0.0
        temp_df['minus_dm'] = 0.0

        # Create temporary columns to avoid loop
        high_diff = temp_df['high'].diff()
        low_diff = temp_df['low'].diff().multiply(-1)  # Invert for easier comparison

        # Calculate +DM: if high_diff > low_diff and high_diff > 0, use high_diff, else 0
        temp_df['plus_dm'] = np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0.0
        )

        # Calculate -DM: if low_diff > high_diff and low_diff > 0, use low_diff, else 0
        temp_df['minus_dm'] = np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0.0
        )

        # Calculate smoothed TR and DM over 14 periods
        period = 14
        temp_df['smoothed_tr'] = temp_df['tr'].rolling(window=period).sum()
        temp_df['smoothed_plus_dm'] = temp_df['plus_dm'].rolling(window=period).sum()
        temp_df['smoothed_minus_dm'] = temp_df['minus_dm'].rolling(window=period).sum()

        # Calculate +DI and -DI
        temp_df['plus_di'] = 100 * temp_df['smoothed_plus_dm'] / temp_df['smoothed_tr']
        temp_df['minus_di'] = 100 * temp_df['smoothed_minus_dm'] / temp_df['smoothed_tr']

        # Calculate DX and ADX
        temp_df['dx'] = 100 * abs(temp_df['plus_di'] - temp_df['minus_di']) / (temp_df['plus_di'] + temp_df['minus_di'])
        temp_df['adx'] = temp_df['dx'].rolling(window=period).mean()

        # Get latest ADX value
        adx_value = float(temp_df['adx'].iloc[-1]) if pd.notna(temp_df['adx'].iloc[-1]) else 0.0
        result['adx'] = adx_value
        result['trend_strength'] = "strong" if adx_value > 25 else "weak"
    except Exception as e:
        print(f"Error calculating ADX: {str(e)}")

    # Calculate slope of EMA20
    try:
        # First calculate EMA20
        ema20 = df['close'].ewm(span=20, adjust=False).mean()

        # Calculate slope over last 5 periods
        lookback = 5
        if len(ema20) >= lookback:
            y_values = ema20.tail(lookback).values
            x_values = np.arange(lookback)
            slope, _ = np.polyfit(x_values, y_values, 1)
            result['ema20_slope'] = float(slope)
            result['ema20_slope_direction'] = "up" if slope > 0 else "down"
    except Exception as e:
        print(f"Error calculating EMA slope: {str(e)}")

    # Calculate price distance from moving averages
    try:
        # Calculate SMA50
        sma50 = df['close'].rolling(window=50).mean()

        # Get current price and SMA50
        current_price = df['close'].iloc[-1]
        latest_sma50 = sma50.iloc[-1]

        # Calculate distance as percentage
        if pd.notna(latest_sma50) and latest_sma50 > 0:
            distance_pct = (current_price / latest_sma50 - 1) * 100
            result['price_distance_from_sma50'] = float(distance_pct)
    except Exception as e:
        print(f"Error calculating price distance: {str(e)}")

    return result

def detect_divergences(price_df: pd.DataFrame, lookback: int = 14) -> Dict[str, Any]:
    """
    Detects potential divergences between price and momentum indicators.
    Bullish divergence: Lower price lows but higher indicator lows
    Bearish divergence: Higher price highs but lower indicator highs

    Args:
        price_df: DataFrame with OHLCV data
        lookback: Number of bars to analyze for divergence

    Returns:
        Dictionary with divergence information
    """
    if price_df is None or price_df.empty or not all(col in price_df.columns for col in ['close']):
        return {}

    if len(price_df) < lookback:
        return {}

    df = price_df.copy()

    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Look for price lows and highs in recent data
    recent_df = df.tail(lookback)

    # Find local minima and maxima
    price_lows = []
    price_highs = []
    rsi_at_price_lows = []
    rsi_at_price_highs = []

    for i in range(1, len(recent_df) - 1):
        # Check for price low
        if recent_df['close'].iloc[i] < recent_df['close'].iloc[i-1] and recent_df['close'].iloc[i] < recent_df['close'].iloc[i+1]:
            price_lows.append(recent_df['close'].iloc[i])
            rsi_at_price_lows.append(recent_df['rsi'].iloc[i])

        # Check for price high
        if recent_df['close'].iloc[i] > recent_df['close'].iloc[i-1] and recent_df['close'].iloc[i] > recent_df['close'].iloc[i+1]:
            price_highs.append(recent_df['close'].iloc[i])
            rsi_at_price_highs.append(recent_df['rsi'].iloc[i])

    result = {}

    # Detect bullish divergence (price making lower lows but RSI making higher lows)
    if len(price_lows) >= 2 and len(rsi_at_price_lows) >= 2:
        if price_lows[-1] < price_lows[-2] and rsi_at_price_lows[-1] > rsi_at_price_lows[-2]:
            result['bullish_divergence'] = True
        else:
            result['bullish_divergence'] = False

    # Detect bearish divergence (price making higher highs but RSI making lower highs)
    if len(price_highs) >= 2 and len(rsi_at_price_highs) >= 2:
        if price_highs[-1] > price_highs[-2] and rsi_at_price_highs[-1] < rsi_at_price_highs[-2]:
            result['bearish_divergence'] = True
        else:
            result['bearish_divergence'] = False

    return result

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

    # Test Ichimoku Cloud calculation
    # Create a larger dataset for proper Ichimoku testing
    import numpy as np
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # Create sample price data with a clear trend
    highs = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    lows = np.linspace(90, 180, 100) + np.random.normal(0, 5, 100)
    closes = np.linspace(95, 190, 100) + np.random.normal(0, 3, 100)

    ichimoku_test_data = {
        'datetime': dates,
        'open': np.linspace(95, 190, 100),
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(1000, 5000, 100)
    }

    ichimoku_df = pd.DataFrame(ichimoku_test_data)

    # Test Ichimoku Cloud calculation
    ichimoku_result = calculate_ichimoku(ichimoku_df)
    print("\nIchimoku Cloud Result:")
    print(ichimoku_result)

