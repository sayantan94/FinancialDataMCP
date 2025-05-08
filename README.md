# 🤖 FinInsight Server: Quant-Grade Market Insights for AI Trading Agents


This project provides a standalone market data server designed specifically to feed sophisticated, *derived* trading insights to LLM-powered AI trading agents. 

Building an effective AI trading bot requires more than just fetching basic price data. To navigate complex, fast-moving markets, especially with strategies involving options or short-term plays, the bot needs access to:

1.  **Insight into Market Structure:** Understanding the forces beneath the surface, like institutional positioning.
2.  **Quantifiable Edge:** Identifying zones and signals with statistically higher probabilities.
3.  **Contextual Data:** Integrating multiple layers of analysis (volume, options, price action).

Standard market data APIs provide the raw materials, but the heavy lifting of transforming that data into actionable insights is left to the user. This project does that heavy lifting.

## 🚀 The Solution: Your Custom Insight Engine

The FinInsight Server acts as a specialized layer between a raw data provider and your LLM trading agent. It constantly processes market data in the background and exposes several key API endpoints that your LLM can call as "tools" to get the information it needs *exactly* when it needs it.

This server is built to provide the following critical insights:

### 1. 💪 Smart Money Flow Signals

*   **What it is:** An inferred signal attempting to identify whether large, institutional players are accumulating (buying heavily) or distributing (selling heavily) an asset, based on analyzing patterns in price and volume data that differ from typical retail behavior.
*   **Why it's important:** Provides early cues about potential significant moves or validates key price levels where professional money is active. Helps the LLM avoid trading against the dominant institutional trend.
*   **Value for Trading Bot:** Allows the bot to potentially enter trades aligned with strong market participants or avoid entering positions where significant institutional opposition is detected. Adds conviction to trade ideas.

### 2. ⚙️ Dealer Gamma & Delta Hedging Impacts (GEX/DEX)

*   **What it is:** Calculates the aggregate Gamma (GEX) and Delta (DEX) exposure of options market makers and dealers. Identifies key price levels (like Gamma Flip levels) where dealer hedging activity is likely to accelerate price moves (negative gamma) or stabilize price (positive gamma).
*   **Why it's important:** Provides crucial context for understanding potential volatility and price behavior, especially near options expiration. Reveals "hidden" forces influencing supply and demand based on derivative positioning.
*   **Value for Trading Bot:** *Essential* for 0 DTE and short-dated options strategies. Helps the bot anticipate sharp moves or pinning behavior, refine entry/exit points based on hedging pressure, and better manage risk around critical gamma levels.

### 3. 🎯 Technical Zones

*   **What it is:** Identifies precise price ranges or levels that represent high-probability support/resistance, reversal points, or potential targets. These zones are calculated using advanced technical analysis techniques, particularly Volume Profile (Value Area High/Low, Point of Control, High/Low Volume Nodes) combined with volatility metrics (like ATR) and other structural analysis.
*   **Why it's important:** Offers more robust and statistically relevant levels compared to simple horizontal lines. Based on where significant volume has traded or where market structure indicates potential inflection points.
*   **Value for Trading Bot:** Provides concrete, precise price levels for setting automated entry orders, stop losses, and take profits with potentially higher accuracy. Reduces reliance on subjective interpretation of charts.

### 4. 📊 Comprehensive TA & Volume Profile Data

*   **What it is:** Provides a structured analysis of standard Technical Analysis indicator values (Moving Averages, RSI, MACD, ATR, VWAP, etc.) alongside the key structural components of the Volume Profile for a defined period.
*   **Why it's important:** Gives the LLM agent a full technical snapshot of the asset. Allows for more complex reasoning by combining different indicator signals with volume-based structural analysis.
*   **Value for Trading Bot:** Acts as a rich data source the bot can use for cross-referencing signals, validating hypotheses, or implementing strategies that require direct access to indicator values and volume structure details.


# 🧠 How it Works with LLM Agents (e.g., Claude)
This server is designed with LLM "Tool Use" (or "Function Calling") in mind.

You provide your LLM (like Claude) with schema definitions describing each of the server's API endpoints.

As the LLM processes trading tasks or analyzes market conditions, it decides if it needs specific data (e.g., "What's the gamma situation on SPY?").

The LLM generates a tool_use request specifying which tool (API endpoint) to call and with what parameters (symbol: "SPY").

Your orchestration code intercepts this request, makes the actual HTTP call to your running FinInsight Server.

Your server processes the request, fetches data from Twelve Data if needed, performs the computation, and returns the structured JSON output.

Your orchestration code sends this JSON output back to the LLM as a tool_result.

The LLM reads the result and incorporates the computed insight into its reasoning process to make a trading decision.


## 📚 API Documentation

The FinInsight Server provides a set of REST API endpoints specifically designed for consumption by external clients, primarily your LLM trading agent using Tool Use capabilities. These endpoints allow the agent to programmatically request computed market insights based on the raw data processed by the server.

All API endpoints use the **`GET`** HTTP method and return data in **JSON** format.

The base path for all API calls is typically `/api/v1/`. For example, the endpoint for Smart Money signals for SPY would be `/api/v1/insights/SPY/smart_money`. The exact base URL (domain and port) will depend on where you deploy the server.

All successful responses include a common top-level structure with `symbol`, `timestamp_utc` (when the data was last updated/calculated), `status` ("success"), and a `message`. Error responses will include `status` ("error") and details in the `message` field.

Here are the primary API endpoints available:

---

### 1. Get Smart Money Flow Signal

*   **Endpoint:** `GET /api/v1/insights/{symbol}/smart_money`
*   **Description:** Retrieves the latest inferred smart money accumulation or distribution signal for the specified stock ticker. This signal is computed by analyzing sophisticated patterns in price and volume data.
*   **Input:**
    *   `symbol` (Path Parameter, **Required**, string): The ticker symbol (e.g., "SPY", "AAPL").
*   **Output:** A JSON object containing the `smart_money_signal` field, detailing the inferred flow (`current_flow`), its `strength`, a `confidence_score` (0-100), `detected_levels` associated with the activity, and relevant `notes`.

---

### 2. Get Dealer Gamma/Delta Exposure Data

*   **Endpoint:** `GET /api/v1/insights/{symbol}/gamma_delta`
*   **Description:** Provides calculated aggregate Gamma (GEX) and Delta (DEX) exposure for market makers, identifying key price levels influenced by their hedging activities for a given expiration.
*   **Input:**
    *   `symbol` (Path Parameter, **Required**, string): The ticker symbol.
    *   `expiry_date` (Query Parameter, Optional, string): The options expiration date in `YYYY-MM-DD` format. If omitted, the server defaults to calculating for the nearest/current day's expiration.
*   **Output:** A JSON object containing the `gamma_delta_exposure` field, including total `gex_total_shares` and `dex_total_shares`, the `gamma_flip_level`, a list of `key_gamma_levels` (strikes with high concentration), and a summary of the `implied_hedging_impact`. Also includes the `expiry_date` and `days_to_expiry` context.

---

### 3. Get High-Probability Technical Zones

*   **Endpoint:** `GET /api/v1/insights/{symbol}/technical_zones`
*   **Description:** Returns a list of pre-computed high-probability support, resistance, and target price zones derived from advanced technical analysis methods like Volume Profile and volatility extensions.
*   **Input:**
    *   `symbol` (Path Parameter, **Required**, string): The ticker symbol.
    *   `timeframe` (Query Parameter, Optional, string): The lookback period or context for the zone calculation (e.g., "day", "week", "last_5_days"). Defaults to "day".
*   **Output:** A JSON object containing the `technical_zones` field, which is a list of zone objects. Each zone includes its `type` ("SUPPORT", "RESISTANCE", etc.), `name`, `level` (for lines) or `range_start`/`range_end` (for ranges), `confidence_score` (0-100), and `source` (how it was derived).

---

### 4. Get Combined Technical Analysis and Volume Profile Structure

*   **Endpoint:** `GET /api/v1/ta_volume_profile/{symbol}`
*   **Description:** Provides a comprehensive snapshot of standard technical analysis indicator values and the key structural components of the Volume Profile for a specified timeframe.
*   **Input:**
    *   `symbol` (Path Parameter, **Required**, string): The ticker symbol.
    *   `timeframe` (Query Parameter, Optional, string): The lookback period or calculation context (e.g., "day", "week"). Defaults to "day".
*   **Output:** A JSON object containing two main fields: `technical_analysis` (with sub-fields for `moving_averages`, `oscillators`, `volatility`, `vwap`, etc.) and `volume_profile_structure` (with fields like `point_of_control`, `value_area_high`, `value_area_low`, `high_volume_nodes`, `low_volume_nodes`, `total_volume`).

---

Detailed documentation, including specific indicator parameters, full schema definitions, and error codes, will be provided in `docs/API_REFERENCE.md` (Coming Soon).