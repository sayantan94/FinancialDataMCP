# 🤖 FinInsight Server: Quant-Grade Market Insights for AI Trading Agents

This is a standalone server designed to deliver market data insights to AI trading agents. It processes data from external brokers and computes structured information to help LLMs and other AI systems make more informed trading decisions. Built using experience in scalable cloud systems, this project focuses on reliable data processing and delivery.

The server currently provides the following information via its API:

*   **Volume Profile Data:** Key price levels based on where trading volume has been concentrated.
*   **Standard Technical Analysis Indicators:** Common metrics like Moving Averages, RSI, MACD, and volatility measures.
*   **Technical Zones:** Pre-calculated price ranges indicating potential support and resistance levels.

## 🤝 Integration with LLM Agents (e.g., Bedrock)

This server is designed to function as a tool provider for AI agents, particularly those supporting external tool use or function calling based on API specifications.

FastAPI automatically generates an API specification following the OpenAPI standard. LLM frameworks like AWS Bedrock's Converse API can be configured to use this server as a tool.

See this sample spec file ->
[spec.json](src/tools/spec.json)
## 🚀 Getting Started

Follow these steps to get the server up and running:

1.  **Prerequisites:**
    *   Python 3.9+
    *   An API Key from a supported market data broker (e.g., Twelve Data, Polygon.io). The current implementation includes fetchers for Twelve Data.
    *   Git

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/FinancialDataMCP.git
    cd FinancialDataMCP
    ```

3.  **Set up Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4.  **Run the Server:**
    ```bash
    python src/main.py
    # Or using FastAPI's uvicorn for development with auto-reloads
    # uvicorn src.main:app --reload --host 0.0.0.0 --port 8100
    ```
    The server should start and listen on the configured port (default 8100).


Once the server is running, you can access the API endpoints. You can find the auto-generated API documentation (Swagger UI) at `http://localhost:8100/docs` in your web browser.

## 🗺️ Future Direction

Planned enhancements include adding more sophisticated insights like Dealer Gamma/Delta Exposure and Smart Money Flow signals, along with performance improvements and caching.
