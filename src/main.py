import uvicorn
from fastapi import FastAPI

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.server.api.v1 import volume_profile
from src.server.api.v1 import technical_analysis_api
from src.server.config import FININSIGHT_HOST, FININSIGHT_PORT

app = FastAPI(
    title="FinancialDataMCP",
    description="MCP Server providing structured market insights for LLM trading agents.",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#API routers
app.include_router(volume_profile.router, prefix="/api/v1", tags=["Volume Profile"])

app.include_router(technical_analysis_api.router, prefix="/api/v1", tags=["Technical Analysis"])


@app.get("/")
async def read_root():
    """Root endpoint to confirm the server is running."""
    return {"message": "FinancialDataMCP Server is running."}

if __name__ == "__main__":
    print(f"Starting FinancialDataMCP Server on http://{FININSIGHT_HOST}:{FININSIGHT_PORT}")
    uvicorn.run(
        app,
        host=FININSIGHT_HOST,
        port=FININSIGHT_PORT,
    )