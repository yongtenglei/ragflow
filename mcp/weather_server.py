# weather_server.py

import random
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Service")


@mcp.tool()
async def get_current_weather(location: str) -> str:
    """
    Get current weather of location.

    Args:
        location: location string (e.g. Shanghai)
    """
    weather_conditions = ["晴天", "多云", "雨天"]
    random_weather = random.choice(weather_conditions)
    return f"{location}今天是{random_weather}。"


@mcp.tool()
async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
