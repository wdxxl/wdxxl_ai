# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Define a simple tool
@mcp.tool()
def say_hello(name: str) -> str:
    """Returns a greeting message."""
    return f"Hello, {name}!"

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")