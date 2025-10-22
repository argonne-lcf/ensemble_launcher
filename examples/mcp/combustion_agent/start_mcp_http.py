from ensemble_launcher.mcp import Server
from flamespeed import compute_flame_speed

mcp = Server(port=9276)

tool = mcp.ensemble_tool(compute_flame_speed)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")