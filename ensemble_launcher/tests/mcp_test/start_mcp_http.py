from ensemble_launcher.mcp import Server
from sim_script import sim

mcp = Server(port=9276)

tool = mcp.ensemble_tool(sim)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")