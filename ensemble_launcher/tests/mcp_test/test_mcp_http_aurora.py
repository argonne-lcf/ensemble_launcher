import getpass
import os
import asyncio
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from ensemble_launcher.mcp import start_tunnel, stop_tunnel



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

client = MultiServerMCPClient(
    {
        "ensemble": {
            "url":"http://127.0.0.1:9276/mcp",
            "transport": "streamable_http"
        }
    }
)

async def main():
    tools = await client.get_tools()
    agent = create_react_agent(llm,tools=tools)
    agent_response = await agent.ainvoke({"messages": "Run 10 CFD simulations for a range of temperature (300 to 2000 K) and pressure (1 to 200 atm) using the tools available. You can simply distribute them linearly"})

    for msg in agent_response['messages']:
        msg.pretty_print()

if __name__ == "__main__":
    ret = start_tunnel("ht1410","x4613c7s0b0n0",9276,9276)
    asyncio.run(main())
    stop_tunnel(*ret)