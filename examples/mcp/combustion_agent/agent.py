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
    agent_response = await agent.ainvoke({"messages": "I have premixed flame that is at 1 atm, 300 K, and equivalence ratio of 1.0 but my inflow velocity is a bit high at 1 m/s the flame keeps blowing off, which of the parameters should I change. I think it won't blow off if the inflow velocity is equal to the flame speed"})

    for msg in agent_response['messages']:
        msg.pretty_print()

if __name__ == "__main__":
    ret = start_tunnel("<username>","<job head node>",9276,9276)
    asyncio.run(main())
    stop_tunnel(*ret)