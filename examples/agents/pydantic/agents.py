from typing import Optional

import requests
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ensemble_launcher.ensemble import Actor
from ensemble_launcher.ensemble.actor import ActorHandle, action


class JokeGenerationAgent(Actor):
    def __init__(
        self,
        name,
        transport="zmq",
        llm_address: str = "localhost:8000",
        llm_model: str = "EMPTY",
        ckpt_dir=...,
    ):
        super().__init__(name, transport, ckpt_dir)
        self._pydantic_agent = None
        self._model = llm_model
        self._remote_address = llm_address

    async def on_start(self):
        self._pydantic_agent = Agent(
            model=OpenAIChatModel(
                model_name=self._model,
                provider=OpenAIProvider(
                    openai_client=AsyncOpenAI(
                        base_url=f"http://{self._remote_address}/v1", api_key="EMPTY"
                    )
                ),
            ),
            model_settings=OpenAIChatModelSettings(tool_choice="auto"),
        )

    @action
    async def generate_jokes(self, count: int):
        """Function generate count number of jokes
        parameters:
            count: number of jokes to generate
        """
        self.logger.info(f"Prompt: Generate {count} jokes")
        result = await self._pydantic_agent.run(f"Generate {count} jokes")
        self.logger.info(f"generate jokes: {result.output}")
        return result.output


class JokeSelectionAgent(Actor):
    def __init__(
        self,
        name,
        transport="zmq",
        llm_address="localhost:8000",
        llm_model="EMPTY",
        generation_agent_handle: Optional[ActorHandle] = None,
        ckpt_dir=...,
    ):
        super().__init__(name, transport, ckpt_dir)
        self._pydantic_agent = None
        self._generation_agent_handle = generation_agent_handle
        self._remote_address = llm_address
        self._model = llm_model

    async def on_start(self):
        if self._remote_address is not None:
            self.logger.info(
                f"Started actor with model {self._model}, address {self._remote_address}"
            )
            self._pydantic_agent = Agent(
                model=OpenAIChatModel(
                    model_name=self._model,
                    provider=OpenAIProvider(
                        openai_client=AsyncOpenAI(
                            base_url=f"http://{self._remote_address}/v1",
                            api_key="EMPTY",
                        ),
                    ),
                ),
                model_settings=OpenAIChatModelSettings(tool_choice="auto"),
                instructions=(
                    "Use 'generate_jokes' tool to generate 'count' number of jokes\n"
                    "To tell a joke follow these steps: \n"
                    "1. Generate 10 jokes using tools you have.\n"
                    "2. Choose the best one among these\n"
                ),
                capabilities=[Thinking(effort="xhigh")],
            )
        if self._generation_agent_handle is not None:
            await self._generation_agent_handle.open()
            self._pydantic_agent.tool_plain(
                self._generation_agent_handle.generate_jokes
            )

    @action
    async def invoke(self, prompt: str):
        if self._generation_agent_handle is None:
            if self._remote_address:
                response = requests.post(
                    url=f"http://{self._remote_address}/v1/completions",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "prompt": prompt,
                    },
                )
                return response.json()
            else:
                return "This is a joke"
        else:
            result = await self._pydantic_agent.run(prompt)
            return result.output

    async def on_stop(self):
        if self._generation_agent_handle:
            await self._generation_agent_handle.stop()
            await self._generation_agent_handle.close()
