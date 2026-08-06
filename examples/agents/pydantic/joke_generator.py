import asyncio
import logging
import os
import uuid

from agents import JokeGenerationAgent, JokeSelectionAgent
from utils import start_el

from ensemble_launcher.inference import OnlineVLLMInference

logging.basicConfig(level=logging.INFO)


async def main():
    client, el = start_el()

    llm_ckpt = os.path.join("/tmp", f"llm_ckpt_{str(uuid.uuid4())[:6]}")
    generator_ckpt = os.path.join("/tmp", f"generator_ckpt_{str(uuid.uuid4())[:6]}")
    selector_ckpt = os.path.join("/tmp", f"selector_ckpt_{str(uuid.uuid4())[:6]}")
    model = "Qwen/Qwen3-0.6B"
    model_cache = os.environ.get("HF_HOME", None)

    ## Start local vllm
    llm = OnlineVLLMInference(
        name="vllm", model=model, cache_dir=model_cache,
        server_args={"port": 9000}, ckpt_dir=llm_ckpt,
    )
    llm_task = llm.create_task(task_id="llm", nnodes=1, ppn=1, ngpus_per_process=1)
    client.submit(llm_task)

    llm_handle = llm.create_handle()
    await llm_handle.open()
    llm_address = await llm_handle.get_address()
    ##
    remote_model = "Qwen/Qwen3-4B"
    remote_llm_address = "10.0.0.96:8192"

    # Start Generation Agent
    generator = JokeGenerationAgent(
        name="generator",
        llm_address=llm_address,
        llm_model=model,
        ckpt_dir=generator_ckpt,
    )
    generator_task = generator.create_task(task_id="generator", nnodes=1, ppn=1)
    client.submit(generator_task)
    generator_handle = generator.create_handle()

    ## Start Selection Agent
    selector = JokeSelectionAgent(
        name="selector",
        ckpt_dir=selector_ckpt,
        llm_address=remote_llm_address,
        llm_model=remote_model,
        generation_agent_handle=generator_handle,
    )
    selector_task = selector.create_task(task_id="selector", nnodes=1, ppn=1)
    selector_future = client.submit(selector_task)
    try:
        selector_handle = selector.create_handle(timeout=60)
        if selector_handle is not None:
            await selector_handle.open()
        else:
            raise TimeoutError
        result = await selector_handle.invoke(
            "Generate 10 jokes and tell me the best one among those to me."
        )
        logging.info(f"Received {result} from selector")
        await selector_handle.stop()
        await selector_handle.close()
    except Exception as e:
        logging.info(f"calling selctor failed with exception {e}")
        logging.info(f"selctor failed with error {selector_future.exception()}")

    await llm_handle.stop()

    el.stop()


if __name__ == "__main__":
    asyncio.run(main())
