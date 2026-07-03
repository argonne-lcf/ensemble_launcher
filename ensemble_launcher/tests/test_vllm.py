import asyncio
import json
import logging
import os
import subprocess
import uuid

import pytest

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm.pipe import transport_registry
from ensemble_launcher.config import (
    LauncherConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import (
    OnlineVLLMInference,
    PrivateOnlineVLLMInference,
    PrivateVLLMInference,
    VLLMInference,
)
from ensemble_launcher.orchestrator import ClusterClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pytestmark = pytest.mark.extensions


@pytest.mark.asyncio
async def test_offline_inference():
    model = "meta-llama/Llama-3.2-1B"
    model_cache = os.environ.get("HF_HOME", None)

    nodes = get_nodes()
    logger.info("Resolved nodes: %s", nodes)

    sys_config = SystemConfig(
        name="aurora", ncpus=12, ngpus=1, cpus=list(range(12)), gpus=list(range(1))
    )
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    launcher_config = LauncherConfig(
        child_executor_name="async_processpool",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=0, nchildren=len(nodes)),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=nodes,
    )

    handle = None
    future = None
    try:
        el.start()
        logger.info("Cluster started (checkpoint_dir=%s)", ckpt_dir)
        await asyncio.sleep(10.0)

        actor_id = "vllm-actor"
        actor_ckpt = os.path.join("/tmp", f".vllm_{uuid.uuid4().hex[:6]}")
        actor = VLLMInference(
            name=actor_id,
            transport="zmq",
            model=model,
            cache_dir=model_cache,
            ckpt_dir=actor_ckpt,
        )
        task = actor.create_task(
            actor_id,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
        )

        with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
            future = client.submit(task)
            logger.info("Actor task submitted: %s", actor_id)


            handle = actor.create_handle(timeout=60)
            if handle is None:
                if future.done() and future.exception() is not None:
                    logger.exception(
                        "Actor task failed before handle could be created",
                        exc_info=future.exception(),
                    )
                    raise future.exception()
                raise TimeoutError(
                    f"Failed to create handle for actor {actor_id} "
                    f"(future done={future.done()})"
                )

            logger.info("Handle created, opening transport connection")
            try:
                await handle.open()
            except Exception:
                logger.exception("Failed to open handle for actor %s", actor_id)
                raise

            logger.info("Sending inference request")
            result = await asyncio.wait_for(handle.generate("hello"), timeout=120)
            logger.info("Inference result: %s", result)

    except Exception:
        logger.exception("Test failed")
        raise
    finally:
        if handle is not None:
            try:
                await handle.stop()
            except Exception:
                logger.exception("Error stopping handle")
            try:
                await handle.close()
            except Exception:
                logger.exception("Error closing handle")

        if future is not None and not future.done():
            future.cancel()
            logger.info("Cancelled incomplete actor future")

        el.stop()
        logger.info("Cluster stopped")


@pytest.mark.asyncio
async def test_offline_inference_async_engine():
    model = "meta-llama/Llama-3.2-1B"
    model_cache = os.environ.get("HF_HOME", None)

    nodes = get_nodes()
    logger.info("Resolved nodes: %s", nodes)

    sys_config = SystemConfig(
        name="aurora", ncpus=12, ngpus=1, cpus=list(range(12)), gpus=list(range(1))
    )
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    launcher_config = LauncherConfig(
        child_executor_name="async_processpool",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=0, nchildren=len(nodes)),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=nodes,
    )

    handle = None
    future = None
    try:
        el.start()
        logger.info("Cluster started (checkpoint_dir=%s)", ckpt_dir)
        await asyncio.sleep(10.0)

        actor_id = "vllm-actor-async"
        actor_ckpt = os.path.join("/tmp", f".vllm_{uuid.uuid4().hex[:6]}")
        actor = VLLMInference(
            name=actor_id,
            transport="zmq",
            model=model,
            cache_dir=model_cache,
            ckpt_dir=actor_ckpt,
            async_engine=True,
            max_workers=2,
        )
        task = actor.create_task(
            actor_id,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
        )

        with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
            future = client.submit(task)
            logger.info("Actor task submitted: %s", actor_id)

            handle = actor.create_handle(timeout=600)
            if handle is None:
                if future.done() and future.exception() is not None:
                    logger.exception(
                        "Actor task failed before handle could be created",
                        exc_info=future.exception(),
                    )
                    raise future.exception()
                raise TimeoutError(
                    f"Failed to create handle for actor {actor_id} "
                    f"(future done={future.done()})"
                )

            logger.info("Handle created, opening transport connection")
            try:
                await handle.open()
            except Exception:
                logger.exception("Failed to open handle for actor %s", actor_id)
                raise

            logger.info("Sending inference request")
            result = await asyncio.wait_for(handle.generate("hello"), timeout=120)
            logger.info("Inference result: %s", result)

    except Exception:
        logger.exception("Test failed")
        raise
    finally:
        if handle is not None:
            try:
                await handle.stop()
            except Exception:
                logger.exception("Error stopping handle")
            try:
                await handle.close()
            except Exception:
                logger.exception("Error closing handle")

        if future is not None and not future.done():
            future.cancel()
            logger.info("Cancelled incomplete actor future")

        el.stop()
        logger.info("Cluster stopped")


@pytest.mark.asyncio
async def test_online_inference():
    model = "meta-llama/Llama-3.2-1B"
    model_cache = os.environ.get("HF_HOME", None)

    nodes = get_nodes()
    logger.info("Resolved nodes: %s", nodes)

    sys_config = SystemConfig(
        name="aurora", ncpus=12, ngpus=1, cpus=list(range(12)), gpus=list(range(1))
    )
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    launcher_config = LauncherConfig(
        child_executor_name="async_processpool",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=0, nchildren=len(nodes)),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=nodes,
    )

    handle = None
    future = None
    try:
        el.start()
        logger.info("Cluster started (checkpoint_dir=%s)", ckpt_dir)
        await asyncio.sleep(10.0)

        actor_id = "vllm-actor"
        actor_ckpt = os.path.join("/tmp", f".vllm_{uuid.uuid4().hex[:6]}")
        actor = OnlineVLLMInference(
            name=actor_id,
            transport="zmq",
            model=model,
            cache_dir=model_cache,
            server_args={"tensor_parallel_size":1, "port":8001},
            ckpt_dir=actor_ckpt,
        )

        task = actor.create_task(
            actor_id,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
        )

        with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
            future = client.submit(task)
            logger.info("Actor task submitted: %s", actor_id)

            handle = actor.create_handle(timeout=600)
            if handle is None:
                if future.done() and future.exception() is not None:
                    logger.exception(
                        "Actor task failed before handle could be created",
                        exc_info=future.exception(),
                    )
                    raise future.exception()
                raise TimeoutError(
                    f"Failed to create handle for actor {actor_id} "
                    f"(future done={future.done()})"
                )

            logger.info("Handle created, opening transport connection")
            try:
                await handle.open()
            except Exception:
                logger.exception("Failed to open handle for actor %s", actor_id)
                raise

            address = await handle.get_address()
            logger.info("Sending inference request")
            payload = {
                "model": "meta-llama/Llama-3.2-1B",
                "prompt": "Once upon a time,",
                "max_tokens": 512,
                "temperature": 0.5,
            }
            cmd = [
                "curl",
                "-X",
                "POST",
                f"http://{address}/v1/completions",
                "-H",
                "Content-Type: application/json",
                "--data",
                json.dumps(payload),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            result = p.stdout
            logger.info("Inference result: %s", result)

    except Exception:
        logger.exception("Test failed")
        raise
    finally:
        if handle is not None:
            try:
                await handle.stop()
            except Exception:
                logger.exception("Error stopping handle")
            try:
                await handle.close()
            except Exception:
                logger.exception("Error closing handle")

        if future is not None and not future.done():
            future.cancel()
            logger.info("Cancelled incomplete actor future")

        el.stop()
        logger.info("Cluster stopped")


@pytest.mark.asyncio
async def test_private_offline_inference():
    model = "meta-llama/Llama-3.2-1B"
    model_cache = os.environ.get("HF_HOME", None)

    nodes = get_nodes()
    logger.info("Resolved nodes: %s", nodes)

    sys_config = SystemConfig(
        name="aurora", ncpus=12, ngpus=1, cpus=list(range(12)), gpus=list(range(1))
    )
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    launcher_config = LauncherConfig(
        child_executor_name="async_processpool",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=0, nchildren=len(nodes)),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=nodes,
    )

    handle = None
    future = None
    try:
        el.start()
        logger.info("Cluster started (checkpoint_dir=%s)", ckpt_dir)
        await asyncio.sleep(10.0)

        actor_id = "vllm-private-actor"
        secret = "secret"
        transport = transport_registry.get("zmq")["transport"]()
        server = transport.get_server_connection("parent", secret, address=None)
        client = transport.get_client_connection(
            actor_id,
            secret,
            remote_address=server.address,
            remote_identity="parent",
            remote_secret_id=secret,
        )
        server.add_expected_remote(actor_id, secret)
        actor = PrivateVLLMInference(
            name=actor_id,
            model=model,
            cache_dir=model_cache,
            client_conn=client,
        )
        task = actor.create_task(
            actor_id,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
        )
        handle = PrivateVLLMInference.create_handle(server)

        with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as cl:
            future = cl.submit(task)
            logger.info("Private actor task submitted: %s", actor_id)

            logger.info("Opening transport connection")
            await handle.open()
            await handle.wait_for_ready(expected=1)

            target_id = f"{actor_id}:{secret}"
            logger.info("Sending inference request")
            result = await asyncio.wait_for(handle.generate("hello",actor_id=target_id), timeout=120)
            logger.info("Inference result: %s", result)

    except Exception:
        logger.exception("Test failed")
        raise
    finally:
        if handle is not None:
            try:
                await handle.stop()
            except Exception:
                logger.exception("Error stopping handle")

        if future is not None:
            try:
                future.result(timeout=30)
            except Exception:
                logger.exception("Actor future did not complete cleanly")

        if handle is not None:
            try:
                await handle.close()
            except Exception:
                logger.exception("Error closing handle")

        el.stop()
        logger.info("Cluster stopped")


@pytest.mark.asyncio
async def test_private_offline_inference_async_engine():
    model = "meta-llama/Llama-3.2-1B"
    model_cache = os.environ.get("HF_HOME", None)

    nodes = get_nodes()
    logger.info("Resolved nodes: %s", nodes)

    sys_config = SystemConfig(
        name="aurora", ncpus=12, ngpus=1, cpus=list(range(12)), gpus=list(range(1))
    )
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    launcher_config = LauncherConfig(
        child_executor_name="async_processpool",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=0, nchildren=len(nodes)),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=nodes,
    )

    handle = None
    future = None
    try:
        el.start()
        logger.info("Cluster started (checkpoint_dir=%s)", ckpt_dir)
        await asyncio.sleep(10.0)

        actor_id = "vllm-private-actor-async"
        secret = "secret"
        transport = transport_registry.get("zmq")["transport"]()
        server = transport.get_server_connection("parent-async", secret, address=None)
        client = transport.get_client_connection(
            actor_id,
            secret,
            remote_address=server.address,
            remote_identity="parent-async",
            remote_secret_id=secret,
        )
        server.add_expected_remote(actor_id, secret)
        actor = PrivateVLLMInference(
            name=actor_id,
            model=model,
            cache_dir=model_cache,
            client_conn=client,
            async_engine=True,
            max_workers=2,
        )
        task = actor.create_task(
            actor_id,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
        )
        handle = PrivateVLLMInference.create_handle(server)

        with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as cl:
            future = cl.submit(task)
            logger.info("Private async actor task submitted: %s", actor_id)

            logger.info("Opening transport connection")
            await handle.open()
            await handle.wait_for_ready(expected=1)

            target_id = f"{actor_id}:{secret}"
            logger.info("Sending inference request")
            result = await asyncio.wait_for(handle.generate("hello",actor_id=target_id), timeout=120)
            logger.info("Inference result: %s", result)

    except Exception:
        logger.exception("Test failed")
        raise
    finally:
        if handle is not None:
            try:
                await handle.stop()
            except Exception:
                logger.exception("Error stopping handle")

        if future is not None:
            try:
                future.result(timeout=30)
            except Exception:
                logger.exception("Actor future did not complete cleanly")

        if handle is not None:
            try:
                await handle.close()
            except Exception:
                logger.exception("Error closing handle")

        el.stop()
        logger.info("Cluster stopped")


@pytest.mark.asyncio
async def test_private_online_inference():
    model = "meta-llama/Llama-3.2-1B"
    model_cache = os.environ.get("HF_HOME", None)

    nodes = get_nodes()
    logger.info("Resolved nodes: %s", nodes)

    sys_config = SystemConfig(
        name="aurora", ncpus=12, ngpus=1, cpus=list(range(12)), gpus=list(range(1))
    )
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    launcher_config = LauncherConfig(
        child_executor_name="async_processpool",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=0, nchildren=len(nodes)),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=nodes,
    )

    handle = None
    future = None
    try:
        el.start()
        logger.info("Cluster started (checkpoint_dir=%s)", ckpt_dir)
        await asyncio.sleep(10.0)

        actor_id = "vllm-private-online"
        secret = "secret"
        transport = transport_registry.get("zmq")["transport"]()
        server = transport.get_server_connection("parent-online", secret, address=None)
        client = transport.get_client_connection(
            actor_id,
            secret,
            remote_address=server.address,
            remote_identity="parent-online",
            remote_secret_id=secret,
        )
        server.add_expected_remote(actor_id, secret)
        actor = PrivateOnlineVLLMInference(
            name=actor_id,
            model=model,
            cache_dir=model_cache,
            client_conn=client,
            tensor_parallel_size=1,
            port=8002,
        )

        task = actor.create_task(
            actor_id,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
        )
        handle = PrivateOnlineVLLMInference.create_handle(server)

        with ClusterClient(
            checkpoint_dir=ckpt_dir, checkpoint_timeout=300, task_buffer_size=0
        ) as cl:
            future = cl.submit(task)
            logger.info("Private online actor task submitted: %s", actor_id)

            logger.info("Opening transport connection")
            await handle.open()
            await handle.wait_for_ready(expected=1)

            target_id = f"{actor_id}:{secret}"
            address = await handle.get_address(actor_id=target_id)
            logger.info("vLLM server address: %s", address)

            payload = {
                "model": "meta-llama/Llama-3.2-1B",
                "prompt": "Once upon a time,",
                "max_tokens": 512,
                "temperature": 0.5,
            }
            cmd = [
                "curl",
                "-X",
                "POST",
                f"http://{address}/v1/completions",
                "-H",
                "Content-Type: application/json",
                "--data",
                json.dumps(payload),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            result = p.stdout
            logger.info("Inference result: %s", result)

    except Exception:
        logger.exception("Test failed")
        raise
    finally:
        if handle is not None:
            try:
                await handle.stop()
            except Exception:
                logger.exception("Error stopping handle")

        if future is not None:
            try:
                future.result(timeout=30)
            except Exception:
                logger.exception("Actor future did not complete cleanly")

        if handle is not None:
            try:
                await handle.close()
            except Exception:
                logger.exception("Error closing handle")

        el.stop()
        logger.info("Cluster stopped")


if __name__ == "__main__":
    asyncio.run(test_private_online_inference())
