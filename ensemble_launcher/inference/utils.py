from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import random
import socket
import uuid
from glob import glob
from logging import Logger
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from time import perf_counter


def _build_model_cache(logger: Logger = None) -> int:
    """
    Contributed by
        Riccardo Balin (ALCF): https://github.com/rickybalin
        Khalid Hossain (ALCF)

    Iterate over vllm.model_executor.models.* modules and pre-generate
    vLLM model-info caches using internal registry APIs.

    This avoids the model-inspection subprocess codepath, which can lead to segfaults.

    Safe behavior:
    - Skips modules that fail to import
    - Skips classes that fail inspection
    - Continues on errors
    - Uses vLLM's own cache naming + serialization
    """

    def is_candidate_model_class(cls: Type) -> bool:
        """
        Heuristic: vLLM model classes usually
        - are classes
        - defined in this module
        - have 'ForCausalLM', 'Model', 'LM', or 'Vision' in name
        """
        if not inspect.isclass(cls):
            return False
        name = cls.__name__
        return any(
            key in name
            for key in (
                "ForCausalLM",
                "Model",
                "LM",
                "Vision",
                "VL",
            )
        )

    def iter_model_classes(module: ModuleType) -> Iterable[Type]:
        for _, obj in vars(module).items():
            if is_candidate_model_class(obj):
                yield obj

    # Import vLLM registry internals
    try:
        import vllm.model_executor.models as models_pkg
        import vllm.model_executor.models.registry as r
    except Exception as e:
        raise e

    Lazy = getattr(r, "_LazyRegisteredModel", None)
    ModelInfo = getattr(r, "_ModelInfo", None)
    if Lazy is None or ModelInfo is None:
        raise ModuleNotFoundError("vllm resitry is not available")

    cache_root = os.environ.get("VLLM_CACHE_ROOT")
    if not cache_root:
        cache_root = f"/tmp/vllm_cache_{uuid.uuid4().hex[:6]}"
        os.environ["VLLM_CACHE_ROOT"] = cache_root
        os.makedirs(cache_root, exist_ok=True)
        if logger is not None:
            logger.info(f"VLLM_CACHE_ROOT not set; created {cache_root}")

    modelinfo_dir = os.path.join(cache_root, "modelinfo")
    if os.path.isdir(modelinfo_dir):
        if logger is not None:
            logger.info(
                f"Found existing modelinfo cache at {modelinfo_dir}; "
                "skipping build"
            )
        return 0

    if logger is not None:
        logger.info(f"Using VLLM_CACHE_ROOT={cache_root}")

    success = 0
    skipped = 0
    failed = 0

    for modinfo in pkgutil.iter_modules(models_pkg.__path__):
        modname = modinfo.name
        fqmod = f"{models_pkg.__name__}.{modname}"

        try:
            module = importlib.import_module(fqmod)
        except Exception as e:
            skipped += 1
            if logger is not None:
                logger.info(f"[SKIP] {fqmod}: import failed ({e.__class__.__name__})")
            continue

        module_file = getattr(module, "__file__", None)
        if not module_file:
            if logger is not None:
                logger.info("module_file is None")
            skipped += 1
            continue

        try:
            module_bytes = Path(module_file).read_bytes()
            module_hash = r.safe_hash(module_bytes, usedforsecurity=False).hexdigest()
        except Exception as e:
            if logger is not None:
                logger.warning(f"module_hash failed with Exception {e}")
            skipped += 1
            continue

        for cls in iter_model_classes(module):
            cls_name = cls.__name__

            try:
                lazy = Lazy(module_name=fqmod, class_name=cls_name)
                model_cls = lazy.load_model_cls()
                mi = ModelInfo.from_model_cls(model_cls)
                lazy._save_modelinfo_to_cache(mi, module_hash)

                cache_path = lazy._get_cache_dir() / lazy._get_cache_filename()
                success += 1
                if logger is not None:
                    logger.info(f"[OK] {fqmod}:{cls_name}")
                    logger.info(f"     -> {cache_path}")

            except Exception as e:
                failed += 1
                if logger is not None:
                    logger.warning(
                        f"[FAIL] {fqmod}:{cls_name} ({e.__class__.__name__})"
                    )
                continue

    if logger is not None:
        logger.info("\nSummary:")
        logger.info(f"  success: {success}")
        logger.info(f"  failed : {failed}")
        logger.info(f"  skipped: {skipped}")

    return 0


build_model_cache = _build_model_cache


def find_free_port(
    port_range: Tuple[int, int], host: str = "127.0.0.1"
) -> Optional[int]:
    """
    Attempts to find a free port within the given range by binding to it.
    Checks ports in a random order to reduce collisions between concurrent startups.
    """
    # Create a list of all ports in the range and shuffle them
    ports_to_check = list(range(port_range[0], port_range[1]))
    random.shuffle(ports_to_check)

    for port in ports_to_check:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue

    return None


def call_llm(
    model: str,
    model_path: str,
    prompts: List,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
):
    start = perf_counter()

    default_llm_kwargs = {
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "enforce_eager": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 1,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 1,
    }
    default_sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    llm_kwargs = {**default_llm_kwargs, **(llm_kwargs or {})}
    sampling_kwargs = {**default_sampling_kwargs, **(sampling_kwargs or {})}

    _actor_port = 10000 + (os.getpid() % 100) * 200
    _actor_port = find_free_port((_actor_port, _actor_port + 200), "localhost")
    os.environ["HF_HOME"] = model_path
    os.environ["VLLM_PORT"] = str(_actor_port) if _actor_port is not None else "0"
    os.environ["VLLM_HOST_IP"] = "localhost"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_actor_port) if _actor_port is not None else "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(llm_kwargs["tensor_parallel_size"])
    tmp_dir = f"/tmp/vllm_cache_{uuid.uuid4().hex[:6]}"
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    build_model_cache()

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(**sampling_kwargs)

    tic = perf_counter()
    llm = LLM(model=model, **llm_kwargs)
    init_time = perf_counter() - tic

    tic = perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    inf_time = perf_counter() - tic

    tot_time = perf_counter() - start

    return {
        "responses": outputs,
        "total_time": tot_time, 
        "initialization_time": init_time,
        "inference_time": inf_time
    }
