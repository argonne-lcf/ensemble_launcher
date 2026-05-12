import os
from glob import glob

from ensemble_launcher.ensemble.actor import PublicActor
from ensemble_launcher.logging import setup_logger


class VLLMInference(PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        tensor_parallel_size: int = 1,
        transport: str = "zmq",
        cache_modelinfo: bool = False,
    ):
        super().__init__(name, transport)
        self.model = model
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self._cache_modelinfo = cache_modelinfo

    def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")
        if self._llm is None:
            if self._cache_modelinfo:
                os.environ["VLLM_CACHE_ROOT"] = self.cache_dir
            from vllm import LLM

            snapshots = glob(
                f"{self.cache_dir}/hub/models--{self.model.replace('/', '--')}/snapshots/*"
            )
            self.logger.info(f"model: {snapshots[0]}")
            try:
                self._llm = LLM(
                    model=snapshots[0],
                    tensor_parallel_size=self.tensor_parallel_size,
                    trust_remote_code=True,
                )
            except Exception as e:
                self.logger.error(f"Starting LLM failed with Exception: {e}")
                raise RuntimeError(str(e))
            self.logger.info("init done!")

    def action(self, prompts="hello", temperature=0.0, max_tokens=1024):
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        outputs = self._llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]

        return results[0] if single else results
