import os

from ensemble_launcher.ensemble.actor import PublicActor
from ensemble_launcher.logging import setup_logger


class OpenAIInference(PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        transport: str = "zmq",
    ):
        super().__init__(name, transport)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._openai_client = None

    def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(
                name=self._name, log_dir=f"{os.getcwd()}/logs"
            )
        if self._openai_client is None:
            from openai import OpenAI

            try:
                del os.environ["http_proxy"]
                del os.environ["https_proxy"]
                del os.environ["HTTP_PROXY"]
                del os.environ["HTTPS_PROXY"]
            except KeyError:
                pass
            os.environ["no_proxy"] = "localhost,127.0.0.1"

            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.logger.info(f"OpenAI client connected to {self.base_url}")

    def action(self, prompts="hello", temperature=0.0, max_tokens=1024):
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        results = []
        for prompt in prompts:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            results.append(response.choices[0].message.content)

        return results[0] if single else results
