from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, List, Tuple

class Executor(ABC):
    @abstractmethod
    def start(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def stop(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def wait(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def result(self, *args, **kwargs):
        pass

    @abstractmethod
    def exception(self, *args, **kwargs):
        pass

    @abstractmethod
    def done(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def shutdown(self, *args, **kwarg):
        pass