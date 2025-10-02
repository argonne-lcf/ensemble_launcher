from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, List, Tuple, Union
from ensemble_launcher.scheduler.resource import JobResource

class Executor(ABC):
    @abstractmethod
    def start(
        self,
        job_resource: JobResource,
        task: Union[str, Callable],
        task_args: Tuple = (),
        task_kwargs: Dict[str, Any] = None,
        env: Dict[str, Any] = None,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def stop(self, task_id: str, force: bool = False, **kwargs) -> bool:
        """Stop a running task."""
        pass

    @abstractmethod
    def wait(self, task_id: str, timeout: float = None, **kwargs) -> bool:
        """Wait for a task to complete."""
        pass

    @abstractmethod
    def result(self, task_id: str, timeout: float = None, **kwargs):
        """Retrieve result of a completed task."""
        pass

    @abstractmethod
    def exception(self, task_id: str, **kwargs):
        """Retrieve exception raised by a task, if any."""
        pass

    @abstractmethod
    def done(self, task_id: str, **kwargs) -> bool:
        """Check if a task is done."""
        pass

    @abstractmethod
    def shutdown(self, force: bool = False, **kwargs):
        """Shutdown executor and cleanup resources."""
        pass