from multiprocessing.queues import Queue as MPQueue
from typing import Dict, Any, Type, Tuple
import logging
from .protocol import QueueProtocol
logger = logging.getLogger(__name__)
try:
    from dragon.native.queue import Queue as DragonQueue
    DRAGON_AVAILABLE = True
except Exception as e:
    logger.warning(f"Dragon is not available")
    DRAGON_AVAILABLE = False


class QueueRegistry:
    def __init__(self):
        self._available_queues: Dict[str, Type[Any]] = {}
    
    def register(self, name:str):
        def decorator(cls: Type[Any]):
            self._available_queues[name] = cls
            return cls
        
        return decorator
    
    def create_queue(self,name:str, args:Tuple=(),kwargs:Dict={}) -> QueueProtocol:
        try:
            return self._available_queues[name](*args,**kwargs)
        except KeyError:
            logger.error(f"{name} Queue is not availabled. Available queues {self._available_queues.keys()}")
            raise

queue_registry = QueueRegistry()

queue_registry.register("multiprocessing")(MPQueue)

if DRAGON_AVAILABLE:
    queue_registry.register("dragon")(DragonQueue)