from enum import Enum, auto


class ChildState(Enum):
    NOTREADY   = auto()  # registered, no resources yet
    READY      = auto()  # resources allocated, ready to launch
    RUNNING    = auto()  # subprocess submitted
    RECOVERING = auto()  # failed, being relaunched; resources NOT freed
    FAILED     = auto()  # terminal failure; resources freed
    SUCCESS    = auto()  # terminal success; resources freed
