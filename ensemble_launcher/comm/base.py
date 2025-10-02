from abc import ABC, abstractmethod


class Comm(ABC):

    @abstractmethod
    def send_to_parent(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def recv_from_parent(self, *args, **kwargs):
        pass

    @abstractmethod
    def send_to_child(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def recv_from_child(self, *args, **kwargs):
        pass

    @abstractmethod
    def blocking_recv_from_parent(self, *args, **kwargs):
        """
        Blocking receive from a specific parent. Waits indefinitely until a message is available.
        """
        pass

    @abstractmethod
    def blocking_recv_from_child(self, *args, **kwargs):
        """
        Blocking receive from a specific child. Waits indefinitely until a message is available.
        """
        pass
    
    def send_to_parents(self, data) -> int:
        for parent_id, pipe in self.parents.items():
            self.send_to_parent(parent_id, data)
            if self.logger:
                self.logger.debug(f"Sent message to parent {parent_id}")
        return 0

    def recv_from_parents(self, timeout: int = 60) -> list:
        messages = []
        for parent_id in self.parents.keys():
            msg = self.recv_from_parent(parent_id, timeout)
            if msg is not None:
                messages.append(msg)
        return messages

    def send_to_children(self, data) -> int:
        for child_id in self.children.keys():
            self.send_to_child(child_id, data)
        return 0

    def recv_from_children(self, timeout: int = 60) -> list:
        messages = []
        for child_id in self.children.keys():
            msg = self.recv_from_child(child_id, timeout)
            if msg is not None:
                messages.append(msg)
        return messages