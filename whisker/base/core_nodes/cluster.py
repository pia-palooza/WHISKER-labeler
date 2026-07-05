import dataclasses
import time
from typing import Any, cast, Sequence

from ...base import topics
from ..node import Node
from ..task import NodeTask
from .queue_router import QueueRouterNode

class ClusterNode(Node):
    def __init__(
        self,
        label: str,
        bus_task: NodeTask,
        subtasks: Sequence[NodeTask],
        subscriptions: set[str] | None = None
    ):
        super().__init__(label=label, subscriptions=subscriptions)
        self._bus_task = bus_task
        self._subtasks: dict[str, NodeTask] = { t.node.uuid : t for t in subtasks}

    @property
    def bus(self) -> QueueRouterNode:
        return cast(QueueRouterNode, self._bus_task.node)

    def setup(self):
        super().setup()

        for task in self._subtasks.values():
            self.bus.register_node(task.node)
        self.bus.register_node(self)
        self._bus_task.run()

        for task in self._subtasks.values():
            task.run()

    def shutdown(self):
        for task in reversed(self._subtasks.values()):
            if task.node.received_shutdown:
                self.logger.info(f"Node {task.node.uuid} has already received a shutdown signal.")
            else:
                self.logger.info(f"Sending SHUTDOWN signal to {task.node.uuid}")
                self.send_outgoing_message(
                    topics.node.ShutdownRequest(),
                    target_node_id=task.node.uuid
                )

        try:
            deadline = time.time() + 5.0
            for task in self._subtasks.values():
                timeout = deadline - time.time()
                if timeout > 0.0:
                    self.logger.info(f"Waiting for task {task._uuid} to shutdown..")
                    task.wait_for_shutdown(timeout=timeout)
                else:
                    raise TimeoutError("Application task shut down timed out.")
        except TimeoutError as e:
            self.logger.exception("Application task shut down timed out.")
        
        self.logger.info(f"Sending SHUTDOWN signal to bus task {self.bus.uuid}")
        self.send_outgoing_message(
            topics.node.ShutdownRequest(),
            target_node_id=self.bus.uuid
        )
        try:
            self.logger.info(f"Waiting for bus task {self.bus.uuid} to shutdown..")
            self._bus_task.wait_for_shutdown(timeout=5.0)
        except TimeoutError as e:
            self.logger.info("Application router shut down timed out.")

        super().shutdown()
