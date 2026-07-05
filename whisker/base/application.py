import time
from .task import Node
from .task import NodeTask
from .core_nodes.queue_router import QueueRouterNode
from .core_nodes.socket_router import SocketRouterNode
from .core_nodes.service_management import ServiceManagementNode
from .core_nodes.callback_observer import CallbackObserverNode
from .core_nodes.cluster import ClusterNode

def run_application_frontend(
    name: str,
    frontend_interface_node: Node,
    user_nodes: list[Node],
    socket_address: str,
) -> NodeTask:
    cluster_task = NodeTask(
        ClusterNode(
            f"{name}.cluster",
            NodeTask(QueueRouterNode(f"{name}.bus")),
            [
                NodeTask(SocketRouterNode(f"{name}.socket")),
                NodeTask(ServiceManagementNode(f"{name}.service_manager")),
                NodeTask(frontend_interface_node),
            ] + [
                NodeTask(node) for node in user_nodes
            ],
        )
    )
    cluster_task.run()
    return cluster_task

def run_application_backend(
    name: str,
    user_nodes: list[Node]
)-> NodeTask:
    cluster_task = NodeTask(
        ClusterNode(
            f"{name}.cluster",
            NodeTask(QueueRouterNode(f"{name}.bus")),
            [
                NodeTask(SocketRouterNode(f"{name}.socket")),
            ] + [
                NodeTask(node) for node in user_nodes
            ],
        )
    )
    cluster_task.run()
    cluster_task.wait_for_shutdown()
    return cluster_task
