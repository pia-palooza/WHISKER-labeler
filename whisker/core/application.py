import logging

from ..base.core_nodes.callback_observer import CallbackObserverNode
from ..base.core_nodes.cli import CliNode
from ..base.core_nodes.cluster import ClusterNode
from ..base.core_nodes.service_management import ServiceManagementNode
from ..base.core_nodes.queue_router import QueueRouterNode
from ..base.core_nodes.socket_router import SocketRouterNode
from ..base.logger import LogSocketReceiver, LoggerConfig, ConsoleConfig, FileConfig, configure_loggers
from ..base.task import NodeTask

DEMO_PORT = 9876

def run_main_logger() -> LogSocketReceiver:
    # 1. Bind the main logger
    main_config = LoggerConfig(
        console=ConsoleConfig(level=logging.INFO), 
    )
    configure_loggers(main_config)
    main_logger = logging.getLogger("MainApplication")

    # 2. Boot the server and pass our main application context explicitly
    receiver = LogSocketReceiver(host="127.0.0.1", port=DEMO_PORT, target_logger=main_logger)
    receiver.start()
    main_logger.info("Main application local logging system initialized.")

    return receiver

class MainApplication:
    def __init__(self, label: str, subtasks: list[NodeTask] | None = None):
        self.receiver = run_main_logger()
        self.cli = NodeTask(CliNode(label=f"{label}.cli"), refresh_rate=0.1)
        self.cli_handler = NodeTask(CallbackObserverNode(f"{label}.cli_handler"), refresh_rate=0.1)
        self.bus = NodeTask(QueueRouterNode(label=f"{label}.queue_router"), refresh_rate=0.1)
        self.socket_router = NodeTask(SocketRouterNode(label=f"{label}.socket"), refresh_rate=0.1)
        self.service_manager = NodeTask(ServiceManagementNode(label=f"{label}.service_manager"), refresh_rate=0.1)
        self.cluster = ClusterNode(
            label=f"{label}.cluster",
            bus_task=self.bus,
            subtasks=[self.socket_router, self.service_manager, self.cli] + (subtasks or [])
        )

    def run(self):
        self.cluster.setup()

    def shutdown(self):
        self.cluster.shutdown()
        self.receiver.stop()
