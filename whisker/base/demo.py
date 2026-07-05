import argparse
import dataclasses
import logging
import os
import time
from enum import Enum

from .application import run_application_frontend, run_application_backend
from .core_nodes.cli import CliNode
from .core_nodes.cluster import ClusterNode
from .core_nodes.service_management import ServiceManagementNode
from .core_nodes.queue_router import QueueRouterNode
from .core_nodes.socket_router import SocketRouterNode
from .messaging import Message, register_topic
from .logger import LogSocketReceiver, LoggerConfig, ConsoleConfig, FileConfig, SocketConfig, configure_loggers
from .node import Node
from .task import NodeTask
from . import topics as topics

DEMO_PORT = 9876
DEMO_MAIN_LOG_FILE = "demo_main_output.log"
DEMO_CHILD_LOG_FILE = "demo_child_output.log"

def run_main_logger() -> LogSocketReceiver:
    # 1. Bind the main logger
    main_config = LoggerConfig(
        console=ConsoleConfig(level=logging.INFO), 
        file=FileConfig(file_path=DEMO_MAIN_LOG_FILE, level=logging.DEBUG, as_json=True)
    )
    configure_loggers(main_config)
    main_logger = logging.getLogger("MainApplication")

    # 2. Boot the server and pass our main application context explicitly
    receiver = LogSocketReceiver(host="127.0.0.1", port=DEMO_PORT, target_logger=main_logger)
    receiver.start()
    main_logger.info("Main application local logging system initialized.")

    return receiver

def run_child_logger():
    child_config = LoggerConfig(
        socket=SocketConfig(host="127.0.0.1", port=DEMO_PORT, level=logging.INFO),
        file=FileConfig(file_path=DEMO_CHILD_LOG_FILE, level=logging.DEBUG, as_json=True)
    )
    configure_loggers(child_config)

    child_logger = logging.getLogger("ChildEnvironment.Service")
    child_logger.info("Child application logging system initialized and connected to main logger.")
    return child_logger

class DemoTopic(str, Enum):
    TEMPERATURE_FAHRENHEIT = "telemetry/temperature/fahrenheit"
    TEMPERATURE_CELSIUS = "telemetry/temperature/celsius"

@register_topic(DemoTopic.TEMPERATURE_FAHRENHEIT)
@dataclasses.dataclass(slots=True)
class TemperatureFahrenheitPayload:
    fahrenheit: float

class MonitorTopic(str, Enum):
    THRESHOLD_EXCEEDED = "monitor/alerts/threshold_exceeded"

@register_topic(DemoTopic.TEMPERATURE_CELSIUS)
@dataclasses.dataclass(slots=True)
class TemperatureCelsiusPayload:
    celsius: float

@register_topic(MonitorTopic.THRESHOLD_EXCEEDED)
@dataclasses.dataclass(slots=True)
class ThresholdAlertPayload:
    name: str
    reading: float
    threshold: float

class SensorNode(Node):
    def __init__(self, label: str):
        super().__init__(label)
        self._temperature = 10.0

    def wakeup(self) -> None:
        """The internal loop executed by the worker thread."""        
        super().wakeup()
        self.send_outgoing_message(TemperatureCelsiusPayload(celsius=self._temperature))
        self._temperature += 0.1

class ThresholdAlertNode(Node):
    def __init__(self, label: str, heartbeat_interval: float = 5.0, threshold: float = 30.0):
        super().__init__(
            label,
            subscriptions=set([DemoTopic.TEMPERATURE_FAHRENHEIT])
        )
        self._threshold = threshold

    def handle_message(self, message: Message) -> bool:
        if  super().handle_message(message):
            return True
        if message.header.topic == DemoTopic.TEMPERATURE_FAHRENHEIT:
            if message.payload.fahrenheit > self._threshold:
                self.send_outgoing_message(
                    ThresholdAlertPayload(
                        name=DemoTopic.TEMPERATURE_FAHRENHEIT.value,
                        reading=message.payload.fahrenheit,
                        threshold=self._threshold
                    )
                )
            return True
        return False

class TempConversionNode(Node):
    def __init__(self, label: str):
        super().__init__(
            label,
            subscriptions=set([DemoTopic.TEMPERATURE_CELSIUS])
        )
    
    def handle_message(self, message: Message) -> bool:
        if  super().handle_message(message):
            return True
        if message.header.topic == DemoTopic.TEMPERATURE_CELSIUS:
            fahrenheit = message.payload.celsius * 9.0 / 5.0 + 32.0
            self.send_outgoing_reply(
                message.header.message_id,
                TemperatureFahrenheitPayload(fahrenheit=fahrenheit)
            )
            return True
        return False

class ServerApplication:
    def __init__(self, label: str, subtasks: list[NodeTask] | None = None):
        self.receiver = run_main_logger()
        self.socket_router = NodeTask(SocketRouterNode(label=f"{label}.socket"), refresh_rate=0.1)
        self.sensor = NodeTask(SensorNode(label=f"{label}.sensor"), refresh_rate=1.0)
        self.threshold = NodeTask(ThresholdAlertNode(label=f"{label}.threshold_alert", threshold=51.0), refresh_rate=0.1)

        self.cluster = ClusterNode(
            label=f"{label}.cluster",
            bus_task=NodeTask(QueueRouterNode(label=f"{label}.queue_router"), refresh_rate=0.1),
            subtasks=[self.socket_router, self.sensor, self.threshold] + (subtasks or [])
        )
    
    def run(self, port: int):
        self.cluster.setup()
        self.cluster.send_outgoing_message(
            topics.socket_router.BindRequest(address=f"127.0.0.1:{port}"),
            target_node_id=self.socket_router.node.uuid
        )
        self.cluster.send_outgoing_message(
            topics.socket_router.StartForwardingRequest(patterns=["node/*", DemoTopic.TEMPERATURE_CELSIUS.value]),
            target_node_id=self.socket_router.node.uuid
        )

    def shutdown(self):
        self.cluster.shutdown()
        self.receiver.stop()

class ManagedServerApplication(ServerApplication):
    def __init__(self, label: str):
        self.service_manager = NodeTask(ServiceManagementNode(label=f"{label}.service_manager"), refresh_rate=0.1)
        self.cli = NodeTask(CliNode(label=f"{label}.cli"), refresh_rate=0.1)
        super().__init__(label, subtasks=[self.service_manager, self.cli])
    
    def run(self, port: int):
        super().run(port)
        self.cluster.send_outgoing_message(
            topics.service_management.StartServiceRequest(
                config=topics.service_management.ServiceConfig(
                    module_name="whisker.base.demo",
                    environment_name="whisker",
                    args=["--client", "--managed", "--port", str(port), "--server-node-id", self.service_manager.node.uuid],
                    startup_timeout=5.0,
                    heartbeat_timeout=10.0
                )
            ),
            target_node_id=self.service_manager.node.uuid
        )
        time.sleep(1.0)
        self.cluster.send_outgoing_message(
            topics.cli.ToggleEnableRequest(enable=True),
            target_node_id=self.cli.node.uuid
        )
    
class ClientApplication:
    def __init__(self, label: str):
        self.logger = run_child_logger()
        self.socket_router_node = SocketRouterNode(label=f"{label}.socket")
        self.temp_conversion_node = TempConversionNode(label=f"{label}.temp_converter")

        self.cluster = ClusterNode(
            label=f"{label}.cluster",
            bus_task=NodeTask(QueueRouterNode(label=f"{label}.queue_router"), refresh_rate=0.1),
            subtasks=[
                NodeTask(self.socket_router_node, refresh_rate=0.1),
                NodeTask(self.temp_conversion_node, refresh_rate=0.1)
            ]
        )

    def run(self, port: int, server_node_id: str | None = None):
        self.cluster.setup()
        self.cluster.send_outgoing_message(
            topics.socket_router.ConnectRequest(address=f"127.0.0.1:{port}"),
            target_node_id=self.socket_router_node.uuid
        )
        forwarding_patterns = ["node/*", DemoTopic.TEMPERATURE_FAHRENHEIT.value]
        if server_node_id:
            forwarding_patterns.append(topics.service_management.Request.ACCEPT_SERVICE_CLIENT_CONNECTION)

        self.cluster.send_outgoing_message(
            topics.socket_router.StartForwardingRequest(patterns=forwarding_patterns),
            target_node_id=self.socket_router_node.uuid
        )

        if server_node_id:
            time.sleep(1.0)
            self.cluster.send_outgoing_message(
                topics.service_management.AcceptServiceClientConnectionRequest(
                    module_name="whisker.base.demo",
                    pid=os.getpid(),
                    uuid=self.cluster.uuid
                ),
                target_node_id=server_node_id
            )

    def shutdown(self):
        self.cluster.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WHISKER Messaging Framework Demo")
    parser.add_argument("--server", action="store_true", help="Run the server-side infrastructure")
    parser.add_argument("--client", action="store_true", help="Run the client-side infrastructure")
    parser.add_argument("--standalone", action="store_true", help="Run the legacy single-process demo")
    parser.add_argument("--managed", action="store_true", help="Connect back to manager admin bridge")
    parser.add_argument("--port", type=int, default=50050, help="Port for socket bridge communication (default: 50050)")
    parser.add_argument("--server-node-id", type=str, default=None, help="UUID of the server node to connect to (for managed clients)")

    args = parser.parse_args()

    if args.server:
        cluster = ServerApplication("server")
        cluster.run(args.port)
        time.sleep(10.0)
        cluster.shutdown()
    elif args.client:
        cluster = ClientApplication("client")
        cluster.run(args.port, args.server_node_id)
        time.sleep(20.0)
        cluster.shutdown()
    elif args.standalone:
        server_cluster = ServerApplication("server")
        client_cluster = ClientApplication("client")
        server_cluster.run(args.port)
        client_cluster.run(args.port)
        time.sleep(10.0)
        client_cluster.shutdown()
        server_cluster.shutdown()
    else:
        cluster = ManagedServerApplication("mg_server")
        cluster.run(args.port)
        time.sleep(10.0)
        cluster.shutdown()

    print("Demo execution complete")