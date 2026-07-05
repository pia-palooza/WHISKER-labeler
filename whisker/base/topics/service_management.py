import dataclasses
from enum import Enum
import subprocess
from typing import Optional, List
from ..messaging import register_topic

class Request(str, Enum):
    START_SERVICE = "service_manager/request/start_service"
    GET_SERVICES_LIST = "service_manager/request/get_services_list"
    GET_SERVICES_INFO = "service_manager/request/get_services_info"
    ACCEPT_SERVICE_CLIENT_CONNECTION = "service_manager/request/accept_service_client_connection"

class Reply(str, Enum):
    SERVICE_LAUNCHED = "service_manager/reply/service/launched"
    SERVICES_LIST = "service_manager/reply/service/list"
    SERVICES_INFO = "service_manager/reply/service/info"
    SERVICE_CLIENT_CONNECTION_ACCEPTED = "service_manager/reply/service_client_connection_accepted"

@dataclasses.dataclass(slots=True)
class ServiceConfig:
    module_name: str
    executable_path: Optional[str] = None
    environment_name: Optional[str] = None
    args: List[str] = dataclasses.field(default_factory=list)
    startup_timeout: float = 10.0
    heartbeat_timeout: float = 5.0

@dataclasses.dataclass(slots=True)
class ManagedServiceState:
    config: ServiceConfig
    process: subprocess.Popen
    uuid: Optional[str] = None
    last_heartbeat: float = 0.0
    is_shutting_down: bool = False

@register_topic(Request.GET_SERVICES_LIST)
@dataclasses.dataclass(slots=True)
class GetServicesListRequest:
    pass # DEV_NOTE: Empty dataclass acts as a strict typing token for the request

@register_topic(Request.GET_SERVICES_INFO)
@dataclasses.dataclass(slots=True)
class GetServicesInfoRequest:
    service_names: list[str]

@register_topic(Request.START_SERVICE)
@dataclasses.dataclass(slots=True)
class StartServiceRequest:
    config: ServiceConfig

@register_topic(Request.ACCEPT_SERVICE_CLIENT_CONNECTION)
@dataclasses.dataclass(slots=True)
class AcceptServiceClientConnectionRequest:
    module_name: str
    pid: int
    uuid: Optional[str] = None

@register_topic(Reply.SERVICE_LAUNCHED)
@dataclasses.dataclass(slots=True)
class ServiceLaunchedReply:
    module_name: str
    pid: int

@register_topic(Reply.SERVICES_LIST)
@dataclasses.dataclass(slots=True)
class ServicesListReply:
    services: List[str]

@register_topic(Reply.SERVICE_CLIENT_CONNECTION_ACCEPTED)
@dataclasses.dataclass(slots=True)
class ServiceClientConnectionAcceptedReply:
    module_name: str
    pid: int
    uuid: Optional[str] = None
