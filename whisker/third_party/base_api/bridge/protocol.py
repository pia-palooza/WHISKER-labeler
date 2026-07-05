import dataclasses
import enum
import logging
import json
from typing import TextIO

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50000

class Message:
    """Base class for all messages exchanged between client and server."""
    pass

@dataclasses.dataclass
class Request(Message):
    """Base class for all requests sent from client to server."""
    pass

@dataclasses.dataclass
class Response(Message):
    """Base class for all responses sent from server to client."""
    pass

#########################################################################

@dataclasses.dataclass
class ShutdownRequest(Request):
    """Request to shut down the server."""
    pass

@dataclasses.dataclass
class ShutdownAcknowledgedResponse(Response):
    """Signals acknowledgment of a shutdown request."""
    pass

#########################################################################

@dataclasses.dataclass
class ErrorResponse(Response):
    """Response indicating an error occurred while processing a request."""

    # Error message describing the issue.
    error_message: str

@dataclasses.dataclass
class TaskInitiatedResponse(Response):
    """Response indicating an asynchronous task was initiated."""

    # A unique identifier for the initiated task.
    task_id: str
    # Additional message or information about the task initiation.
    message: str

@dataclasses.dataclass
class TaskCompletedResponse(Response):
    """Response indicating an asynchronous task has completed."""

    # A unique identifier for the completed task.
    task_id: str
    # Result or output of the completed task.
    result: str

#########################################################################

class TaskStatus(str, enum.Enum):
    """Enumeration of possible task statuses."""

    # Request was received but the task is not running yet.
    PENDING = "PENDING"
    # The task is currently running.
    RUNNING = "RUNNING"
    # The task has completed successfully.
    COMPLETED = "COMPLETED"
    # The task has failed.
    FAILED = "FAILED"
    # The status of the task is unknown.
    UNKNOWN = "UNKNOWN"

@dataclasses.dataclass
class QueryTaskStatusRequest(Request):
    """Queries the status of a previously issued task."""

    # Identifier string for the task to query.
    task_id: str

@dataclasses.dataclass
class QueryTaskStatusResponse(Response):
    """Response containing the status of a queried task."""

    # Whether the task is still running.
    status: TaskStatus
    # Additional message or information about the task status.
    message: str

#########################################################################

@dataclasses.dataclass
class EchoRequest(Request):
    """A simple command to test connectivity."""
    # Message to be echoed back by the server.
    message: str
    # Delay in seconds before responding.
    delay: float = 0.0

@dataclasses.dataclass
class EchoResponse(Response):
    """Response for the EchoCommand."""

    # The echoed message from the server.
    echoed_message: str

#########################################################################

BASE_MESSAGE_TYPE_MAP: dict[str, type[Message]] = {
    "Response": Response,
    "ShutdownRequest": ShutdownRequest,
    "ShutdownAcknowledgedResponse": ShutdownAcknowledgedResponse,
    "ErrorResponse": ErrorResponse,
    "TaskInitiatedResponse": TaskInitiatedResponse,
    "TaskCompletedResponse": TaskCompletedResponse,
    "QueryTaskStatusRequest": QueryTaskStatusRequest,
    "QueryTaskStatusResponse": QueryTaskStatusResponse,
    "EchoRequest": EchoRequest,
    "EchoResponse": EchoResponse,
}

#########################################################################

def _instantiate_message_from_dict(
    message_dict: dict,
    message_type_map: dict[str, type[Message]]
) -> Message:
    """
    (Internal) Instantiates a dataclass from a dict with 'type' and 'payload'.
    """
    message_type = message_dict.get("type")
    payload = message_dict.get("payload")

    if not isinstance(message_type, str) or not isinstance(payload, dict):
        raise ValueError(
            "Serialized message must contain 'type' (str) and 'payload' (dict)"
        )

    message_class = message_type_map.get(message_type)
    if not message_class:
        raise ValueError(
            f"Unknown message type: {message_type}"
        )

    # Instantiate the dataclass using the payload as keyword arguments
    try:
        return message_class(**payload)
    except TypeError as e:
        raise ValueError(
            f"Payload mismatch for type '{message_type}': {e}"
        ) from e

def serialize_message(
    message: Message,
    wfile: TextIO
):
    """
    Serializes, JSON-encodes, and sends a Message object to a wfile stream.
    """
    if not dataclasses.is_dataclass(message) or not isinstance(message, Message):
         raise TypeError(f"Object to send must be a Message dataclass: {type(message).__name__}")
         
    message_json = {
        "type": type(message).__name__,
        "payload": dataclasses.asdict(message)
    }
    json.dump(message_json, wfile)
    # Ensure a clean termination of the stream for json.load on the other end
    wfile.write('\n')
    wfile.flush()
    logging.debug(f"Message serialized and flushed: {type(message).__name__}")


def deserialize_message(
    rfile: TextIO,
    message_type_map: dict[str, type[Message]]
) -> Message:
    """
    Waits for, JSON-decodes, and deserializes a Message object from an rfile stream.
    """
    # This blocks until a full JSON object is received,
    # (e.g., terminated by newline) AND the client signals EOF (SHUT_WR).
    request_json = json.load(rfile)
    logging.debug(f"Received raw message dict.")
    
    message = _instantiate_message_from_dict(request_json, message_type_map)
    if not isinstance(message, Message):
        # This should be caught by _instantiate_message_from_dict, but as a safeguard:
        raise TypeError(f"Deserialized object is not a Message: {type(message).__name__}")

    return message
