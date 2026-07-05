import copy
import json
import socket
import time
import logging

from . import protocol

DEFAULT_MAX_RETRIES = 12
DEFAULT_RETRY_DELAY_SEC = 5

class BaseClient:
    """Base client class for communication with a bridge Server."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: int = DEFAULT_RETRY_DELAY_SEC,
        additional_messages: dict[str, type] | None = None,
    ):
        self.host = host if host is not None else protocol.DEFAULT_HOST
        self.port = port if port is not None else protocol.DEFAULT_PORT
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(f"{__name__}.Client({self.port})")
        self.message_type_map = copy.deepcopy(protocol.BASE_MESSAGE_TYPE_MAP)
        if additional_messages:
            self.message_type_map.update(additional_messages)

    def _establish_connection(self) -> socket.socket:
        conn = None
        for attempt in range(self.max_retries):
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((self.host, self.port))
                self.logger.debug(f"Connection established to {self.host}:{self.port}")
                break  # Success!
            except ConnectionRefusedError:
                self.logger.warning(
                    f"Connection refused to {self.host}:{self.port}. "
                    "Server might be down or starting. "
                    f"Retrying ({attempt + 1}/{self.max_retries})..."
                )
                if attempt + 1 == self.max_retries:
                    self.logger.error("Max retries reached.")
                    raise ConnectionError(
                        f"Could not connect to server at {self.host}:{self.port}."
                    )
                time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unhandled connection error: {e}", exc_info=True)
                raise ConnectionError(f"Unhandled connection error: {e}")

        if not conn:
            raise ConnectionError("Failed to establish connection after retries.")
        
        return conn
    
    def _serialize_request(self, wfile, request: protocol.Request):
        try:
            self.logger.debug(f"Sending request: {type(request).__name__}")
            protocol.serialize_message(request, wfile)
            self.logger.debug("Object serialized, written to buffer, and flushed.")
        except Exception as dump_err:
            self.logger.error(f"Error during serialization: {dump_err}", exc_info=True)
            raise ConnectionError("Failed to send command data") from dump_err

    def _deserialize_response(self, rfile) -> protocol.Response:
        self.logger.debug("Awaiting response...")
        
        # DEV_NOTE: Use the common function from protocol.py
        response = protocol.deserialize_message(rfile, self.message_type_map)

        if not isinstance(response, protocol.Response):
            self.logger.error(f"Received object is not a Response: {type(response).__name__}")
            raise ValueError("Received object is not a Response instance")

        return response

    def send_request(self, request: protocol.Request) -> protocol.Response:
        """Connects, sends a single Request object, and gets a Response."""
        self.logger.debug(f"Attempting to send object of type: {type(request).__name__}")
        conn = self._establish_connection()
        
        try:
            # DEV_NOTE: This robust makefile pattern remains unchanged
            with conn.makefile("w", encoding="utf-8") as wfile, conn.makefile("r", encoding="utf-8") as rfile:
                self._serialize_request(wfile, request)
                
                # CRITICAL: Signal EOF to the server's rfile (json.load)
                conn.shutdown(socket.SHUT_WR)
                self.logger.debug("Write half of connection shut down (SHUT_WR).")
                
                # Wait for the server's response
                response = self._deserialize_response(rfile)
        except (EOFError, json.JSONDecodeError, ValueError) as e:
            # This might happen if the server closes connection unexpectedly during response read
            self.logger.error(
                f"Server connection error or bad data during response read: {e}",
                exc_info=True
            )
            raise ConnectionError(f"Server connection error during response: {e}")
        except ConnectionError:
            raise
        except Exception as e:
            # Catch other unexpected errors during communication
            self.logger.error(f"Error during communication: {e}", exc_info=True)
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as close_err:
                    self.logger.warning(f"Error closing connection: {close_err}")
            self.logger.debug("Connection closed.")
        
        return response

    def wait_for_task_to_complete(
        self,
        task_id: str,
        poll_interval_sec: float = 0.5
    ) -> protocol.Response:
        """Polls the server for task completion using the given task_id."""
        from .protocol import QueryTaskStatusRequest, QueryTaskStatusResponse, TaskStatus

        self.logger.info(f"Polling for task completion, ID: {task_id}")
        while True:
            time.sleep(poll_interval_sec)
            query_request = QueryTaskStatusRequest(task_id=task_id)
            response = self.send_request(query_request)

            if not isinstance(response, QueryTaskStatusResponse):
                self.logger.error(
                    f"Unexpected response type during task polling: {type(response).__name__}"
                )
                raise ConnectionError("Invalid response during task status query.")

            status = response.status
            self.logger.info(f"Task Status: {status}. Message: {response.message}")

            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.UNKNOWN):
                self.logger.info(f"Task {task_id} resolved with status: {status}")
                return response