import copy
import socket
import time
import threading
import json
import logging
import uuid
from typing import Callable, Any, Tuple

from .protocol import (
    ErrorResponse,
    Response,
    Request,
    EchoRequest,
    EchoResponse,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ShutdownRequest,
    ShutdownAcknowledgedResponse,
    BASE_MESSAGE_TYPE_MAP,
    TaskStatus,
    TaskInitiatedResponse,
    QueryTaskStatusRequest,
    QueryTaskStatusResponse,
    serialize_message,
    deserialize_message,
)

# RequestHandler signature changed:
# No longer passes client_socket, as _handle_client_connection manages I/O.
RequestHandler = Callable[[Request, Tuple[str, int]], Response | None]

class RequestHandlerUnknownMessageTypeError(ValueError):
    pass

class BaseServer:
    """
    Listens for JSON objects and dispatches them to a handler function.
    """

    def __init__(
        self,
        user_request_handler: RequestHandler | None = None,
        host: str | None = None,
        port: int | None = None,
        additional_messages: dict[str, type] | None = None,
    ):
        self.user_request_handler = user_request_handler
        self.host = host if host is not None else DEFAULT_HOST
        self.port = port if port is not None else self.default_port()
        self.logger = logging.getLogger(f"{__name__}.Server({self.port})")
        self._shutdown_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self.message_type_map = copy.deepcopy(BASE_MESSAGE_TYPE_MAP)
        if additional_messages:
            self.message_type_map.update(additional_messages)
        self.logger.info(f'Known messages: {list(self.message_type_map.keys())}')
        
        # Async Task Management
        # Stores task_id -> {'status': TaskStatus, 'result': Response | None, 'message': str}
        self._async_task_table: dict[str, dict[str, Any]] = {}
        self._task_lock = threading.Lock()

    # ... (add_async_task, update_async_task_state, get_async_task_state,
    #      _async_task_runner, _query_task_status_handler are unchanged) ...
    def add_async_task(self, func: Callable[[str], Response | None], message: str = "Task initiated") -> str:
        """
        Registers a new function to be run asynchronously and starts it in a new thread.
        Returns a unique Task ID.
        """
        task_id = str(uuid.uuid4())
        
        with self._task_lock:
            self._async_task_table[task_id] = {
                'status': TaskStatus.PENDING,
                'result': None,
                'message': message,
            }
        
        # Start the runner thread immediately
        thread = threading.Thread(
            target=self._async_task_runner,
            args=(task_id, func),
            daemon=True, # Ensure thread doesn't block shutdown
        )
        thread.start()
        
        self.logger.info(f"Async task {task_id} added and started.")
        
        return task_id
    
    def update_async_task_state(self, task_id: str, status: TaskStatus, 
                                result: Response | None = None, message: str | None = None):
        """Atomically updates the status, result, and message for a given task ID."""
        with self._task_lock:
            if task_id in self._async_task_table:
                if status:
                    self._async_task_table[task_id]['status'] = status
                if result:
                    self._async_task_table[task_id]['result'] = result
                if message is not None:
                    self._async_task_table[task_id]['message'] = message
            else:
                self.logger.warning(f"Attempted to update unknown task ID: {task_id}")

    def get_async_task_state(self, task_id: str) -> dict[str, Any]:
        """Atomically retrieves the state for a given task ID."""
        with self._task_lock:
            return self._async_task_table.get(task_id, {
                'status': TaskStatus.UNKNOWN,
                'result': None,
                'message': f"Task ID {task_id} not found.",
            })

    def _async_task_runner(self, task_id: str, func: Callable[[str], Response]):
        """Executes the actual task function and updates the task table state."""
        self.update_async_task_state(task_id, TaskStatus.RUNNING)
        self.logger.debug(f"Task {task_id} status updated to RUNNING.")
        
        try:
            # Execute the user-provided function
            result = func(task_id)
            
            # Update state on success
            self.update_async_task_state(
                task_id, 
                TaskStatus.COMPLETED, 
                result=result, 
                message=f"Task {task_id} completed successfully."
            )
            self.logger.info(f"Task {task_id} completed.")
        except Exception as e:
            # Update state on failure
            self.update_async_task_state(
                task_id, 
                TaskStatus.FAILED, 
                message=f"Task {task_id} failed: {e!r}"
            )
            self.logger.error(f"Task {task_id} failed with error: {e}", exc_info=True)

    def _query_task_status_handler(self, request: QueryTaskStatusRequest) -> QueryTaskStatusResponse:
        """Handler for QueryTaskStatusRequest."""
        task_state = self.get_async_task_state(request.task_id)
        
        status = task_state['status']
        message = task_state['message']
        
        # If completed, the result (which is a Response object) is returned in the message.
        if status == TaskStatus.COMPLETED and task_state['result']:
            result_type = type(task_state['result']).__name__
            message = f"Completed. Result Type: {result_type}. Message: {message}"
        
        return QueryTaskStatusResponse(
            status=status, 
            message=message
        )

    # DEV_NOTE: _receive_request method is removed, its logic is now
    # inside _handle_client_connection

    def _handle_client_connection(self, client_socket: socket.socket, addr: Tuple[str, int]):
        """
        Manages one client connection, handling I/O and dispatching requests.
        This now mirrors the robust 'with makefile' pattern from the client.
        """
        self.logger.debug(f"Accepted connection from {addr}")
        request: Request | None = None
        response: Response | None = None

        try:
            # Open both wfile and rfile for the duration of the connection
            with client_socket.makefile("w", encoding="utf-8") as wfile, \
                 client_socket.makefile("r", encoding="utf-8") as rfile:

                # 1. Read Request
                # This blocks until client sends data AND calls SHUT_WR
                try:
                    request = deserialize_message(rfile, self.message_type_map)
                    if not isinstance(request, Request):
                        raise TypeError(f"Deserialized object is not a Request: {type(request).__name__}")
                    self.logger.debug(f"Received request from {addr}: {type(request).__name__}")
                
                except (ValueError, TypeError, EOFError, json.JSONDecodeError) as e:
                    self.logger.error(f"Request deserialization failed from {addr}: {e}", exc_info=False)
                    response = ErrorResponse(error_message=f"Bad request format: {e}")
                
                # 2. Process Request (if one was successfully received)
                if request:
                    try:
                        # Dispatch to handler, which now just returns a response
                        response = self._handle_request(request, addr)
                    except Exception as handler_exc:
                        self.logger.error(
                            f"Request handler failed unexpectedly for {type(request)}: {handler_exc}",
                            exc_info=True
                        )
                        response = ErrorResponse(error_message=f"Request handler error: {handler_exc}")

                # 3. Send Response (if one was generated)
                if response:
                    self.logger.debug(f"Sending response to {addr}: {type(response).__name__}")
                    serialize_message(response, wfile)
                else:
                    self.logger.warning(f"No response generated for request from {addr}")
            
            # On 'with' block exit, wfile and rfile are flushed and closed,
            # which closes the socket and signals EOF to the client's rfile.

        except Exception as e:
            # Catch errors from makefile() or other unexpected issues
            if not self._shutdown_event.is_set():
                self.logger.error(f"Unhandled worker error for {addr}: {e}", exc_info=True)
        finally:
            try:
                # This close is redundant if 'with' block completes,
                # but vital if makefile() fails before 'with' starts.
                client_socket.close()
            except Exception as close_err:
                self.logger.warning(f"Error closing client socket for {addr}: {close_err}")
                
            self.logger.debug(f"Connection to {addr} closed.")
        
    def _handle_request(self, request: Request, addr: Tuple[str, int]) -> Response | None:
        """
        Dispatches a request to the correct handler.
        (Refactored: This method now *returns* the response, not sends it)
        """
        response = None
        handled_request = False
        
        # 1. Try user-defined handler first
        if self.user_request_handler:
            try:
                # DEV_NOTE: Signature changed, no longer passes client_socket
                response = self.user_request_handler(request, addr)
                handled_request = response is not None
            except RequestHandlerUnknownMessageTypeError:
                # User handler doesn't know this type, fall through to base
                handled_request = False
            except Exception as user_handler_exc:
                self.logger.error(
                    f"User request handler failed for {type(request)}: {user_handler_exc}",
                    exc_info=True
                )
                return ErrorResponse(error_message=f"User handler error: {user_handler_exc}")

        # 2. Fall back to base handler
        if not handled_request:
            try:
                # DEV_NOTE: Signature changed, no longer passes client_socket
                response = self.base_request_handler(request, addr)
            except Exception as base_handler_exc:
                self.logger.error(
                    f"Base request handler failed for {type(request)}: {base_handler_exc}",
                    exc_info=True
                )
                return ErrorResponse(error_message=f"Base handler error: {base_handler_exc}")

        if response is None:
            # This should be caught by base_request_handler, but as a safeguard
            return ErrorResponse(error_message=f"No handler found for {type(request).__name__}")

        return response

    def start(self):
        """Starts the server listening loop."""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            # After binding, if port was 0, it's now assigned. Update self.port
            self.host, self.port = self._server_socket.getsockname()
            self._server_socket.listen(5)
            self._server_socket.settimeout(1.0) # Timeout to check shutdown flag
            self.logger.info(f"Server listening on {self.host}:{self.port}...")

            while not self._shutdown_event.is_set():
                try:
                    client_socket, addr = self._server_socket.accept()
                    handler_thread = threading.Thread(
                        target=self._handle_client_connection,
                        args=(client_socket, addr),
                        daemon=True,
                    )
                    handler_thread.start()
                except socket.timeout:
                    continue 
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        self.logger.error(f"Server accept loop error: {e}", exc_info=True)

            self.logger.info("Server shutdown initiated.")

        except Exception as e:
            self.logger.critical(f"Server failed to start or bind: {e}", exc_info=True)
        finally:
            if self._server_socket:
                try:
                    self._server_socket.close()
                except Exception as close_err:
                     self.logger.warning(f"Error closing server socket: {close_err}")
            self.logger.info("Server socket closed.")

    def shutdown(self):
        """Signals the server to stop listening and shut down."""
        self.logger.info("Shutdown signal received.")
        self._shutdown_event.set()
        # Break the blocking .accept() call
        if self._server_socket:
            try:
                 # Closing the socket will make .accept() raise an exception
                 self._server_socket.close()
            except Exception: pass

    def _deferred_shutdown(self):
        """Helper to call shutdown after a short delay."""
        # Give the _handle_client_connection thread time to send the response
        time.sleep(0.1) 
        self.shutdown()

    def _async_echo_task(self, task_id: str, request: EchoRequest) -> Response:
        """The actual long-running task to be executed asynchronously."""
        self.logger.info(f"Executing async echo task with delay={request.delay}s...")
        start_time = time.time()
        while time.time() - start_time < request.delay:
            time.sleep(0.1)  # Sleep in small increments
            # Check for shutdown signal to abort long tasks
            if self._shutdown_event.is_set():
                self.logger.warning(f"Aborting task {task_id} due to server shutdown.")
                return ErrorResponse(error_message="Task aborted by server shutdown")

            self.update_async_task_state(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                message=f"Echo task running... {time.time() - start_time:.1f}/{request.delay:.1f}s elapsed."
            )
        return EchoResponse(echoed_message=request.message)

    def echo_handler(self, request: Request) -> Response | None:
        """
        Request handler. Initiates an async task if delay > 0, otherwise handles sync.
        Returns Response for sync completion, or TaskInitiatedResponse if async task is initiated.
        """
        if not isinstance(request, EchoRequest):
            return ErrorResponse(
                error_message=f"Unknown request type in handler: {type(request).__name__}"
            )
        
        if request.delay > 0.0:
            task_id = self.add_async_task(
                lambda task_id: self._async_echo_task(task_id, request),
                message=f"Echoing '{request.message[:10]}...' with {request.delay}s delay."
            )
            return TaskInitiatedResponse(
                task_id=task_id, 
                message=f"Task initiated for echo with delay {request.delay}s. Polling required."
            )

        else:
            self.logger.info("Executing synchronous echo task.")
            return EchoResponse(echoed_message=request.message)

    def base_request_handler(self, request: Request, addr: Tuple[str, int]) -> Response | None:
        """Default request handler that can be overridden or extended."""
        if isinstance(request, EchoRequest):
            return self.echo_handler(request)
        elif isinstance(request, ShutdownRequest):
            return self._shutdown_request_handler(request, addr)
        if isinstance(request, QueryTaskStatusRequest):
            return self._query_task_status_handler(request)

        # Let the caller handle the 'no handler found' case
        raise RequestHandlerUnknownMessageTypeError(
            f"Unhandled request type in base handler: {type(request).__name__}"
        )

    def _shutdown_request_handler(self, request: Request, addr: Tuple[str, int]) -> Response:
        """
        Handles a ShutdownRequest.
        Returns the Ack response and triggers a deferred shutdown.
        """
        self.logger.info(f"ShutdownRequest received from {addr}. Initiating shutdown.")
        # We must return the response *before* shutting down,
        # so we spawn a thread to call self.shutdown() after a short delay.
        threading.Thread(target=self._deferred_shutdown, daemon=True).start()
        return ShutdownAcknowledgedResponse()

    @classmethod
    def default_port(cls) -> int:
        """Returns the default port for the server."""
        return DEFAULT_PORT
