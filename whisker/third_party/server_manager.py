import logging
import subprocess
import threading
import time
import socket
import atexit
from typing import Dict, Optional
from pathlib import Path

from whisker.third_party.base_api.bridge.protocol import (
    EchoRequest,
    EchoResponse,
    ShutdownRequest,
    ShutdownAcknowledgedResponse,
    DEFAULT_HOST,
)
from whisker.third_party.base_api.bridge.client import BaseClient
from .command_runner import CommandRunner, CommandExecutionError
from datetime import datetime

# --- Singleton Pattern ---
_manager_instance: Optional["BackendServerManager"] = None
_manager_lock = threading.RLock()  # Use RLock instead of Lock for reentrancy

def get_server_manager() -> "BackendServerManager":
    """Returns the singleton instance of the BackendServerManager."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = BackendServerManager()        
    return _manager_instance


# --- End Singleton Pattern ---


class BackendServerManager:
    """Manages the lifecycle of background backend server processes."""

    def __init__(self):
        self._servers: Dict[str, subprocess.Popen | None] = {}
        self._server_info: Dict[str, Dict] = {}
        self._lock = threading.RLock()  # Use RLock for reentrant locking
        self._runner = CommandRunner()
        self.logger = logging.getLogger(__name__)
        self._log_dir_base: Optional[Path] = None
        atexit.register(self.stop_all_servers)

    def set_logging_dir(self, logging_dir_base: Path | str | None):
        """Sets the base path where the 'server_logs' directory will be created."""
        if logging_dir_base:
            self._log_dir_base = Path(logging_dir_base)
            self.logger.debug(f"Base directory for third-party server logs set to: {self._log_dir_base}")
        else:
            self._log_dir_base = None
            self.logger.warning("Base directory for server logs is not set. Logs will be discarded.")

    def is_server_running(self, backend_name: str) -> bool:
        """Checks if a server process exists and is still running."""
        # This method acquires the lock
        with self._lock:
            process = self._servers.get(backend_name)
            if process and process.poll() is None:
                return True
            if process:
                self.logger.warning(
                    f"Server process for '{backend_name}' found terminated unexpectedly (exit code: {process.poll()})."
                )
                self._servers.pop(backend_name, None)
                self._server_info.pop(backend_name, None)
            return False

    def _wait_for_server_ready(self, backend_name: str, port: int, timeout: int = 120) -> bool:
        """Tries to connect to the server port to ensure it's ready."""
        start_time = time.time()
        self.logger.info(f"Waiting for server '{backend_name}' on port {port} to become ready...")
        while time.time() - start_time < timeout:
            if not self.is_server_running(backend_name):
                self.logger.error(f"Server process '{backend_name}' terminated during startup check.")
                return False

            try:
                client = BaseClient(port=port)
                response = client.send_request(
                    EchoRequest(
                        message=f"Requesting echo from server '{backend_name}' on port {port}"
                    )
                )
                if isinstance(response, EchoResponse):
                    self.logger.info(f"Received echo from server '{backend_name}' on port {port}. It's ready!")
                    return True
            except (socket.timeout, ConnectionRefusedError):
                pass
            except Exception as e:
                self.logger.error(f"Error checking server readiness for '{backend_name}': {e}")
                return False

            time.sleep(0.5)
        self.logger.error(f"Server '{backend_name}' on port {port} did not become ready within {timeout} seconds.")
        return False

    def _quick_echo_check(self, port: int) -> bool:
        """
        Tries a single, fast echo request to see if a server is already
        listening on the port. Returns True on success, False on any failure.
        """
        try:
            # Use a client with no retries.
            # max_retries=0 means 1 attempt.
            client = BaseClient(host=DEFAULT_HOST, port=port, max_retries=0, retry_delay=0)
            response = client.send_request(EchoRequest(message="ping"))

            if isinstance(response, EchoResponse):
                self.logger.debug(f"Quick echo check on port {port} succeeded.")
                return True
        except (ConnectionError, ConnectionRefusedError):
            # This is the expected failure case if the port is free.
            self.logger.debug(f"Quick echo check on port {port} failed (connection refused/error).")
            return False
        except Exception as e:
            # Other errors (e.g., protocol mismatch, timeout, json error)
            self.logger.debug(f"Quick echo check on port {port} failed unexpectedly: {e}")
            return False

        # Fallthrough in case of unexpected non-EchoResponse
        return False

    def ensure_server_running(
        self,
        backend_name: str,
        server_script_path: str,
        conda_env_name: str,
        port: int,
    ):
        """
        Starts the server if it's not running, redirecting output to log files.
        If a server is already running on the port, it registers it as external.
        """
        # Check 1: Is this manager already managing this process?
        if self.is_server_running(backend_name):
            self.logger.debug(f"Server '{backend_name}' is already managed and running.")
            return

        # --- NEW: Check for externally managed server ---
        # This is a non-locking check.
        if self._quick_echo_check(port):
            with self._lock:
                # Double-check 1: Did *this* manager start it while we were checking?
                if self.is_server_running(backend_name):
                    self.logger.debug(f"Server '{backend_name}' was started by this manager concurrently.")
                    return

                # Double-check 2: Have we *already* registered this external server?
                # is_server_running returns False for 'None' processes, so we must
                # check _server_info directly for the 'external' flag.
                info = self._server_info.get(backend_name)
                if info and info.get('external'):
                    self.logger.debug(f"Server '{backend_name}' is already registered as external.")
                    return

                self.logger.info(
                    f"Detected externally managed server for '{backend_name}' on port {port}. "
                    "Registering it for shutdown management only."
                )
                self._servers[backend_name] = None  # Mark as no Popen object
                self._server_info[backend_name] = {
                    'script': 'EXTERNAL',
                    'env': 'EXTERNAL',
                    'port': port,
                    'pid': None,
                    'log': 'EXTERNAL',
                    'external': True
                }
            return
        # --- END NEW LOGIC ---

        # If we get here, the port is free. Proceed with original logic.
        with self._lock:
            # Original double-check-lock.
            if self.is_server_running(backend_name):
                self.logger.debug(f"Server '{backend_name}' found running after acquiring lock.")
                return

            self.logger.info(f"Starting server for '{backend_name}'...")
            script_path_resolved = str(Path(server_script_path).resolve())

            # --- Create Log Paths ---
            stdout_log_path_str = None
            stderr_log_path_str = None
            log_dir = None
            if self._log_dir_base:
                try:
                    log_dir = self._log_dir_base / "server_logs"
                    logging.info(f"Ensuring server log directory exists: {log_dir.resolve()}")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    log_file_name = f"{backend_name}_server_{timestamp}.log"
                    stdout_log_path = log_dir / log_file_name
                    stdout_log_path_str = str(stdout_log_path.resolve())

                    # --- MODIFICATION ---
                    # Set stderr to None to trigger the logic in CommandRunner
                    # to redirect stderr to the *same file handle* as stdout.
                    # This is more robust than opening two handles to the same file.
                    stderr_log_path_str = None
                    # --- END MODIFICATION ---

                    logging.info(f"Will attempt redirecting '{backend_name}' server output to {stdout_log_path_str}")
                except Exception as e:
                    logging.error(f"Failed to create or access log directory {log_dir}: {e}. Server output will be discarded.", exc_info=True)
                    stdout_log_path_str = None
                    stderr_log_path_str = None
            else:
                logging.warning("Base logging directory not set in ServerManager. Server output will be discarded.")

            try:
                logging.info(f"Attempting to call CommandRunner.run_script_async for '{backend_name}'...")
                process = self._runner.run_script_async(
                    conda_env_name=conda_env_name,
                    script_path=script_path_resolved,
                    script_args=[],
                    stdout_log_path=stdout_log_path_str,
                    stderr_log_path=stderr_log_path_str,  # This will now be None
                )

                self._servers[backend_name] = process
                self._server_info[backend_name] = {
                    'script': script_path_resolved, 'env': conda_env_name, 'port': port,
                    'pid': process.pid, 'log': stdout_log_path_str,
                    'external': False  # Explicitly mark as *not* external
                }
                self.logger.info(
                    f"Server process for '{backend_name}' started with PID {process.pid}."
                )

                if not self._wait_for_server_ready(backend_name, port):
                    log_msg = f" Check logs at: {stdout_log_path_str}" if stdout_log_path_str else ""
                    self.logger.error(f"Failed to confirm server '{backend_name}' startup. Terminating process.{log_msg}")
                    # Calls stop_server (which acquires lock again) - OK with RLock
                    self.stop_server(backend_name)
                    raise RuntimeError(f"Server '{backend_name}' failed to start properly.{log_msg}")

                self.logger.info(f"Server '{backend_name}' confirmed running and ready.")

            except (CommandExecutionError, FileNotFoundError, RuntimeError) as e:
                self.logger.critical(f"Failed to start server '{backend_name}': {e}", exc_info=True)
                self._servers.pop(backend_name, None)
                self._server_info.pop(backend_name, None)
                raise
            except Exception as e:
                self.logger.critical(f"Unexpected error starting server '{backend_name}': {e}", exc_info=True)
                self._servers.pop(backend_name, None)
                self._server_info.pop(backend_name, None)
                raise RuntimeError(f"Unexpected error starting server '{backend_name}'.") from e

    def stop_server(self, backend_name: str):
        """
        Stops a specific backend server process.
        Performs I/O (network, process) *after* releasing the lock.
        """
        # --- Begin Critical Section ---
        with self._lock:
            process = self._servers.pop(backend_name, None)
            info = self._server_info.pop(backend_name, None)
        # --- End Critical Section ---

        # All subsequent operations are blocking I/O and MUST be outside the lock

        if not process and not info:
            self.logger.debug(f"No active server process found for '{backend_name}' to stop.")
            return

        pid = info.get('pid', 'N/A') if info else 'N/A'
        log_path_msg = f", Log: {info['log']}" if info and info.get('log') else ""
        self.logger.info(f"Stopping server '{backend_name}' (PID: {pid}{log_path_msg})...")

        # 1. Attempt graceful shutdown via network request
        if info and 'port' in info:
            try:
                # Use low retries; if it's not listening, don't hang here
                client = BaseClient(port=info['port'], max_retries=1, retry_delay=0)
                response = client.send_request(ShutdownRequest())
                self.logger.debug(f"Sent ShutdownRequest to '{backend_name}'.")
                if isinstance(response, ShutdownAcknowledgedResponse):
                    self.logger.info(f"Recieved ShutdownAcknowledgedResponse from '{backend_name}'")
            except Exception as client_err:
                self.logger.warning(
                    f"Failed to send ShutdownRequest to '{backend_name}' (PID: {pid}). "
                    f"Process may be unresponsive or already down. Error: {client_err}"
                )

        # 2. Manage the process lifecycle
        if process and process.poll() is None:
            try:
                # Wait for graceful shutdown (from network request)
                process.wait(timeout=5)
                self.logger.info(f"Server '{backend_name}' (PID: {pid}) stopped gracefully (Exit code: {process.poll()}).")
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Server '{backend_name}' (PID: {pid}) did not terminate after 5s, sending SIGTERM...")
                try:
                    process.terminate()  # SIGTERM
                    process.wait(timeout=2)
                    self.logger.info(f"Server '{backend_name}' stopped (Exit code: {process.poll()}).")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Server '{backend_name}' (PID: {pid}) did not terminate gracefully, sending SIGKILL.")
                    process.kill()  # SIGKILL
                    process.wait(timeout=2)  # Give OS time to kill
                    self.logger.info(f"Server '{backend_name}' stopped (Exit code: {process.poll()}).")
            except Exception as e:
                self.logger.error(f"Error stopping server '{backend_name}' (PID: {pid}): {e}", exc_info=True)

        elif process:
            self.logger.info(f"Server '{backend_name}' (PID: {pid}) was already stopped (Exit code: {process.poll()}).")

    def stop_all_servers(self):
        """Stops all managed backend server processes."""
        logging.info("Stopping all managed backend servers...")

        # Make a copy of the keys to avoid modifying dict while iterating
        for name in list(self._servers.keys()):
            self.stop_server(name)

        logging.info("Finished stopping servers.")
        atexit.unregister(self.stop_all_servers)

    def __del__(self):
        self.stop_all_servers()