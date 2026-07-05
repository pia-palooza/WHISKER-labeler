import os
from pathlib import Path
import time
from typing import Dict
import subprocess
import sys

from ...base import topics as topics
from ..messaging import Message
from ..node import Node

topics.service_management = topics.service_management

def get_conda_env_python(environment_name: str) -> str | None:
    # Favor standard conda environment variables for robust path resolution
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).parent.parent
        
        env_path_win = conda_root / "envs" / environment_name / "python.exe"
        if env_path_win.is_file():
            return str(env_path_win)
            
        env_path_posix = conda_root / "envs" / environment_name / "bin" / "python"
        if env_path_posix.is_file():
            return str(env_path_posix)

    # Fallback heuristic if variables are missing
    current_exe = Path(sys.executable)
    for parent in current_exe.parents:
        if parent.name.lower() == "anaconda3":
            env_path = parent / "envs" / environment_name / "python.exe"
            if env_path.is_file():
                return str(env_path)
            
            env_path_posix = parent / "envs" / environment_name / "bin" / "python"
            if env_path_posix.is_file():
                return str(env_path_posix)
            
    return None

class ServiceManagementNode(Node):
    def __init__(self, label: str = "service_manager"):
        super().__init__(
            label=label,
            subscriptions=set(
                [topics.node.Reply.PONG]
                + list(topics.service_management.Request.__members__.values())
            )
        )
        self._services: Dict[str, topics.service_management.ManagedServiceState] = {}
        self._last_pings: Dict[str, float] = {}

    def wakeup(self) -> None:
        super().wakeup()
        self._check_service_health()

    def start_service(self, message: Message) -> None:
        config = message.payload.config
        name = config.module_name

        if name in self._services:
            self.logger.info(msg=f"Service '{name}' is already managed.")
            return

        if config.environment_name:
            if config.executable_path:
                self.logger.error(
                    "Both executable_path and environment_name are set. "
                    "Using value from environment_name.",
                )
            executable_path = get_conda_env_python(config.environment_name)
            if not executable_path:
                self.logger.warning(
                    f"Environment '{config.environment_name}' python not found. "
                    "Falling back to sys.executable."
                )
                executable_path = sys.executable
        else:
            executable_path = config.executable_path or sys.executable

        cmd = [executable_path, "-m", name] + config.args
        self.logger.info(f"Command: {cmd}")
        
        try:
            process = subprocess.Popen(cmd)
            self._services[name] = topics.service_management.ManagedServiceState(
                config=config,
                process=process,
                last_heartbeat=time.time()
            )
            self.send_outgoing_reply(message.header.sender_id, topics.service_management.ServiceLaunchedReply(module_name=name, pid=process.pid))
        except Exception as e:
            self.logger.error(f"Failed to start service '{name}': {e}")

    def handle_message(self, message: Message) -> bool:
        if super().handle_message(message):
            return True

        topic = message.header.topic
        if message.header.target_node_id == self._uuid:
            if topic == topics.node.Reply.PONG:
                self._record_heartbeat(message.header.sender_id)
                return True
            elif topic == topics.service_management.Request.START_SERVICE:
                self.start_service(message)
                return True
            elif topic == topics.service_management.Request.ACCEPT_SERVICE_CLIENT_CONNECTION:
                state = self._services.get(message.payload.module_name)
                if state and not state.is_shutting_down:
                    state.uuid = message.payload.uuid or message.header.sender_id
                    state.last_heartbeat = time.time()
                    self.send_outgoing_reply(
                        message.header.message_id,
                        topics.service_management.ServiceClientConnectionAcceptedReply(
                            module_name=state.config.module_name,
                            pid=message.payload.pid,
                            uuid=state.uuid
                        )
                    )
                return True
        return False

    def _run_loop_iteration(self) -> None:
        """Overriden by child classes to implement a function which is called on each run loop iteration."""
        self._check_service_health()

    def _record_heartbeat(self, sender_uuid: str) -> None:
        now = time.time()
        for state in self._services.values():
            if state.uuid == sender_uuid:
                state.last_heartbeat = now
                return

    def _check_service_health(self) -> None:
        now = time.time()
        dead_services = []

        for name, state in self._services.items():
            if state.is_shutting_down:
                continue

            # 1. OS-Level Process Verification
            if state.process.poll() is not None:
                self.logger.error(f"Service '{name}' OS process died unexpectedly.")
                dead_services.append(name)
                continue

            # 2. Framework Heartbeat Verification
            time_since_beat = now - state.last_heartbeat
            if time_since_beat > state.config.heartbeat_timeout:
                if state.uuid is not None or time_since_beat > state.config.startup_timeout:
                    self.logger.error(f"Service '{name}' heartbeat timed out.")
                    self._terminate_service_hard(name, state)
                    dead_services.append(name)

        for name in dead_services:
            self._services.pop(name, None)

    def stop_service(self, name: str, timeout: float = 5.0) -> None:
        if name not in self._services:
            return
        
        state = self._services[name]
        state.is_shutting_down = True
        
        if state.uuid and state.process.poll() is None:
            self.send_outgoing_message(topics.node.ShutdownRequest(), target_node_id=state.uuid)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                if state.process.poll() is not None:
                    self.logger.info(f"Service '{name}' shut down gracefully.")
                    break
                time.sleep(0.1)

        if state.process.poll() is None:
            self._terminate_service_hard(name, state)
            
        self._services.pop(name, None)

    def _terminate_service_hard(self, name: str, state: topics.service_management.ManagedServiceState) -> None:
        self.logger.warning(f"Force killing service '{name}' (PID {state.process.pid}).")
        state.process.kill()
        try:
            state.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.logger.error(f"Service '{name}' resisted SIGKILL.")

    def shutdown(self) -> None:
        for name in list(self._services.keys()):
            self.stop_service(name, timeout=5.0)
            
        super().shutdown()