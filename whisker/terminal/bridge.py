import dataclasses
from typing import Any, Tuple, List, Optional
import logging

from whisker.third_party.base_api.bridge.protocol import (
    Request, Response, Message
)
from whisker.third_party.base_api.bridge.server import BaseServer
from whisker.third_party.base_api.bridge.client import BaseClient
from .application import Application

@dataclasses.dataclass
class CLICommandRequest(Request):
    """Request to execute a single CLI command."""
    args: List[str]

@dataclasses.dataclass
class CLICommandResponse(Response):
    """Response containing the result of a single CLI command."""
    result: dict[str, Any]

@dataclasses.dataclass
class BatchCLICommandRequest(Request):
    """Request to execute a sequence of CLI commands."""
    commands: List[List[str]]

@dataclasses.dataclass
class BatchCLICommandResponse(Response):
    """Response containing a sequence of CLI command results."""
    responses: List[dict[str, Any]]

CLI_MESSAGE_TYPE_MAP = {
    "CLICommandRequest": CLICommandRequest,
    "CLICommandResponse": CLICommandResponse,
    "BatchCLICommandRequest": BatchCLICommandRequest,
    "BatchCLICommandResponse": BatchCLICommandResponse,
}

CLI_SERVER_DEFAULT_PORT = 50001

class CLIServer(BaseServer):
    """Server that executes CLI commands using a persistent Application instance."""
    
    def __init__(self, host: str | None = None, port: int | None = None):
        super().__init__(
            host=host, 
            port=port or CLI_SERVER_DEFAULT_PORT,
            additional_messages=CLI_MESSAGE_TYPE_MAP,
            user_request_handler=self.handle_cli_request
        )
        # We initialize the application once. 
        # The first command will still be slow if it imports heavy deps, 
        # but subsequent commands will be fast.
        self.app = Application(["--json"])

    def handle_cli_request(self, request: Request, addr: Tuple[str, int]) -> Response | None:
        if isinstance(request, CLICommandRequest):
            self.logger.info(f"Executing CLI command: {request.args}")
            result = self.app.run_command(request.args)
            return CLICommandResponse(result=result)
        
        elif isinstance(request, BatchCLICommandRequest):
            self.logger.info(f"Executing batch of {len(request.commands)} CLI commands")
            responses = []
            for cmd_args in request.commands:
                result = self.app.run_command(cmd_args)
                responses.append(result)
            return BatchCLICommandResponse(responses=responses)
        
        return None

class CLIClient(BaseClient):
    """Client for sending commands to a CLIServer."""
    
    def __init__(self, host: str | None = None, port: int | None = None, **kwargs):
        super().__init__(
            host=host,
            port=port or CLI_SERVER_DEFAULT_PORT,
            additional_messages=CLI_MESSAGE_TYPE_MAP,
            **kwargs
        )

    def execute_command(self, args: List[str]) -> dict[str, Any]:
        """Sends a single command to the server and returns the result."""
        request = CLICommandRequest(args=args)
        response = self.send_request(request)
        if isinstance(response, CLICommandResponse):
            return response.result
        raise RuntimeError(f"Unexpected response type from server: {type(response).__name__}")

    def execute_batch(self, commands: List[List[str]]) -> List[dict[str, Any]]:
        """Sends a batch of commands to the server and returns their results."""
        request = BatchCLICommandRequest(commands=commands)
        response = self.send_request(request)
        if isinstance(response, BatchCLICommandResponse):
            return response.responses
        raise RuntimeError(f"Unexpected response type from server: {type(response).__name__}")
