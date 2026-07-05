import os
import sys

from ...base import topics
from ..messaging import Message
from ..node import Node

class CliNode(Node):
    def __init__(self, label: str):
        super().__init__(
            label,
            subscriptions=set(topics.cli.Request.__members__.values())
        )
        self._input_chars: list[str] = []
        self._prompt_printed: bool = False
        self._enabled: bool = False
        
        # UNIX-specific buffer tracking
        self._unix_buffer: str = ""


    def handle_message(self, message: Message) -> bool:
        if super().handle_message(message):
            return True

        topic = message.header.topic
        if topic == topics.cli.Request.TOGGLE_ENABLE:
            self._enabled = message.payload.enable
            self.logger.info(f"CLI input {'enabled' if self._enabled else 'disabled'}")
            return True
        return False

    def wakeup(self) -> None:                
        super().wakeup()

        if self._enabled:
            if not self._prompt_printed:
                print(f"{self._label}> ", end="", flush=True)
                self._prompt_printed = True
                
            self._drain_input_stream()

    def _drain_input_stream(self) -> None:
        """
        Completely flushes the stream processing queue by accounting for
        runtime engine internal buffer caching.
        """
        if sys.platform == "win32":
            import msvcrt
            while msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ("\r", "\n"):
                    print()
                    command = "".join(self._input_chars)
                    self._input_chars.clear()
                    
                    if command:
                        self.send_outgoing_message(topics.cli.UserInputTelemetry(content=command))
                    self._prompt_printed = False
                    
                elif char == "\b":
                    if self._input_chars:
                        self._input_chars.pop()
                        print(" \b", end="", flush=True)
                else:
                    self._input_chars.append(char)
            
        else:
            import select
            
            # Step 1: Check the raw OS file descriptor without blocking
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                # Read raw bytes directly from the OS to bypass sys.stdin's internal read-ahead cache
                raw_bytes = os.read(sys.stdin.fileno(), 4096)
                if raw_bytes:
                    self._unix_buffer += raw_bytes.decode(errors="ignore")

            # Step 2: Process all complete lines now stored in our managed buffer
            while "\n" in self._unix_buffer:
                line, self._unix_buffer = self._unix_buffer.split("\n", 1)
                command = line.rstrip("\r")
                if command:
                    self.send_outgoing_message(topics.cli.UserInputTelemetry(content=command))
                self._prompt_printed = False
