import io
import json
import logging
import logging.handlers
import socketserver
import struct
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

VALID_LOG_LEVELS = {
    logging.DEBUG, logging.INFO, logging.WARNING,
    logging.ERROR, logging.CRITICAL
}

_logger: Optional[logging.Logger] = None

def _log_invocation_args() -> None:
    try:
        call_args = sys.argv[:]
        whisker_path = Path(call_args[0]).resolve().parent.parent.parent
        call_args[0] = call_args[0].replace(str(whisker_path), "")
        logging.info(f"Invocation arguments: {call_args}")
    except Exception as e:
        logging.info(f"Invocation arguments (fallback): {sys.argv} | Note: {e}")


class JsonSocketHandler(logging.handlers.SocketHandler):
    """Transmits LogRecords as length-prefixed JSON strings over TCP."""

    def makePickle(self, record: logging.LogRecord) -> bytes:
        exc_text = None
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))

        log_dict = {
            "name": record.name,
            "msg": record.getMessage(),
            "levelname": record.levelname,
            "levelno": record.levelno,
            "pathname": record.pathname,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "created": record.created,
            "exc_text": exc_text,
        }
        encoded_json = json.dumps(log_dict).encode("utf-8")
        length_prefix = struct.pack(">L", len(encoded_json))
        return length_prefix + encoded_json

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "funcName": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exc_text"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

class LogStreamHandler(socketserver.StreamRequestHandler):
    """Decodes incoming JSON logs and handles them locally."""

    def handle(self):
        while True:
            try:
                chunk = self.request.recv(4)
                if len(chunk) < 4:
                    break
                slen = struct.unpack(">L", chunk)[0]
                chunk = self.request.recv(slen)
                while len(chunk) < slen:
                    chunk = chunk + self.request.recv(slen - len(chunk))

                log_data = json.loads(chunk.decode("utf-8"))
                record = logging.makeLogRecord(log_data)

                if log_data.get("exc_text"):
                    record.msg = f"{record.msg}\n{log_data['exc_text']}"

                # Crucial Fix: Use the server's tracking target to route back
                # to the file and console handlers, ignoring child environment resets.
                self.server.target_logger.handle(record)
            except Exception:
                break


class LogSocketReceiver(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host="127.0.0.1", port=9020, target_logger=None):
        super().__init__((host, port), LogStreamHandler)
        self.target_logger = target_logger or logging.getLogger()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self.shutdown()
        self.server_close()

def configure_console_logger(
    level: int = logging.INFO, fmt: str = DEFAULT_FORMAT, stream: Optional[io.TextIOBase] = None
) -> None:
    global _logger
    root = logging.getLogger()
    _logger = root

    if level not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level {level} requested.")

    console_handler = next(
        (h for h in root.handlers if type(h) is logging.StreamHandler),
        None
    )
    if console_handler is not None:
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(stream or sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt))
        console_handler.setLevel(level)
        root.addHandler(console_handler)
    
    # Ensure root allows logs to pass through to the lowest configured handler
    if root.getEffectiveLevel() > level or root.level == logging.NOTSET:
        root.setLevel(level)

    if level == logging.DEBUG:
        logging.debug("Logging initialized in DEBUG mode.")
    _log_invocation_args()

def configure_file_logger(
    file_path: Union[str, Path],
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    as_json: bool = False,
) -> None:
    root = logging.getLogger()
    log_path = Path(file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if any(
        isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(log_path.resolve())
        for h in root.handlers
    ):
        return

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    
    if as_json:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(fmt))
        
    root.addHandler(file_handler)
    root.setLevel(level)


def configure_socket_logger(host: str = "127.0.0.1", port: int = 9020, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h, JsonSocketHandler) for h in root.handlers):
        return
        
    socket_handler = JsonSocketHandler(host, port)
    socket_handler.setLevel(level)
    root.addHandler(socket_handler)
    
    # Only lower the root funnel, never restrict it if another handler needs more verbosity
    if root.getEffectiveLevel() > level or root.level == logging.NOTSET:
        root.setLevel(level)

@dataclass(frozen=True)
class ConsoleConfig:
    level: int = logging.INFO
    fmt: str = DEFAULT_FORMAT
    stream: Optional[io.TextIOBase] = None


@dataclass(frozen=True)
class FileConfig:
    file_path: Union[str, Path]
    level: int = logging.INFO
    fmt: str = DEFAULT_FORMAT
    max_bytes: int = 10_485_760
    backup_count: int = 5
    as_json: bool = False

@dataclass(frozen=True)
class SocketConfig:
    host: str = "127.0.0.1"
    port: int = 9020
    level: int = logging.INFO


@dataclass(frozen=True)
class LoggerConfig:
    console: Optional[ConsoleConfig] = None
    file: Optional[FileConfig] = None
    socket: Optional[SocketConfig] = None


def configure_loggers(config: LoggerConfig) -> None:
    if config.console is not None:
        configure_console_logger(
            level=config.console.level, fmt=config.console.fmt, stream=config.console.stream
        )
    if config.file is not None:
        configure_file_logger(
            file_path=config.file.file_path,
            level=config.file.level,
            fmt=config.file.fmt,
            max_bytes=config.file.max_bytes,
            backup_count=config.file.backup_count,
            as_json=config.file.as_json
        )
    if config.socket is not None:
        configure_socket_logger(host=config.socket.host, port=config.socket.port, level=config.socket.level)


def configure_workspace_logging(workspace_dir: Union[str, Path], level: int = logging.INFO) -> Path:
    """
    Enables file logging to a session-specific file within the workspace.
    """
    global _logger
    root = logging.getLogger()
    _logger = root

    if level not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level {level} requested.")

    workspace_path = Path(workspace_dir)
    sessions_dir = workspace_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = sessions_dir / f"session_{timestamp}.log"

    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve())
        for h in root.handlers
    ):
        return log_path

    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode='w', encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(log_formatter)

    root.addHandler(file_handler)
    
    if root.getEffectiveLevel() > level or root.level == logging.NOTSET:
        root.setLevel(level)

    logging.debug(f"File logging enabled. Log file: {log_path}")
    return log_path


def get_logger() -> Optional[logging.Logger]:
    """Returns the configured root logger, or None if not configured."""
    global _logger
    return _logger


def shutdown() -> None:
    """Closes and removes all handlers from the root logger."""
    global _logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    _logger = None

