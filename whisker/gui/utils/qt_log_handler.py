from datetime import datetime
import logging
import sys

from PyQt6.QtCore import QObject, pyqtSignal

class QtLogHandler(logging.Handler, QObject):
    """
    A custom logging handler that emits a signal for each log record.
    """

    log_received = pyqtSignal(str)

    def __init__(self, logger=None, parent=None):
        try:
            super().__init__()
            QObject.__init__(self, parent)
            
            self._colors = {
                logging.DEBUG: "gray",
                logging.INFO: "black",
                logging.WARNING: "darkorange",
                logging.ERROR: "red",
                logging.CRITICAL: "red",
            }
            self._prefixes = {
                logging.DEBUG: "[DEBUG]",
                logging.INFO: "[INFO]",
                logging.WARNING: "[WARN]",
                logging.ERROR: "[ERROR]",
                logging.CRITICAL: "[CRITICAL]",
            }

            if logger:
                self._attach_to_logger(logger)
        except Exception as e:
            print(f"Failed to initialize QtLogHandler: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """Formats the log record into an HTML string and emits the signal."""
        try:
            msg = self.format(record)
            log_time = datetime.fromtimestamp(record.created)
            time_str = log_time.strftime("%H:%M:%S.%f")[:-3]

            level_no = record.levelno
            color = self._colors.get(level_no, "black")
            prefix = self._prefixes.get(level_no, "[INFO]")

            log_entry = (
                f'<span style="color: gray;">{time_str}</span> '
                f'<span style="color: {color}; font-weight: bold;">{prefix}:</span> '
                f'<span style="color: {color};">{msg}</span>'
            )
            
            self.log_received.emit(log_entry)
            
        except RuntimeError as e:
            # Catching the common PyQt "object deleted" error silently
            if 'has been deleted' not in str(e):
                print(f"QtLogHandler Runtime Error: {e}", file=sys.stderr)
        except Exception as e:
            # Fallback for unexpected formatting or timestamp issues
            print(f"Failed to emit log record: {e}", file=sys.stderr)

    def _attach_to_logger(self, logger: logging.Logger):
        """Attaches the handler to a logger."""
        try:
            self.setFormatter(logging.Formatter("%(message)s"))
            self.setLevel(logging.DEBUG)
            logger.addHandler(self)
            logging.debug(f"Attached QtLogHandler to {logger.name}")
        except Exception as e:
            print(f"Could not attach QtLogHandler to logger: {e}", file=sys.stderr)