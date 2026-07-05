# START_DIFF: whisker/gui/application.py [Integrate ServerManager]
import argparse
import logging
from typing import List

from PyQt6.QtWidgets import QApplication

from whisker.core.utils import log_ascii_banner
from whisker.gui.main_window import MainWindow
from whisker.gui.debug_window import DebugWindow
from whisker.gui.signals import MessageBus

class Application:
    def __init__(
        self,
        argv: List[str],
        style: str = "windowsvista",
        log_level: int = logging.INFO,
    ):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--ui_debug", action="store_true")
        args, remaining = parser.parse_known_args(argv[:])

        self.app = QApplication(remaining)
        self.app.setStyle(style)
        # Note: We no longer set a global stylesheet on the app instance.
        # Instead, the MainWindow and its children handle themes for better performance.
        
        if args.ui_debug:
            MessageBus.enable_debug_output(True)

        self.window = MainWindow(log_level=log_level)
        self.window.show()
        
        if args.ui_debug:
            logging.info("UI debug mode enabled.")
            self.debug_window = DebugWindow()
            self.debug_window.show()

        log_ascii_banner()
        logging.info("WHISKER GUI application started successfully.")
        
    def exec(self) -> int:
        exit_code = self.app.exec()
        logging.info("GUI event loop finished.")
        return exit_code
