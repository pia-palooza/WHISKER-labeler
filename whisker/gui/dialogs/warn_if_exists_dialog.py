from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QApplication
from PyQt6.QtCore import QObject, pyqtSignal, Qt

class DialogBridge(QObject):
    """Internal helper to bridge threads."""
    request_dialog = pyqtSignal(str, object)  # message, container for result

    def __init__(self):
        super().__init__()
        self.request_dialog.connect(self._on_request, Qt.ConnectionType.BlockingQueuedConnection)

    def _on_request(self, message, result_wrap):
        # This part runs on the Main Thread
        dialog = WarnIfExistsDialog(message)
        result_wrap['value'] = (dialog.exec() == QDialog.DialogCode.Accepted)

# Create a single instance to manage dialog requests
BRIDGE = DialogBridge()

class WarnIfExistsDialog(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warning")
        self.setModal(True)

        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)

        btns = QHBoxLayout()
        btns.addStretch()
        for text, slot in [("Yes", self.accept), ("No", self.reject)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btns.addWidget(btn)
        layout.addLayout(btns)

    @classmethod
    def run(cls, message, parent=None):
        # If we're already on the Main Thread, just show it
        if QApplication.instance().thread() == QObject().thread():
            dialog = cls(message, parent)
            return dialog.exec() == QDialog.DialogCode.Accepted

        # If we're on a worker thread, emit to the bridge and wait
        # We use a dict to bypass scoping issues
        result_wrap = {'value': False}
        BRIDGE.request_dialog.emit(message, result_wrap)
        return result_wrap['value']