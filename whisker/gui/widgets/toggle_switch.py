import sys
from dataclasses import dataclass
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox
from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtProperty, QEasingCurve, QRect
from PyQt6.QtGui import QPainter, QColor, QFont

@dataclass
class SizeSettings:
    width: int = 80
    height: int = 32
    margin: int = 3

@dataclass
class StyleSettings:
    on_color: str = "#4CAF50"
    off_color: str = "#BBBBBB"
    on_text_color: str = "#FFFFFF"
    off_text_color: str = "#FFFFFF"
    thumb_color: str = "#FFFFFF"
    font: QFont = QFont("Segoe UI", 9, QFont.Weight.Bold)

class ToggleSwitch(QCheckBox):
    def __init__(self, on_text="ON", off_text="OFF", parent=None):
        super().__init__(parent)
        self._on_text = on_text
        self._off_text = off_text
        self._size_cfg = SizeSettings()
        self._style_cfg = StyleSettings()
        
        self._circle_pos = 0.0
        self._anim = QPropertyAnimation(self, b"circle_pos", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setDuration(200)
        
        self._update_geometry()

    # --- Getters / Setters ---
    def set_size_settings(self, settings: SizeSettings):
        self._size_cfg = settings
        self._update_geometry()

    def get_size_settings(self) -> SizeSettings: return self._size_cfg

    def set_style_settings(self, settings: StyleSettings):
        self._style_cfg = settings
        self.update()

    def get_style_settings(self) -> StyleSettings: return self._style_cfg

    def _update_geometry(self):
        s = self._size_cfg
        self.setFixedSize(s.width, s.height)
        self._thumb_size = s.height - (s.margin * 2)
        self._on_pos = s.width - self._thumb_size - s.margin
        self._off_pos = s.margin
        self._circle_pos = self._on_pos if self.isChecked() else self._off_pos
        self.update()

    @pyqtProperty(float)
    def circle_pos(self): return self._circle_pos

    @circle_pos.setter
    def circle_pos(self, pos):
        self._circle_pos = pos
        self.update()

    def nextCheckState(self):
        super().nextCheckState()
        end = self._on_pos if self.isChecked() else self._off_pos
        self._anim.stop()
        self._anim.setEndValue(end)
        self._anim.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        
        # Track
        color = self._style_cfg.on_color if self.isChecked() else self._style_cfg.off_color
        p.setBrush(QColor(color))
        p.drawRoundedRect(0, 0, self.width(), self.height(), self.height()//2, self.height()//2)

        # Text
        text_color = self._style_cfg.on_text_color if self.isChecked() else self._style_cfg.off_text_color
        p.setPen(QColor(text_color))
        p.setFont(self._style_cfg.font)
        
        text = self._on_text if self.isChecked() else self._off_text
        tx = 0 if self.isChecked() else self._thumb_size
        rect = QRect(int(tx), 0, self.width() - int(self._thumb_size), self.height())
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # Thumb
        p.setBrush(QColor(self._style_cfg.thumb_color))
        p.drawEllipse(int(self._circle_pos), self._size_cfg.margin, self._thumb_size, self._thumb_size)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Simple CLI extraction
    off_val = sys.argv[1] if len(sys.argv) > 1 else "NO"
    on_val = sys.argv[2] if len(sys.argv) > 2 else "YES"
    w_val = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    h_val = int(sys.argv[4]) if len(sys.argv) > 4 else 32

    win = QWidget()
    win.setWindowTitle("Custom Toggle Settings Test")
    layout = QVBoxLayout(win)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    toggle = ToggleSwitch(on_val, off_val)
    
    # Test the setters
    custom_size = SizeSettings(width=w_val, height=h_val, margin=4)
    custom_style = StyleSettings(
        on_color="#2196F3",
        on_text_color="#FFD700", # Golden text when ON
        off_text_color="#000000"  # Black text when OFF
    )
    
    toggle.set_size_settings(custom_size)
    toggle.set_style_settings(custom_style)
    
    layout.addWidget(toggle)
    win.resize(400, 300)
    win.show()
    sys.exit(app.exec())