"""A click-through, always-on-top overlay that rains gold coins for a few seconds.

Used to celebrate a save while **Money Mode** is enabled. It is a *top-level*
translucent window (so it reliably paints over everything, including the video
surface), covers the whole screen the app is on, passes all input straight
through, animates falling coins, then deletes itself.
"""
import math
import random

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

_COIN_SHADES = ["#FFD700", "#F5C518", "#E6B800", "#FFDF6B", "#D4AF37"]
_RIM = QColor("#B8860B")
_DOLLAR = QColor("#7A5C00")


class _Coin:
    __slots__ = ("x", "y", "vx", "vy", "r", "phase", "spin", "shade")

    def __init__(self, width: int, height: int):
        self.r = random.uniform(9, 22)
        self.x = random.uniform(0, max(1, width))
        self.y = random.uniform(-height, 0)
        self.vx = random.uniform(-0.8, 0.8)
        self.vy = random.uniform(3.5, 8.5)
        self.phase = random.uniform(0, math.tau)  # rotation phase -> "flip" effect
        self.spin = random.uniform(0.08, 0.24)
        self.shade = random.choice(_COIN_SHADES)


class CoinConfettiOverlay(QWidget):
    """Rains gold coins across the whole screen for `duration_ms`, then removes itself."""

    def __init__(self, parent: QWidget = None, duration_ms: int = 5000, count: int = 130):
        # Top-level (no layout parent) so translucency + stacking are reliable,
        # but keep an ownership link to `parent` so it dies with the app.
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Cover the whole screen the app is currently on.
        screen = parent.screen() if parent is not None else None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())

        self._coins = [_Coin(self.width(), self.height()) for _ in range(count)]

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60 fps
        QTimer.singleShot(duration_ms, self._finish)

        self.show()
        self.raise_()

    def _tick(self):
        w, h = self.width(), self.height()
        for c in self._coins:
            c.y += c.vy
            c.x += c.vx
            c.phase += c.spin
            if c.y - c.r > h:  # recycle from just above the top
                c.y = random.uniform(-40, -c.r)
                c.x = random.uniform(0, max(1, w))
        self.update()

    def _finish(self):
        self._timer.stop()
        self.close()  # WA_DeleteOnClose schedules deletion

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        for c in self._coins:
            # |cos(phase)| squashes the width so each coin looks like it's flipping.
            wf = max(0.15, abs(math.cos(c.phase)))
            rw = c.r * wf
            rect = QRectF(c.x - rw, c.y - c.r, rw * 2, c.r * 2)

            grad = QRadialGradient(c.x, c.y - c.r * 0.3, c.r * 1.4)
            base = QColor(c.shade)
            grad.setColorAt(0.0, base.lighter(135))
            grad.setColorAt(1.0, base.darker(120))
            p.setBrush(grad)
            p.setPen(QPen(_RIM, 1.5))
            p.drawEllipse(rect)

            if wf > 0.55:  # only stamp "$" when the coin is roughly face-on
                font.setPointSizeF(max(6.0, c.r * 1.05))
                p.setFont(font)
                p.setPen(_DOLLAR)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "$")
        p.end()
