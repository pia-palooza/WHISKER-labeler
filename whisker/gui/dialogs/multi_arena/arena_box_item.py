from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QBrush, QPen
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsItem


class ArenaBoxItem(QGraphicsRectItem):
    """
    A fixed-size, axis-aligned arena "stamp" placed over one arena in a frame.

    Unlike the ROI panel's ``ResizableRectItem`` (which resizes and rotates), an
    arena box is intentionally *move-only*: every box in a dataset shares one
    width/height, so only its position varies. The box lives in scene
    coordinates that map 1:1 to full-frame pixels, so its top-left position is
    the ``(x, y)`` placement we persist.
    """

    def __init__(self, width: int, height: int, color_hex: str = "#00E5A0", index: int = 1):
        super().__init__(0.0, 0.0, float(width), float(height))
        self._color_hex = color_hex
        self.index = index

        pen = QPen(QColor(color_hex), 3)
        pen.setCosmetic(True)  # constant on-screen width regardless of zoom
        self.setPen(pen)
        brush_color = QColor(color_hex).lighter(120)
        brush_color.setAlpha(40)
        self.setBrush(QBrush(brush_color))  # translucent fill

        self.setAcceptHoverEvents(True)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    # -- geometry ----------------------------------------------------------
    def set_box_size(self, width: int, height: int):
        """Resizes the box in place (top-left fixed). Used when the shared
        per-dataset box size changes."""
        self.prepareGeometryChange()
        self.setRect(0.0, 0.0, float(width), float(height))
        self.update()

    def top_left(self) -> QPointF:
        """The box's full-frame-pixel top-left (its scene position)."""
        return self.pos()

    # -- interaction -------------------------------------------------------
    def hoverMoveEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def itemChange(self, change, value):
        # Clamp the box so it stays within the frame whenever it fits. value is
        # the proposed new top-left position (scene coords) for a move.
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene() is not None:
            scene_rect = self.scene().sceneRect()
            if scene_rect is not None and not scene_rect.isNull():
                rect = self.rect()
                new_x, new_y = value.x(), value.y()
                max_x = scene_rect.width() - rect.width()
                max_y = scene_rect.height() - rect.height()
                if max_x >= 0:
                    new_x = min(max(new_x, 0.0), max_x)
                if max_y >= 0:
                    new_y = min(max(new_y, 0.0), max_y)
                return QPointF(new_x, new_y)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        # Selection emphasis: a white dashed outline on top of the colored pen.
        if self.isSelected():
            sel_pen = QPen(QColor("#FFFFFF"), 2, Qt.PenStyle.DashLine)
            sel_pen.setCosmetic(True)
            painter.setPen(sel_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect())

        # Index badge in the top-left corner so users can tell boxes apart.
        rect = self.rect()
        painter.setPen(QPen(QColor(self._color_hex), 2))
        painter.drawText(
            QRectF(rect.left() + 4, rect.top() + 2, 40, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            str(self.index),
        )
