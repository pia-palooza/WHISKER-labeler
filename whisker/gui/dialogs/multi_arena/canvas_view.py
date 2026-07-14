from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsView


class MultiArenaCanvasView(QGraphicsView):
    """
    A zoomable canvas for placing arena boxes over a reference frame.

    Mirrors the ROI panel's ``ZoomableGraphicsView`` interaction model:
    Ctrl+wheel zooms about the cursor, buttons/`zoom_step` zoom about the
    center, and `fit_to_scene` resets. Adds arrow-key nudging of the selected
    box(es) for fine placement (Shift = coarse).
    """

    ZOOM_MIN = 0.1
    ZOOM_MAX = 25.0
    NUDGE_FINE = 1.0
    NUDGE_COARSE = 10.0

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _current_scale(self) -> float:
        s = self.transform().m11()
        return s if s else 1.0

    def _apply_zoom(self, factor: float):
        cur = self._current_scale()
        target = max(self.ZOOM_MIN, min(self.ZOOM_MAX, cur * factor))
        f = target / cur
        if abs(f - 1.0) > 1e-6:
            self.scale(f, f)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self._apply_zoom(1.25 if event.angleDelta().y() > 0 else 0.8)
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom_step(self, factor: float):
        """Zoom via buttons, anchored at the view center."""
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._apply_zoom(factor)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def fit_to_scene(self):
        """Resets zoom to fit the whole frame in the view."""
        self.resetTransform()
        scene = self.scene()
        if scene is not None:
            rect = scene.sceneRect()
            if rect is not None and not rect.isNull():
                self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        step = self.NUDGE_COARSE if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else self.NUDGE_FINE
        dx, dy = 0.0, 0.0
        key = event.key()
        if key == Qt.Key.Key_Left:
            dx = -step
        elif key == Qt.Key.Key_Right:
            dx = step
        elif key == Qt.Key.Key_Up:
            dy = -step
        elif key == Qt.Key.Key_Down:
            dy = step

        scene = self.scene()
        if (dx or dy) and scene is not None and scene.selectedItems():
            for item in scene.selectedItems():
                item.moveBy(dx, dy)  # clamped by the item's itemChange
            event.accept()
            return
        super().keyPressEvent(event)
