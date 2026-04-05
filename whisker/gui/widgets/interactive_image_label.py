from typing import Dict, Optional, List, Any, Tuple

from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap, QFont, QColor, QPen, QWheelEvent
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class InteractiveImageLabel(QGraphicsView):
    point_placed = pyqtSignal(QPointF)
    point_dragged = pyqtSignal(str, str, QPointF)

    DRAG_THRESHOLD_SQ = 10 * 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._prediction_points: List[Dict[str, Any]] = []
        self._gt_points: List[Dict[str, Any]] = []
        self._primer_points: List[Dict[str, Any]] = []
        self._show_gt_points = True

        self._show_names = False
        self._font = QFont("Arial", 8)

        self._is_drag_mode = False
        self._dragged_keypoint_info: Optional[Tuple[str, str]] = None
        
        # Default Configuration: Panning enabled, interact with Right Click
        self._zoom_enabled = True
        self._point_button = Qt.MouseButton.RightButton
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_interaction_mode(self, pan_zoom: bool, point_button: Qt.MouseButton):
        """Configures how the user interacts with the canvas and points."""
        self._zoom_enabled = pan_zoom
        self._point_button = point_button
        if pan_zoom:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_full_pixmap(self, pixmap: Optional[QPixmap]):
        self.scene().clear()
        self._pixmap_item = None
        if pixmap and not pixmap.isNull():
            self._pixmap_item = self.scene().addPixmap(pixmap)
            
            # Inflate the scene boundary massively to allow "out of bounds" panning
            w, h = pixmap.width(), pixmap.height()
            self.scene().setSceneRect(QRectF(-w * 2, -h * 2, w * 5, h * 5))
            
            # Fit strictly to the image itself on load, not the padded bounds
            self.fitInView(QRectF(pixmap.rect()), Qt.AspectRatioMode.KeepAspectRatio)
            
        self.viewport().update()

    def set_drag_mode(self, enabled: bool):
        self._is_drag_mode = enabled
        self._dragged_keypoint_info = None

    def set_display_data(self, prediction_points: List[Dict], gt_points=None):
        self._prediction_points = prediction_points or []
        self._gt_points = gt_points or []
        self.viewport().update()

    def set_prediction_display_data(self, primer_points: List[Dict], *args):
        self._primer_points = primer_points or []
        self.viewport().update()

    def set_show_gt_points(self, show: bool):
        if self._show_gt_points != show:
            self._show_gt_points = show
            self.viewport().update()

    def set_show_names(self, show: bool):
        self._show_names = show
        self.viewport().update()

    def set_font_size(self, size: int):
        if size > 0:
            self._font.setPointSize(size)
            self.viewport().update()

    def wheelEvent(self, event: QWheelEvent):
        if not self._zoom_enabled:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        
        old_pos = self.mapToScene(event.position().toPoint())

        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, ev: QMouseEvent):
        if not self._pixmap_item:
            return

        # Check against the configured interaction button
        if ev.button() == self._point_button:
            scene_pos = self.mapToScene(ev.position().toPoint())

            if self._is_drag_mode:
                view_scale = self.transform().m11()
                # Adjust the selection hit-box based on the view scale so it remains constant on screen
                adjusted_thresh_sq = self.DRAG_THRESHOLD_SQ / (view_scale**2)
                found_keypoint = None

                for identity_data in self._prediction_points:
                    identity_id = identity_data["name"]
                    for part_name, image_coords in identity_data["points"].items():
                        dist_vector = scene_pos - image_coords
                        dist_sq = dist_vector.x() ** 2 + dist_vector.y() ** 2

                        if dist_sq < adjusted_thresh_sq:
                            adjusted_thresh_sq = dist_sq
                            found_keypoint = (identity_id, part_name)

                if found_keypoint:
                    self._dragged_keypoint_info = found_keypoint
            else:
                # Ensure they are placing the point ON the image, not in the padded panning void
                if self._pixmap_item.boundingRect().contains(scene_pos):
                    self.point_placed.emit(scene_pos)
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._dragged_keypoint_info and (ev.buttons() & self._point_button):
            scene_pos = self.mapToScene(ev.position().toPoint())
            identity_id, part_name = self._dragged_keypoint_info
            self.point_dragged.emit(identity_id, part_name, scene_pos)
            return
            
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == self._point_button:
            self._dragged_keypoint_info = None
        super().mouseReleaseEvent(ev)

    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)
        if not self._pixmap_item:
            return

        # DEV_NOTE: By saving the painter and resetting the transform, we can draw 
        # using pure viewport (screen) coordinates. This entirely prevents Windows 
        # DirectWrite font rendering bugs and ensures lines/points are always a crisp, 
        # constant pixel size regardless of the zoom level.
        painter.save()
        painter.resetTransform()

        if self._primer_points:
            self._draw_points_set(painter, self._primer_points, "primer")

        self._draw_points_set(painter, self._prediction_points, "solid")

        if self._show_gt_points:
            self._draw_points_set(painter, self._gt_points, "hollow")
            
        painter.restore()

    def _draw_points_set(self, painter: QPainter, identity_points_data, style: str):
        if self._show_names:
            painter.setFont(self._font)

        # Get the transformation matrix from scene (image) to viewport (screen)
        vt = self.viewportTransform()

        for identity_data in identity_points_data:
            color = identity_data.get("color", QColor("white"))
            points = identity_data.get("points", {})
            
            # Map all scene coordinates to screen coordinates
            screen_points = {
                name: vt.map(pos) 
                for name, pos in points.items()
            }

            lines_to_draw = identity_data.get("lines", [])
            if lines_to_draw:
                line_color = QColor(color)
                if style == "hollow":
                    line_color.setAlphaF(0.8)
                    pen_style = Qt.PenStyle.DashLine
                    pen_width = 2.0
                elif style == "primer":
                    line_color.setAlphaF(0.6)
                    pen_style = Qt.PenStyle.DashLine
                    pen_width = 1.5
                else:
                    line_color.setAlphaF(0.5)
                    pen_style = Qt.PenStyle.SolidLine
                    pen_width = 1.0
                
                painter.setPen(QPen(line_color, pen_width, pen_style))

                for p1, p2 in lines_to_draw:
                    # Draw lines between mapped screen coordinates
                    painter.drawLine(vt.map(p1), vt.map(p2))

            painter.setPen(Qt.PenStyle.NoPen)
            
            if style == "hollow":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                pen = QPen(color, 1.5)
                painter.setPen(pen)
                radius = 3.0
            elif style == "primer":
                primer_fill = QColor(color)
                primer_fill.setAlphaF(0.4)
                painter.setBrush(primer_fill)
                pen = QPen(color, 1.0, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                radius = 4.0
            else:
                painter.setBrush(color)
                radius = 4.0
                pen = QPen(color)

            for part_name, screen_pos in screen_points.items():
                painter.drawEllipse(screen_pos, radius, radius)

                if self._show_names:
                    painter.setPen(QColor("white"))
                    text_pos = screen_pos + QPointF(6, 4)
                    painter.drawText(text_pos, part_name)
                    
                    if style in ("hollow", "primer"):
                        painter.setPen(pen)
                    else:
                        painter.setPen(Qt.PenStyle.NoPen)