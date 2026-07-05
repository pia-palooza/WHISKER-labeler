# UPDATE_FILE: whisker/gui/widgets/media_viewer.py
import logging
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QRectF, QTimer, QPointF
from PyQt6.QtGui import QPixmap, QResizeEvent, QPainter, QColor, QPen, QBrush, QImage, QFont
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QStyle,
    QSpinBox, QFormLayout, QDoubleSpinBox, QStyleOptionSlider, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsProxyWidget, QStackedWidget,
    QGraphicsItemGroup, QGraphicsLineItem, QGraphicsEllipseItem,
    QGraphicsSimpleTextItem, QGraphicsRectItem
)

from whisker.gui.constants import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS
from whisker.gui.widgets.info_overlay import InfoOverlay


class ClickableSlider(QSlider):
    """Standard QSlider, but now with 100% more clicking capability."""
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        hit = self.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_Slider, opt, event.pos(), self
        )

        if hit == QStyle.SubControl.SC_SliderGroove:
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
            )
            if self.orientation() == Qt.Orientation.Horizontal:
                ratio = (event.position().x() - groove.x()) / groove.width()
            else:
                ratio = (groove.height() - (event.position().y() - groove.y())) / groove.height()

            ratio = max(0.0, min(1.0, ratio))
            val = self.minimum() + (self.maximum() - self.minimum()) * ratio
            self.setValue(int(val))
            self.sliderMoved.emit(int(val))
            event.accept()
        else:
            super().mousePressEvent(event)


class OverlayPainter:
    """The artist of the group. Draws skeletons and boxes."""
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self._prediction_points: List[Dict] = []
        self._gt_points: List[Dict] = []
        self._detection_boxes: List[Dict] = []
        self._show_gt_poses = True
        self._show_keypoint_names = False
        
        # New: Offset for side-by-side view
        self._gt_offset_x = 0.0

        self._pose_group = self._create_group(2)
        self._detection_group = self._create_group(3)

    def _create_group(self, z_val):
        g = QGraphicsItemGroup()
        g.setZValue(z_val)
        self.scene.addItem(g)
        return g

    def update_pose_data(self, prediction: List[Dict], gt: List[Dict]):
        self._prediction_points, self._gt_points = prediction or [], gt or []
        self._redraw_poses()

    def update_detection_data(self, boxes: List[Dict]):
        self._detection_boxes = boxes or []
        self._redraw_detections()

    def set_pose_visibility(self, show_gt: bool, show_names: bool):
        if self._show_gt_poses != show_gt or self._show_keypoint_names != show_names:
            self._show_gt_poses, self._show_keypoint_names = show_gt, show_names
            self._redraw_poses()
            
    def set_gt_offset(self, offset_x: float):
        """Sets the horizontal offset for drawing Ground Truth poses."""
        if self._gt_offset_x != offset_x:
            self._gt_offset_x = offset_x
            self._redraw_poses()

    def clear(self):
        self._prediction_points = []
        self._gt_points = []
        self._detection_boxes = []
        self._clear_group(self._pose_group)
        self._clear_group(self._detection_group)

    def _clear_group(self, group: QGraphicsItemGroup):
        # Nuke it from orbit to prevent ghosting
        self.scene.removeItem(group)
        new_group = self._create_group(group.zValue())
        if group == self._pose_group: self._pose_group = new_group
        else: self._detection_group = new_group

    def _redraw_poses(self):
        self._clear_group(self._pose_group)
        # Predictions always at x=0
        self._draw_points(self._prediction_points, "solid", offset_x=0.0)
        
        if self._show_gt_poses:
            # GT drawn at offset
            gt_style = "solid" if self._gt_offset_x > 0.0 else "hollow"
            self._draw_points(self._gt_points, gt_style, offset_x=self._gt_offset_x)

    def _draw_points(self, data: List[Dict], style: str, offset_x: float = 0.0):
        for identity in data:
            color = identity.get("color", QColor("white"))
            
            # Lines
            if lines := identity.get("lines", []):
                lc = QColor(color)
                lc.setAlphaF(0.5)
                pen = QPen(lc, 2 if style == "hollow" else 1, 
                           Qt.PenStyle.DashLine if style == "hollow" else Qt.PenStyle.SolidLine)
                for p1, p2 in lines:
                    item = QGraphicsLineItem(p1.x() + offset_x, p1.y(), p2.x() + offset_x, p2.y())
                    item.setPen(pen)
                    self._pose_group.addToGroup(item)

            # Points
            brush = QBrush(Qt.BrushStyle.NoBrush if style == "hollow" else color)
            pen = QPen(color, 1.5) if style == "hollow" else QPen(Qt.PenStyle.NoPen)
            r = 2.5 if style == "hollow" else 3.0

            for name, pt in identity.get("points", {}).items():
                ell = QGraphicsEllipseItem(pt.x() + offset_x - r, pt.y() - r, r*2, r*2)
                ell.setPen(pen)
                ell.setBrush(brush)
                self._pose_group.addToGroup(ell)
                
                if self._show_keypoint_names:
                    txt = QGraphicsSimpleTextItem(name)
                    txt.setPos(pt.x() + offset_x + 5, pt.y() - 8)
                    txt.setBrush(QBrush(QColor("white")))
                    self._pose_group.addToGroup(txt)

    def _redraw_detections(self):
        self._clear_group(self._detection_group)
        for box in self._detection_boxes:
            color = QColor(box.get('color', '#00FF00'))
            style_str = box.get('style', 'solid')
            
            pen_style = Qt.PenStyle.DashLine if style_str == 'dashed' else Qt.PenStyle.SolidLine
            pen = QPen(color, 2, pen_style)
            
            rect = QGraphicsRectItem(box['x'], box['y'], box['w'], box['h'])
            rect.setPen(pen)
            
            if 'fill_color' in box:
                rect.setBrush(QBrush(QColor(box['fill_color'])))
            elif 'alpha' in box:
                brush_color = QColor(color)
                brush_color.setAlphaF(box['alpha'])
                rect.setBrush(QBrush(brush_color))
                
            self._detection_group.addToGroup(rect)
            
            if lbl := box.get('label'):
                txt = QGraphicsSimpleTextItem(lbl)
                txt.setBrush(QBrush(color))
                txt.setPos(box['x'], box['y'] - 15)
                self._detection_group.addToGroup(txt)

class MediaControls(QWidget):
    """
    The dashboard. Buttons, sliders, and spinboxes live here.
    """
    # User Intent Signals
    play_toggled = pyqtSignal()
    seek_requested = pyqtSignal(int)      # ms
    step_requested = pyqtSignal(int)      # +1 or -1 frames
    skip_requested = pyqtSignal(int)      # +/- N frames
    goto_frame_requested = pyqtSignal(int)
    speed_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 2, 20, 5)

        # Slider (Wrapped in layout for alignment)
        self.slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.seek_requested.emit)
        
        slider_row = QHBoxLayout()
        # Effectively 120px left, 80px right padding (accounting for main layout 5/20 margins)
        slider_row.setContentsMargins(115, 0, 60, 0)
        slider_row.addWidget(self.slider)
        layout.addLayout(slider_row)

        # Controls Row
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(15)
        
        # Buttons
        btns = QHBoxLayout()
        btns.setSpacing(2)
        
        self.btn_step_back = self._icon_btn(QStyle.StandardPixmap.SP_MediaSeekBackward, "Step Back")
        self.btn_step_back.clicked.connect(lambda: self.step_requested.emit(-1))
        
        self.btn_play = self._icon_btn(QStyle.StandardPixmap.SP_MediaPlay, "Play/Pause")
        self.btn_play.clicked.connect(self.play_toggled.emit)
        
        self.btn_step_fwd = self._icon_btn(QStyle.StandardPixmap.SP_MediaSeekForward, "Step Forward")
        self.btn_step_fwd.clicked.connect(lambda: self.step_requested.emit(1))
        
        btns.addWidget(self.btn_step_back)
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_step_fwd)

        btns.addSpacing(10)
        
        self.btn_skip_back = QPushButton("<<")
        self.btn_skip_back.clicked.connect(lambda: self.skip_requested.emit(-self.spin_step.value()))
        
        self.btn_skip_fwd = QPushButton(">>")
        self.btn_skip_fwd.clicked.connect(lambda: self.skip_requested.emit(self.spin_step.value()))
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 10000)
        self.spin_step.setValue(10)
        self.spin_step.setToolTip("Frame skip size")

        btns.addWidget(self.btn_skip_back)
        btns.addWidget(self.btn_skip_fwd)
        btns.addWidget(self.spin_step)
        
        ctrl_layout.addLayout(btns)
        ctrl_layout.addStretch()

        # Info Labels
        info = QVBoxLayout()
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_frame = QLabel("Frame: 0 / 0")
        info.addWidget(self.lbl_time)
        info.addWidget(self.lbl_frame)
        ctrl_layout.addLayout(info)

        # Right Side Inputs
        form = QFormLayout()
        
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 8.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.valueChanged.connect(self.speed_changed.emit)
        form.addRow("Speed:", self.spin_speed)

        goto_box = QHBoxLayout()
        self.spin_goto = QSpinBox()
        self.btn_goto = QPushButton("Go")
        self.btn_goto.clicked.connect(lambda: self.goto_frame_requested.emit(self.spin_goto.value()))
        goto_box.addWidget(self.spin_goto)
        goto_box.addWidget(self.btn_goto)
        form.addRow("Go to:", goto_box)
        
        ctrl_layout.addLayout(form)
        layout.addLayout(ctrl_layout)

        # Store widgets to enable/disable easily
        self._interactive_widgets = [
            self.slider, self.btn_step_back, self.btn_step_fwd, self.btn_play,
            self.btn_skip_back, self.btn_skip_fwd, self.spin_step, 
            self.spin_speed, self.spin_goto, self.btn_goto
        ]

    def _icon_btn(self, icon, tip):
        b = QPushButton()
        b.setIcon(self.style().standardIcon(icon))
        b.setToolTip(tip)
        return b

    def set_controls_enabled(self, enabled: bool):
        for w in self._interactive_widgets:
            w.setEnabled(enabled)
        if not enabled:
            self.lbl_time.setText("00:00 / 00:00")
            self.lbl_frame.setText("Frame: 0 / 0")
        self.setVisible(enabled)

    def set_playing_state(self, is_playing: bool):
        icon = QStyle.StandardPixmap.SP_MediaPause if is_playing else QStyle.StandardPixmap.SP_MediaPlay
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def update_time(self, current_ms: int, duration_ms: int):
        self.lbl_time.setText(f"{self._fmt(current_ms)} / {self._fmt(duration_ms)}")
        
        # Avoid slider feedback loop if user is dragging it
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(current_ms)
            self.slider.blockSignals(False)

    def update_frames(self, current_f: int, total_f: int):
        self.lbl_frame.setText(f"Frame: {current_f} / {max(0, total_f - 1)}")

    def set_duration(self, ms: int):
        self.slider.setRange(0, ms)

    def set_frame_limit(self, total_frames: int):
        self.spin_goto.setRange(0, max(0, total_frames - 1))

    def _fmt(self, ms: int):
        s = ms // 1000
        return f"{s // 60:02}:{s % 60:02}"


class VideoGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._left_name = ""
        self._right_name = ""
        self._side_by_side = False
        self._show_overlay_names = True

    def set_overlay_names(self, left: str, right: str):
        self._left_name = left
        self._right_name = right
        self.viewport().update()

    def set_side_by_side(self, enabled: bool):
        self._side_by_side = enabled
        self.viewport().update()

    def set_show_overlay_names(self, show: bool):
        self._show_overlay_names = show
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)
        if not self._side_by_side or not self._show_overlay_names:
            return

        # Get scene height and width of a single video frame.
        # The scene Rect has width w*2 in side-by-side.
        scene_rect = self.scene().sceneRect()
        w = scene_rect.width() / 2.0
        h = scene_rect.height()

        painter.save()
        painter.resetTransform()

        # Draw left name at scene QPointF(0, h)
        self._draw_overlay_label(painter, self._left_name, QPointF(0, h))
        # Draw right name at scene QPointF(w, h)
        self._draw_overlay_label(painter, self._right_name, QPointF(w, h))

        painter.restore()

    def _draw_overlay_label(self, painter: QPainter, text: str, bottom_left_scene: QPointF):
        if not text:
            return
        
        vt = self.viewportTransform()
        screen_pt = vt.map(bottom_left_scene)
        
        x = screen_pt.x() + 10
        y = screen_pt.y() - 10
        
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()
        
        rect = QRectF(x - 6, y - text_height - 4, text_width + 12, text_height + 8)
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(rect, 4.0, 4.0)
        
        painter.setPen(QColor("white"))
        painter.drawText(QPointF(x, y - 3), text)


class MediaViewerWidget(QWidget):
    """
    The Boss. Orchestrates the Player, the Painter, and the Controls.
    """
    frame_changed = pyqtSignal(int)
    DEFAULT_OVERLAY_FONT_SIZE = InfoOverlay.DEFAULT_FONT_SIZE

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._media_player = QMediaPlayer()
        self._fps = 0.0
        self._total_frames = 0
        self._duration_ms = 0
        self._is_video_loaded = False
        
        # Side-by-side mode state
        self._side_by_side = False
        self._cap: Optional[cv2.VideoCapture] = None  # Persistent capture for side-by-side

        self._setup_ui()
        self.set_media(None)
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.view_stack = QStackedWidget()
        layout.addWidget(self.view_stack)

        # Placeholder
        self.placeholder = QLabel("Select a media file to view")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_stack.addWidget(self.placeholder)

        # Main Player Container
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        
        # Graphics View
        self.scene = QGraphicsScene(self)
        self.view = VideoGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("border: none; background: #222;")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Scene Items
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        
        # Primary Image Item (used for images OR video when we need to grab frames manually)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        
        # Secondary Image Item (for side-by-side split view)
        self.image_item_right = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item_right)
        self.image_item_right.setVisible(False)

        # Overlay
        self.info_overlay = InfoOverlay()
        self.overlay_proxy = QGraphicsProxyWidget()
        self.overlay_proxy.setWidget(self.info_overlay)
        self.overlay_proxy.setZValue(1)
        self.scene.addItem(self.overlay_proxy)

        # Helpers
        self.painter = OverlayPainter(self.scene)
        self.controls = MediaControls()

        vbox.addWidget(self.view, 1)
        vbox.addWidget(self.controls)
        
        self.view_stack.addWidget(container)

    def _connect_signals(self):
        # Player -> Self
        self._media_player.positionChanged.connect(self._on_position_changed)
        self._media_player.durationChanged.connect(self._on_duration_changed)
        self._media_player.errorOccurred.connect(self._handle_error)
        self._media_player.mediaStatusChanged.connect(self._handle_status)
        self._media_player.playbackStateChanged.connect(self._handle_state)
        
        # Connect to VideoSink (QGraphicsVideoItem)
        self._media_player.setVideoOutput(self.video_item)

        # Controls -> Self
        self.controls.play_toggled.connect(self.toggle_play_pause)
        self.controls.seek_requested.connect(self._seek_ms)
        self.controls.step_requested.connect(self._step_frames)
        self.controls.skip_requested.connect(self._step_frames) # Reuse logic
        self.controls.goto_frame_requested.connect(self.seek_to_frame)
        self.controls.speed_changed.connect(self._media_player.setPlaybackRate)

    # --- API ---

    def set_media(self, file_path: Optional[Path]):
        self._media_player.stop()
        
        # Cleanup persistent capture
        if self._cap:
            self._cap.release()
            self._cap = None

        self.painter.clear()
        self.video_item.setVisible(False)
        self.image_item.setVisible(False)
        self.image_item_right.setVisible(False)
        self._is_video_loaded = False

        if not file_path or not file_path.is_file():
            self.controls.set_controls_enabled(False)
            self.view_stack.setCurrentWidget(self.placeholder)
            return

        self.view_stack.setCurrentWidget(self.view.parent())
        suffix = file_path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            self.controls.set_controls_enabled(False)
            self._load_image(file_path)
        elif suffix in VIDEO_EXTENSIONS:
            self.controls.set_controls_enabled(True)
            self._load_video(file_path)
        else:
            self.view_stack.setCurrentWidget(self.placeholder)
            self.placeholder.setText(f"Unsupported: {suffix}")

        if self._side_by_side:
            self.set_side_by_side(True, force=True)

    def set_overlay_names(self, left: str, right: str):
        self.view.set_overlay_names(left, right)

    def set_show_overlay_names(self, show: bool):
        self.view.set_show_overlay_names(show)

    def set_side_by_side(self, enabled: bool, force: bool = False):
        """
        Enables/Disables side-by-side comparison mode.
        If enabled, the canvas doubles in width and shows GT on the right.
        """
        if self._side_by_side == enabled and not force:
            return
            
        self._side_by_side = enabled
        self.view.set_side_by_side(enabled)
        
        logging.info(f"MediaViewer updating side-by-side mode: {enabled=}, {self._is_video_loaded=}")
        if enabled:
            # Switch Logic:
            # 1. Hide the video streaming item (we can't easily duplicate the stream object).
            # 2. We will manually grab the current frame from the video item and paint it
            #    onto the two image items (Left/Right) whenever the frame updates.
            self.video_item.setVisible(False)
            self.image_item.setVisible(True)
            self.image_item_right.setVisible(True)
            
            # Force a frame update to populate the pixmaps
            self._update_side_by_side_frame()
            
        else:
            # Revert logic
            self.image_item_right.setVisible(False)
            
            if self._is_video_loaded:
                self.video_item.setVisible(True)
                self.image_item.setVisible(False)
            else:
                # It's a static image
                self.image_item.setVisible(True)
                
            self.painter.set_gt_offset(0.0)

        # Refit the view to the new scene bounds
        self._fit_view()

    def _update_side_by_side_frame(self):
        """
        Grabs the current frame from the persistent capture object and duplicates it.
        This allows us to draw different overlays on each copy.
        """
        if not self._side_by_side:
            return

        pix = None
        
        # Source 1: If it's a video, grab using the persistent capture object
        if self._is_video_loaded and self._cap and self._cap.isOpened():
            f_idx = self.get_current_frame()
            if f_idx >= 0:
                # Only seek if necessary (though for videos, we often need to)
                # Optimization: check CAP_PROP_POS_FRAMES before setting
                # (Some backends support this check, others don't, but it's worth a try)
                current_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                if int(current_pos) != f_idx:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                
                ret, frame = self._cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    # Must keep a reference or copy data? QImage usually needs data to persist.
                    # Creating QImage from data copies it if we don't pass a cleanup function,
                    # but effectively for this local scope, we need it to survive until setPixmap.
                    q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q_img)
        
        # Source 2: For static image:
        elif self.image_item.pixmap():
            pix = self.image_item.pixmap()

        if pix:
            self.image_item.setPixmap(pix)
            self.image_item_right.setPixmap(pix)
            
            w = pix.width()
            self.image_item_right.setPos(w, 0)
            
            # Tell painter to offset GT by width
            self.painter.set_gt_offset(float(w))
            
            # Expand Scene
            self.scene.setSceneRect(0, 0, w * 2, pix.height())

    def _load_image(self, path: Path):
        pix = QPixmap(str(path))
        if not pix.isNull():
            self.image_item.setPixmap(pix)
            self.scene.setSceneRect(self.image_item.boundingRect())
            self.image_item.setVisible(True)
            self._fit_view()

    def _load_video(self, path: Path):
        # 1. Open persistent capture for metadata and side-by-side mode
        self._cap = cv2.VideoCapture(str(path))
        
        if self._cap.isOpened():
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if self._fps > 0:
                self._is_video_loaded = True
                self.video_item.setSize(QRectF(0, 0, w, h).size())
                self.scene.setSceneRect(self.video_item.boundingRect())
                self.video_item.setVisible(True)
                self.controls.set_frame_limit(self._total_frames)
                
                # 2. Setup QMediaPlayer for standard playback
                self._media_player.setSource(QUrl.fromLocalFile(str(path)))
                return

        logging.warning(f"Failed to load video: {path.name}")
        self.placeholder.setText(f"Error loading video metadata.")
        if self._cap:
            self._cap.release()
            self._cap = None

    # --- Playback Logic ---

    def toggle_play_pause(self):
        if self._media_player.source().isEmpty(): return
        
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.pause()
        else:
            # Auto-replay if near end
            if self._media_player.position() >= self._duration_ms - 100:
                self._media_player.setPosition(0)
            self._media_player.play()

    def _seek_ms(self, ms: int):
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.pause()
        self._media_player.setPosition(ms)
        
        # Trigger manual frame update for side-by-side
        if self._side_by_side and self._is_video_loaded:
             self._update_side_by_side_frame()

        if self._is_video_loaded:
            self.frame_changed.emit(self._ms_to_frame(ms))

    def _step_frames(self, delta_frames: int):
        if not self._is_video_loaded: return
        self.seek_to_frame(self.get_current_frame() + delta_frames)

    def seek_to_frame(self, frame_no: int):
        if not self._is_video_loaded: return
        target = max(0, min(frame_no, self._total_frames - 1))
        self._seek_ms(self._frame_to_ms(target))

    # --- Event Handlers ---

    def _on_position_changed(self, ms: int):
        self.controls.update_time(ms, self._duration_ms)
        if self._is_video_loaded:
            frame = self._ms_to_frame(ms)
            self.controls.update_frames(frame, self._total_frames)
            
            # If playing in side-by-side mode, update the static frame capture
            if self._side_by_side and self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                 self._update_side_by_side_frame()
            
            self.frame_changed.emit(frame)

    def _on_duration_changed(self, ms: int):
        self._duration_ms = ms
        self.controls.set_duration(ms)

    def _handle_state(self, state):
        self.controls.set_playing_state(state == QMediaPlayer.PlaybackState.PlayingState)

    def _handle_status(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            # Kickstart video to show the first frame
            QTimer.singleShot(0, self._kickstart_video)

        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._media_player.setPosition(self._duration_ms)
            self._media_player.pause()

    def _kickstart_video(self):
        """Helper to safely initialize video state after loading."""
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
            self._media_player.pause()
            self._media_player.setPosition(0)
        self._fit_view()

    def _handle_error(self, _, err_str):
        logging.error(f"Player Error: {err_str}")
        self.set_media(None)
        self.placeholder.setText(f"Playback Error: {err_str}")

    def resizeEvent(self, event: Optional[QResizeEvent]):
        super().resizeEvent(event)
        self._fit_view()

    def _fit_view(self):
        # 1. Identify which item we are actually looking at
        target_item = None
        
        if self._side_by_side and self.image_item.pixmap():
            # In side-by-side, the scene bounds are manually managed 
            # to be [0, 0, w*2, h]. We use the sceneRect itself.
            rect = self.scene.sceneRect()
            self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.overlay_proxy.setGeometry(rect)
            return

        if self.video_item.isVisible():
            target_item = self.video_item
        elif self.image_item.isVisible():
            target_item = self.image_item
        
        # 2. Fit the view to that specific item
        if target_item and not target_item.boundingRect().isEmpty():
            rect = target_item.boundingRect()
            self.view.fitInView(target_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.scene.setSceneRect(rect)
            self.overlay_proxy.setGeometry(rect)

    # --- Converters & Accessors ---

    def _ms_to_frame(self, ms: int) -> int:
        return min(round((ms / 1000.0) * self._fps), self._total_frames - 1) if self._fps else 0

    def _frame_to_ms(self, frame: int) -> int:
        return int((frame / self._fps) * 1000) if self._fps else 0

    def get_current_frame(self) -> int:
        return self._ms_to_frame(self._media_player.position()) if self._is_video_loaded else -1
    
    # Delegated Passthroughs for external use
    def set_overlay_visible(self, v): self.info_overlay.setVisible(v)
    def set_overlay_text(self, t): self.info_overlay.set_text(t)
    def set_overlay_font_size(self, s): self.info_overlay.set_font_size(s)
    def set_pose_data(self, p, g): self.painter.update_pose_data(p, g)
    def set_detection_data(self, b): self.painter.update_detection_data(b)
    def set_show_gt_poses(self, s): self.painter.set_pose_visibility(s, self.painter._show_keypoint_names)
    def set_show_keypoint_names(self, s): self.painter.set_pose_visibility(self.painter._show_gt_poses, s)
    @property
    def fps(self): return self._fps
    @property
    def total_frames(self): return self._total_frames
    @property
    def video_width(self):
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        return 0.0
    @property
    def video_height(self):
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return 0.0