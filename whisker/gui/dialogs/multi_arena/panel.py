import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QGraphicsScene,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QFileDialog,
    QToolButton,
    QSlider,
)

from whisker.core.study.dataset import VIDEO_FILE_EXTENSIONS
from whisker.gui.dialogs.multi_arena.arena_box_item import ArenaBoxItem
from whisker.gui.dialogs.multi_arena.canvas_view import MultiArenaCanvasView

logger = logging.getLogger(__name__)

# A small palette so adjacent arena boxes are visually distinguishable.
_BOX_COLORS = [
    "#00E5A0", "#FF6B6B", "#4D96FF", "#FFD93D",
    "#C86BFF", "#FF9F45", "#6BCB77", "#F06595",
]

_DEFAULT_BOX_W = 320
_DEFAULT_BOX_H = 320


class MultiArenaDatasetPanel(QWidget):
    """
    Placement editor for a multi-arena dataset (Phase 2 — in-memory only).

    The user picks a source folder of videos, defines one shared arena box size
    (W x H) for the whole dataset, then places one box over each arena on each
    video's reference frame. Only the box *size* is shared across the dataset;
    each placement contributes an (x, y) position, and the number/positions of
    boxes vary per video.

    All state is held in memory; nothing is written and no dataset is created in
    this phase. ``get_multi_arena_info()`` exposes the current configuration for
    later persistence/wiring.
    """

    config_changed = pyqtSignal()
    create_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # --- In-memory model ---
        self._folder_path: Optional[Path] = None
        self._box_w: int = _DEFAULT_BOX_W
        self._box_h: int = _DEFAULT_BOX_H
        # video_rel_path (forward slashes) -> list of (x, y) top-left positions
        self._placements: Dict[str, List[Tuple[int, int]]] = {}
        self._current_video: Optional[str] = None

        # --- Scene / frame I/O state ---
        self._current_video_path: Optional[Path] = None
        self._total_frames: int = 0
        self._background_item = None
        self._box_items: List[ArenaBoxItem] = []

        # True when editing an existing dataset (name is locked; the action
        # button reads "Save Changes" instead of "Create ...").
        self._edit_mode: bool = False

        # Combo-box status icons: green check for videos that have arenas drawn,
        # a same-sized blank placeholder for those that don't (keeps text aligned).
        self._done_icon, self._empty_icon = self._build_indicator_icons()

        self._build_ui()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_canvas())
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Bottom action row (mirrors the single-arena dialog's OK/Cancel).
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_requested)
        self.create_btn = QPushButton("Create Multi-Arena Dataset")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self.create_requested)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.create_btn)
        root.addLayout(btn_row)

        self.config_changed.connect(self._update_create_enabled)
        self.config_changed.connect(self._update_video_indicators)

        # Disabled until a folder with videos is selected (needs frame_slider
        # from the canvas, so run after both panels are built).
        self._set_placement_enabled(False)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Name + folder
        setup_group = QGroupBox("Dataset")
        setup_form = QFormLayout(setup_group)
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(lambda _: self.config_changed.emit())
        setup_form.addRow("Dataset Name:", self.name_edit)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel("<i>No folder selected...</i>")
        self.folder_label.setWordWrap(True)
        select_folder_btn = QPushButton("Select Folder...")
        select_folder_btn.clicked.connect(self._select_folder)
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(select_folder_btn)
        setup_form.addRow("Source Folder:", folder_row)
        layout.addWidget(setup_group)

        # Shared arena box size
        box_group = QGroupBox("Arena Box Size (shared by whole dataset)")
        box_form = QFormLayout(box_group)
        self.box_w_spin = QSpinBox()
        self.box_w_spin.setRange(16, 8192)
        self.box_w_spin.setValue(self._box_w)
        self.box_h_spin = QSpinBox()
        self.box_h_spin.setRange(16, 8192)
        self.box_h_spin.setValue(self._box_h)
        self.box_w_spin.valueChanged.connect(self._on_box_size_changed)
        self.box_h_spin.valueChanged.connect(self._on_box_size_changed)
        box_form.addRow("Width (px):", self.box_w_spin)
        box_form.addRow("Height (px):", self.box_h_spin)
        layout.addWidget(box_group)

        # Per-video placement
        place_group = QGroupBox("Arena Placement (per video)")
        place_layout = QVBoxLayout(place_group)

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video:"))
        self.video_combo = QComboBox()
        self.video_combo.currentIndexChanged.connect(self._on_video_changed)
        video_row.addWidget(self.video_combo, 1)
        place_layout.addLayout(video_row)

        btn_row = QHBoxLayout()
        self.add_box_btn = QPushButton("Add Arena Box")
        self.add_box_btn.clicked.connect(self._add_box)
        self.remove_box_btn = QPushButton("Remove Selected")
        self.remove_box_btn.clicked.connect(self.remove_selected_boxes)
        self.clear_boxes_btn = QPushButton("Clear This Video")
        self.clear_boxes_btn.clicked.connect(self.clear_current_video_boxes)
        btn_row.addWidget(self.add_box_btn)
        btn_row.addWidget(self.remove_box_btn)
        btn_row.addWidget(self.clear_boxes_btn)
        place_layout.addLayout(btn_row)

        self.box_count_label = QLabel("Boxes on this video: 0")
        place_layout.addWidget(self.box_count_label)

        hint = QLabel(
            "<i>Drag boxes to position; arrow keys nudge the selected box "
            "(Shift = coarse). Ctrl+wheel zooms.</i>"
        )
        hint.setWordWrap(True)
        place_layout.addWidget(hint)

        layout.addWidget(place_group)
        layout.addStretch()

        return panel

    def _build_canvas(self) -> QWidget:
        container = QWidget()
        canvas_layout = QVBoxLayout(container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.scene = QGraphicsScene()
        self.view = MultiArenaCanvasView(self.scene)
        self.view.setStyleSheet("background-color: #222222;")
        canvas_layout.addWidget(self.view, 1)

        player_row = QHBoxLayout()
        player_row.addWidget(QLabel("Reference Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)
        player_row.addWidget(self.frame_slider, 1)
        self.frame_label = QLabel("Frame: 0/0")
        player_row.addWidget(self.frame_label)

        player_row.addSpacing(12)
        player_row.addWidget(QLabel("Zoom:"))
        zoom_out = QToolButton()
        zoom_out.setText("−")
        zoom_out.clicked.connect(lambda: self.view.zoom_step(0.8))
        zoom_in = QToolButton()
        zoom_in.setText("+")
        zoom_in.clicked.connect(lambda: self.view.zoom_step(1.25))
        zoom_fit = QToolButton()
        zoom_fit.setText("Fit")
        zoom_fit.clicked.connect(self.view.fit_to_scene)
        player_row.addWidget(zoom_out)
        player_row.addWidget(zoom_in)
        player_row.addWidget(zoom_fit)
        canvas_layout.addLayout(player_row)

        return container

    def _set_placement_enabled(self, enabled: bool):
        for w in (self.video_combo, self.add_box_btn, self.remove_box_btn,
                  self.clear_boxes_btn, self.frame_slider):
            w.setEnabled(enabled)

    # ==================================================================
    # Folder / video selection
    # ==================================================================
    def _select_folder(self):
        start_dir = str(self._folder_path) if self._folder_path else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder", start_dir)
        if not folder:
            return
        self.set_source_folder(Path(folder))

    def set_source_folder(self, folder_path: Path):
        """Scans a folder for videos and resets placement state. Public so tests
        (and later persistence) can drive it without a file dialog."""
        video_files: List[str] = []
        for ext in VIDEO_FILE_EXTENSIONS:
            for p in folder_path.rglob(f"*{ext}"):
                rel = str(p.relative_to(folder_path)).replace("\\", "/")
                video_files.append(rel)
        video_files = sorted(set(video_files))

        self._folder_path = folder_path
        self._placements = {}
        self._current_video = None
        self._clear_box_items()

        self.folder_label.setText(f"<b>{folder_path.name}</b> ({len(video_files)} videos)")
        if not self.name_edit.text().strip():
            self.name_edit.setText(folder_path.name)

        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for rel in video_files:
            self.video_combo.addItem(rel, rel)
        self.video_combo.blockSignals(False)

        has_videos = bool(video_files)
        self._set_placement_enabled(has_videos)
        if has_videos:
            self.video_combo.setCurrentIndex(0)
            self._on_video_changed(0)
        self.config_changed.emit()

    def load_existing(
        self,
        name: str,
        folder_path: Path,
        box_width: int,
        box_height: int,
        placements: Dict[str, List[Tuple[int, int]]],
    ):
        """Load an existing multi-arena config for editing.

        Points the editor at the dataset's original source folder (so video
        relative paths match the stored placement keys), pre-loads the shared
        box size and every placement, and locks the dataset name — renaming
        would create a new dataset directory and orphan all existing per-arena
        artifacts."""
        self._edit_mode = True

        # Scans the folder (this resets placements to {} and selects video 0).
        self.set_source_folder(Path(folder_path))

        self.name_edit.setText(name)
        self.name_edit.setEnabled(False)
        self.name_edit.setToolTip("Renaming is disabled while editing an existing dataset.")

        self.box_w_spin.blockSignals(True)
        self.box_h_spin.blockSignals(True)
        self.box_w_spin.setValue(int(box_width))
        self.box_h_spin.setValue(int(box_height))
        self.box_w_spin.blockSignals(False)
        self.box_h_spin.blockSignals(False)
        self._box_w = int(box_width)
        self._box_h = int(box_height)

        # Inject the saved placements, then redraw the currently-selected video's
        # boxes. We bypass _on_video_changed's leading sync so the empty scene
        # doesn't wipe the placements we just set.
        self._placements = {
            v: [(int(x), int(y)) for (x, y) in pts]
            for v, pts in placements.items() if pts
        }
        self._clear_box_items()
        if self._current_video:
            self._load_placements_for(self._current_video)
        self._update_box_count()

        self.create_btn.setText("Save Changes")
        self.config_changed.emit()

    def _on_video_changed(self, _index: int):
        # Persist the video we're leaving, then load the newly selected one.
        self._sync_current_placements()
        self._clear_box_items()

        new_video = self.video_combo.currentData()
        self._current_video = new_video
        if not new_video or not self._folder_path:
            return

        self._load_video_background(self._folder_path / new_video)
        self._load_placements_for(new_video)
        self._update_box_count()

    # ==================================================================
    # Reference-frame background (mirrors ROI panel)
    # ==================================================================
    def _load_video_background(self, video_path: Path):
        self._current_video_path = video_path
        self._background_item = None

        cap = cv2.VideoCapture(str(video_path))
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self._total_frames > 0:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, self._total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f"Frame: 0/{self._total_frames}")
            self._load_frame_at_index(0)
            self.view.fit_to_scene()
        else:
            self.frame_slider.setRange(0, 0)
            self.frame_label.setText("Frame: 0/0")

    def _load_frame_at_index(self, index: int):
        if not self._current_video_path or not self._current_video_path.exists():
            return

        cap = cv2.VideoCapture(str(self._current_video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return

        if self._background_item is not None:
            try:
                self.scene.removeItem(self._background_item)
            except Exception:
                pass

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        q_img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.scene.setSceneRect(0, 0, w, h)
        self._background_item = self.scene.addPixmap(pixmap)
        self._background_item.setZValue(-100)  # behind the boxes

    def _on_slider_value_changed(self, value: int):
        self.frame_label.setText(f"Frame: {value}/{self._total_frames}")
        self._load_frame_at_index(value)

    # ==================================================================
    # Box management
    # ==================================================================
    def add_box_at(self, x: float, y: float):
        """Adds an arena box with its top-left at scene position (x, y)."""
        color = _BOX_COLORS[len(self._box_items) % len(_BOX_COLORS)]
        item = ArenaBoxItem(self._box_w, self._box_h, color, index=len(self._box_items) + 1)
        self.scene.addItem(item)
        item.setPos(float(x), float(y))  # itemChange clamps into the frame
        self._box_items.append(item)
        self._update_box_count()
        self.config_changed.emit()
        return item

    def _add_box(self):
        rect = self.scene.sceneRect()
        if rect.isNull():
            return
        # Spawn near the center, cascading slightly so stacked boxes are visible.
        n = len(self._box_items)
        cx = rect.width() / 2.0 - self._box_w / 2.0 + (n % 5) * 20
        cy = rect.height() / 2.0 - self._box_h / 2.0 + (n % 5) * 20
        self.add_box_at(cx, cy)

    def remove_selected_boxes(self):
        selected = [it for it in self._box_items if it.isSelected()]
        if not selected:
            return
        for item in selected:
            self.scene.removeItem(item)
            self._box_items.remove(item)
        self._reindex_boxes()
        self._update_box_count()
        self.config_changed.emit()

    def clear_current_video_boxes(self):
        if not self._box_items:
            return
        self._clear_box_items()
        self._update_box_count()
        self.config_changed.emit()

    def _clear_box_items(self):
        for item in self._box_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self._box_items = []

    def _reindex_boxes(self):
        for i, item in enumerate(self._box_items):
            item.index = i + 1
            item.update()

    def _load_placements_for(self, video_rel: str):
        for (x, y) in self._placements.get(video_rel, []):
            self.add_box_at(x, y)

    def _on_box_size_changed(self, _value: int):
        self._box_w = self.box_w_spin.value()
        self._box_h = self.box_h_spin.value()
        for item in self._box_items:
            item.set_box_size(self._box_w, self._box_h)
        self.config_changed.emit()

    def _update_box_count(self):
        self.box_count_label.setText(f"Boxes on this video: {len(self._box_items)}")

    # ==================================================================
    # Video-status indicators
    # ==================================================================
    def _build_indicator_icons(self) -> Tuple[QIcon, QIcon]:
        """Builds the two combo-box status icons: a green check for videos that
        have at least one arena box placed, and a same-sized transparent
        placeholder for those that don't (so the video names stay aligned)."""
        size = 16

        done = QPixmap(size, size)
        done.fill(Qt.GlobalColor.transparent)
        painter = QPainter(done)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2ECC71"))
        painter.drawEllipse(1, 1, size - 2, size - 2)
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF([
            QPointF(4.5, 8.5), QPointF(7.0, 11.0), QPointF(11.5, 5.0),
        ]))
        painter.end()

        empty = QPixmap(size, size)
        empty.fill(Qt.GlobalColor.transparent)

        return QIcon(done), QIcon(empty)

    def _update_video_indicators(self):
        """Refreshes the green 'arena drawn' check next to each video in the
        combo box. A video is marked done once it has at least one placed box,
        so it's obvious which videos still need arenas."""
        # Capture the active video's live boxes first so its icon is accurate.
        self._sync_current_placements()
        for i in range(self.video_combo.count()):
            video_rel = self.video_combo.itemData(i)
            has_boxes = bool(self._placements.get(video_rel))
            icon = self._done_icon if has_boxes else self._empty_icon
            self.video_combo.setItemIcon(i, icon)

    # ==================================================================
    # State access
    # ==================================================================
    def _sync_current_placements(self):
        """Reads the current scene's box positions back into the model."""
        if self._current_video is None:
            return
        positions = [
            (int(round(item.pos().x())), int(round(item.pos().y())))
            for item in self._box_items
        ]
        if positions:
            self._placements[self._current_video] = positions
        else:
            self._placements.pop(self._current_video, None)

    def get_placements(self) -> Dict[str, List[Tuple[int, int]]]:
        """Returns a copy of all per-video placements (current video synced)."""
        self._sync_current_placements()
        return {v: list(pts) for v, pts in self._placements.items() if pts}

    def get_multi_arena_info(self) -> Optional[dict]:
        """Current configuration, or None if incomplete (no name / folder / any
        placed box). Coordinates are full-frame-pixel top-left positions."""
        name = self.name_edit.text().strip()
        placements = self.get_placements()
        if not name or not self._folder_path or not placements:
            return None
        return {
            "name": name,
            "folder_path": self._folder_path,
            "box_width": self._box_w,
            "box_height": self._box_h,
            "placements": placements,
        }

    def is_valid(self) -> bool:
        return self.get_multi_arena_info() is not None

    def _update_create_enabled(self):
        self.create_btn.setEnabled(self.is_valid())
