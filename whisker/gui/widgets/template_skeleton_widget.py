import math
from typing import List, Dict, Tuple

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QFormLayout,
)

from whisker.gui.widgets.interactive_image_label import InteractiveImageLabel


class TemplateSkeletonWidget(QWidget):
    """
    A widget for defining a "Template Skeleton" (Standard Pose) and a
    Heading Axis for egocentric alignment.
    """

    CANVAS_SIZE = 500

    accepted = pyqtSignal()
    rejected = pyqtSignal()

    def __init__(
        self,
        body_parts: List[str],
        skeleton: list[tuple[str, str]],
        initial_coords: Dict[str, Tuple[float, float]],
        initial_axis: tuple[str, str] | None,
        parent=None,
    ):
        super().__init__(parent)

        self.body_parts = body_parts
        self.skeleton = skeleton
        self._current_coords = initial_coords.copy()
        self._current_axis = initial_axis

        if not self._current_coords and self.body_parts:
            self._init_default_layout()

        self._init_ui()
        self._update_canvas()

    def _init_default_layout(self):
        cx, cy = self.CANVAS_SIZE / 2, self.CANVAS_SIZE / 2
        radius = 150
        n = len(self.body_parts)
        for i, bp in enumerate(self.body_parts):
            angle = 2 * math.pi * i / n
            x = cx + radius * math.cos(angle - math.pi / 2)
            y = cy + radius * math.sin(angle - math.pi / 2)
            self._current_coords[bp] = (x, y)

    def _init_ui(self):
        # Main layout with stretch on left/right to center the editor content
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(30) # Generous spacing between canvas and controls
        
        layout.addStretch(1) # Left centering spacer
        
        # Left side: Canvas
        canvas_container = QVBoxLayout()
        canvas_container.setSpacing(10)
        
        canvas_title = QLabel("Standard Pose (Drag Points)")
        canvas_title.setStyleSheet("font-size: 11pt; font-weight: bold; color: #4F8A8B;")
        canvas_container.addWidget(canvas_title)
        
        self.canvas_label = InteractiveImageLabel()
        self.canvas_label.setFixedSize(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.canvas_label.setStyleSheet("border: 1px solid #ccc; border-radius: 4px; background-color: #222;")
        
        # Configure interactions specifically for the static template mode
        self.canvas_label.set_interaction_mode(
            pan_zoom=False, point_button=Qt.MouseButton.LeftButton
        )
        
        bg = QPixmap(self.CANVAS_SIZE, self.CANVAS_SIZE)
        bg.fill(QColor("#222222"))
        self.canvas_label.set_full_pixmap(bg)
        self.canvas_label.set_drag_mode(True)
        self.canvas_label.set_show_names(True)
        self.canvas_label.point_dragged.connect(self._on_point_dragged)
        
        canvas_container.addWidget(self.canvas_label)
        canvas_container.addStretch() # Align to top
        layout.addLayout(canvas_container)
        
        # Right side: Controls panel (Fixed width to avoid stretching)
        controls_widget = QWidget()
        controls_widget.setFixedWidth(340)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(15)
        
        header_label = QLabel("Template Skeleton Editor")
        header_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4F8A8B;")
        controls_layout.addWidget(header_label)
        
        instr_group = QGroupBox("Instructions")
        instr_layout = QVBoxLayout(instr_group)
        instr_label = QLabel(
            "1. <b>Drag points</b> on the canvas to match the animal's standard pose "
            "(e.g., fully stretched, facing Up/North).\n\n"
            "2. <b>Select the Heading Axis</b> below. This defines the 'Forward' direction "
            "for egocentric alignment."
        )
        instr_label.setWordWrap(True)
        instr_label.setStyleSheet("font-size: 10pt; line-height: 1.4;")
        instr_layout.addWidget(instr_label)
        controls_layout.addWidget(instr_group)
        
        axis_group = QGroupBox("Heading Axis (Compass)")
        form_layout = QFormLayout(axis_group)
        form_layout.setVerticalSpacing(10)
        form_layout.setHorizontalSpacing(10)
        
        self.axis_from_combo = QComboBox()
        self.axis_from_combo.addItems(self.body_parts)
        self.axis_to_combo = QComboBox()
        self.axis_to_combo.addItems(self.body_parts)
        
        if self._current_axis:
            self.axis_from_combo.setCurrentText(self._current_axis[0])
            self.axis_to_combo.setCurrentText(self._current_axis[1])
        elif len(self.body_parts) >= 2:
            self.axis_to_combo.setCurrentIndex(1)
            
        self.axis_from_combo.currentTextChanged.connect(self._update_canvas)
        self.axis_to_combo.currentTextChanged.connect(self._update_canvas)
        
        form_layout.addRow("From (e.g., Tail):", self.axis_from_combo)
        form_layout.addRow("To (e.g., Neck):", self.axis_to_combo)
        controls_layout.addWidget(axis_group)
        
        reset_btn = QPushButton("Reset Layout")
        reset_btn.setStyleSheet("padding: 6px;")
        reset_btn.clicked.connect(self._reset_to_circle)
        controls_layout.addWidget(reset_btn)
        
        controls_layout.addStretch(1) # Push action buttons to the bottom of the container, matching canvas height
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        cancel_btn = QPushButton("Back")
        cancel_btn.setStyleSheet("padding: 8px 16px;")
        cancel_btn.clicked.connect(self.rejected.emit)
        
        save_btn = QPushButton("OK")
        save_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 8px 24px;
                background-color: #4F8A8B;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5b9e9f;
            }
            QPushButton:pressed {
                background-color: #437576;
            }
        """)
        save_btn.clicked.connect(self.accepted.emit)
        
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        controls_layout.addLayout(btn_layout)
        
        layout.addWidget(controls_widget)
        layout.addStretch(1) # Right centering spacer

    def _reset_to_circle(self):
        self._current_coords.clear()
        self._init_default_layout()
        self._update_canvas()

    def _on_point_dragged(self, identity: str, part_name: str, pos: QPointF):
        self._current_coords[part_name] = (pos.x(), pos.y())
        self._update_canvas()

    def _update_canvas(self):
        points_map = {
            bp: QPointF(x, y) for bp, (x, y) in self._current_coords.items()
        }

        bp_from = self.axis_from_combo.currentText()
        bp_to = self.axis_to_combo.currentText()
        axis_lines = []
        if bp_from in points_map and bp_to in points_map:
            axis_lines.append((points_map[bp_from], points_map[bp_to]))

        skeleton_lines = []
        for bp1, bp2 in self.skeleton:
            if bp1 in points_map and bp2 in points_map:
                skeleton_lines.append((points_map[bp1], points_map[bp2]))

        self.canvas_label.set_display_data(
            [
                {
                    "name": "template",
                    "color": QColor("cyan"),
                    "points": points_map,
                    "lines": axis_lines,
                }
            ],
            [
                {
                    "name": "skeleton",
                    "color": QColor("white"),
                    "lines": skeleton_lines,
                }
            ],
        )

    def get_result(self) -> Tuple[Dict[str, Tuple[float, float]], Tuple[str, str]]:
        axis = (self.axis_from_combo.currentText(), self.axis_to_combo.currentText())
        return self._current_coords, axis
