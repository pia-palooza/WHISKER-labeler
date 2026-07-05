from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, 
    QSpinBox, QPushButton, QFileDialog, QLineEdit, QGroupBox, 
    QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt

class ExportClipDialog(QDialog):
    def __init__(self, parent=None, total_frames=0, fps=30.0, current_frame=0):
        super().__init__(parent)
        self.setWindowTitle("Export Video Clip")
        self.setModal(True)
        
        self.total_frames = total_frames
        self.fps = fps if fps > 0 else 30.0
        self.current_frame = current_frame
        
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Range Selection ---
        range_group = QGroupBox("Clip Range")
        range_layout = QVBoxLayout(range_group)
        
        # Mode Toggle
        mode_layout = QHBoxLayout()
        self.rb_frames = QRadioButton("Frame Number")
        self.rb_time = QRadioButton("Time (Seconds)")
        self.rb_frames.setChecked(True)
        
        self.rb_frames.toggled.connect(self._update_inputs)
        
        mode_layout.addWidget(self.rb_frames)
        mode_layout.addWidget(self.rb_time)
        range_layout.addLayout(mode_layout)
        
        # Inputs
        form = QFormLayout()
        
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, self.total_frames - 1)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, self.total_frames - 1)
        
        form.addRow("Start:", self.spin_start)
        form.addRow("End:", self.spin_end)
        range_layout.addLayout(form)
        
        # Quick Buttons
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Whole Video")
        btn_all.clicked.connect(self._set_whole_video)
        
        btn_current = QPushButton("Current Frame ± 5s")
        btn_current.clicked.connect(self._set_around_current)
        
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_current)
        range_layout.addLayout(btn_layout)
        
        layout.addWidget(range_group)
        
        # --- Output Selection ---
        out_group = QGroupBox("Output")
        out_layout = QHBoxLayout(out_group)
        
        self.line_path = QLineEdit()
        self.line_path.setPlaceholderText("Select output file...")
        btn_browse = QPushButton("...")
        btn_browse.clicked.connect(self._browse_output)
        
        out_layout.addWidget(self.line_path)
        out_layout.addWidget(btn_browse)
        
        layout.addWidget(out_group)
        
        # --- Dialog Buttons ---
        btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_export = QPushButton("Export Clip")
        btn_export.setDefault(True)
        btn_export.clicked.connect(self._validate_and_accept)
        
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_export)
        layout.addLayout(btns)
        
        # Defaults
        self._set_whole_video()

    def _update_inputs(self):
        # We maintain internal state in Frames, but display differently
        # For simplicity in this iteration, we just reset or re-calc limits.
        # Ideally, we convert values on toggle, but user intent resets usually.
        is_frame = self.rb_frames.isChecked()
        suffix = "" if is_frame else " s"
        maximum = self.total_frames - 1 if is_frame else int(self.total_frames / self.fps)
        
        self.spin_start.setSuffix(suffix)
        self.spin_end.setSuffix(suffix)
        self.spin_start.setRange(0, maximum)
        self.spin_end.setRange(0, maximum)

    def _set_whole_video(self):
        if self.rb_frames.isChecked():
            self.spin_start.setValue(0)
            self.spin_end.setValue(self.total_frames - 1)
        else:
            self.spin_start.setValue(0)
            self.spin_end.setValue(int(self.total_frames / self.fps))

    def _set_around_current(self):
        # ± 5 seconds
        center = self.current_frame
        radius = int(5 * self.fps)
        
        start = max(0, center - radius)
        end = min(self.total_frames - 1, center + radius)
        
        if self.rb_time.isChecked():
            start = int(start / self.fps)
            end = int(end / self.fps)
            
        self.spin_start.setValue(start)
        self.spin_end.setValue(end)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Video (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.line_path.setText(path)

    def _validate_and_accept(self):
        if not self.line_path.text():
            QMessageBox.warning(self, "Error", "Please select an output file path.")
            return
            
        s = self.spin_start.value()
        e = self.spin_end.value()
        
        if s >= e:
            QMessageBox.warning(self, "Error", "Start frame must be before end frame.")
            return
            
        self.accept()

    def get_range(self):
        s = self.spin_start.value()
        e = self.spin_end.value()
        
        if self.rb_time.isChecked():
            s = int(s * self.fps)
            e = int(e * self.fps)
            
        return max(0, s), min(self.total_frames - 1, e)

    def get_output_path(self):
        return Path(self.line_path.text())