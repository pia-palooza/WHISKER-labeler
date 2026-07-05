# START_DIFF: whisker/gui/workflows/behavior_classification/widgets/probability_plot.py [Add verification_df support and region plotting]
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from scipy.ndimage import binary_opening

from PyQt6.QtCore import Qt, QSettings, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, 
    QCheckBox, QFrame, QLabel, QSizePolicy, QPushButton
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from whisker.gui.base.collapsible_panel import HorizontalCollapsiblePanel
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

class ProbabilityPlotWidget(QWidget):
    """
    A high-performance dashboard widget for analyzing behavior probabilities.
    Optimized to update cursor positions without redrawing the entire figure layout.
    """

    selection_changed = pyqtSignal(list)
    bout_clicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        # Data State
        self._probs_df: Optional[pd.DataFrame] = None
        self._gt_df: Optional[pd.DataFrame] = None
        self._pred_binary_df: Optional[pd.DataFrame] = None
        self._verification_df: Optional[pd.DataFrame] = None
        self._fps: float = 30.0
        self._current_frame: int = 0
        self._project_name: Optional[str] = None
        self._prob_threshold: float = 0.5
        self._min_bout_duration: float = 0.0 # in seconds
        
        # View State
        self._is_zoomed = False
        self._zoom_radius = 150
        self._checkboxes: Dict[str, QCheckBox] = {}
        
        # Matplotlib Cache (Optimization)
        self._cursor_lines: List[Line2D] = [] # Fast access to vertical lines
        self._axes = []                       # Fast access to subplots

        self._init_ui()
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _on_canvas_click(self, event):
        """Handles clicks on the plot area to seek to a specific frame."""
        if event.xdata is not None:
            frame = int(round(event.xdata))
            self.bout_clicked.emit(frame)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Top: Header with Toggle ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 2, 10, 2)
        header_layout.addStretch()

        self.toggle_behaviors_btn = QPushButton("▼ Show Behaviors")
        self.toggle_behaviors_btn.setCheckable(True)
        self.toggle_behaviors_btn.setChecked(True)
        self.toggle_behaviors_btn.setFlat(True)
        self.toggle_behaviors_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_behaviors_btn.setStyleSheet(
            "QPushButton { text-align: right; font-size: 11px; color: #888; border: none; padding: 2px; }"
            "QPushButton:hover { color: #ccc; }"
            "QPushButton:checked { color: #fff; font-weight: bold; }"
        )
        self.toggle_behaviors_btn.toggled.connect(self._on_toggle_sidebar)
        header_layout.addWidget(self.toggle_behaviors_btn)
        main_layout.addLayout(header_layout)

        # --- Bottom: Content (Plot + Sidebar) ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Matplotlib Canvas
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self.canvas, stretch=1)

        # Sidebar
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(180)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(5, 0, 5, 5)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(2, 2, 2, 2)
        self.checkbox_layout.setSpacing(2)
        self.checkbox_layout.addStretch() 
        
        self.scroll_area.setWidget(self.checkbox_container)
        self.sidebar_layout.addWidget(self.scroll_area)
        
        content_layout.addWidget(self.sidebar)
        main_layout.addLayout(content_layout)

    @pyqtSlot(bool)
    def _on_toggle_sidebar(self, checked: bool):
        """Shows or hides the behavior selection panel."""
        self.toggle_behaviors_btn.setText("▼ Show Behaviors" if checked else "◀ Show Behaviors")
        self.sidebar.setVisible(checked)
        self.canvas.draw_idle()

    def set_project_name(self, name: Optional[str]):
        """Sets the project name for persisting behavior selections."""
        self._project_name = name
        # If we already have checkboxes, reload their state
        if self._checkboxes and self._project_name:
            self._load_selection_state()

    def set_prob_threshold(self, threshold: float):
        """Sets the probability threshold for '_binary' behavior variants."""
        self._prob_threshold = threshold
        self._setup_plots()

    def set_min_bout_duration(self, duration_sec: float):
        """Sets the minimum bout duration (seconds) for '_binary' behavior variants."""
        self._min_bout_duration = duration_sec
        self._setup_plots()

    def _save_selection_state(self):
        """Saves currently selected behaviors to QSettings."""
        if not self._project_name:
            return
        
        settings = QSettings("whisker", "behavior_selections")
        selected = self.get_selected_behaviors()
        
        # Load existing dict to avoid overwriting other projects
        all_selections = settings.value("project_selections", {})
        all_selections[self._project_name] = selected
        settings.setValue("project_selections", all_selections)

    def _load_selection_state(self):
        """Loads selected behaviors from QSettings."""
        if not self._project_name:
            return
            
        settings = QSettings("whisker", "behavior_selections")
        all_selections = settings.value("project_selections", {})
        selected = all_selections.get(self._project_name)
        
        if selected is not None:
            # Block signals to avoid triggering multiple redraws
            for b, cb in self._checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(b in selected)
                cb.blockSignals(False)
            self._setup_plots()
            self.selection_changed.emit(self.get_selected_behaviors())

    def get_selected_behaviors(self) -> List[str]:
        """Returns a list of currently checked behaviors."""
        return [b for b, cb in self._checkboxes.items() if cb.isChecked()]

    def set_zoom_mode(self, enabled: bool, radius_frames: int):
        self._is_zoomed = enabled
        self._zoom_radius = radius_frames
        self.set_current_frame(self._current_frame) # Refresh view immediately

    def plot_probabilities(
        self,
        probs_df: Optional[pd.DataFrame],
        gt_df: Optional[pd.DataFrame] = None,
        pred_binary_df: Optional[pd.DataFrame] = None,
        verification_df: Optional[pd.DataFrame] = None,
        fps: Optional[float] = None,
    ):
        self._probs_df = probs_df
        self._gt_df = gt_df
        self._pred_binary_df = pred_binary_df
        self._verification_df = verification_df
        if fps: self._fps = fps

        # Check if we have ANY data to plot
        has_probs = self._probs_df is not None and not self._probs_df.empty
        has_gt = self._gt_df is not None and not self._gt_df.empty
        has_preds = self._pred_binary_df is not None and not self._pred_binary_df.empty
        has_verif = self._verification_df is not None and not self._verification_df.empty

        if not (has_probs or has_gt or has_preds or has_verif):
            self.clear()
            return

        # Determine behaviors union
        behaviors = set()
        if has_probs: behaviors.update(self._probs_df.columns)
        if has_gt: behaviors.update(self._gt_df.columns)
        if has_preds: behaviors.update(self._pred_binary_df.columns)
        if has_verif: behaviors.update(self._verification_df['behavior'].unique())
        
        current_behaviors = sorted(list(behaviors))
        
        # Refresh checkboxes only if columns changed to preserve state
        if set(current_behaviors) != set(self._checkboxes.keys()):
            self._rebuild_checkboxes(current_behaviors)
            
        # Force a redraw of the plots with the new data
        self._setup_plots()

    def update_verification_data(self, df: pd.DataFrame):
        """Updates the verification dataframe and triggers a redraw."""
        self._verification_df = df
        self._setup_plots()

    def _rebuild_checkboxes(self, behaviors: List[str]):
        # Clear old checkboxes
        for cb in self._checkboxes.values():
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()
        
        # Preserve the stretch at the bottom
        item = self.checkbox_layout.takeAt(self.checkbox_layout.count() - 1)
        
        for b in sorted(behaviors):
            cb = QCheckBox(b)
            cb.setChecked(True)
            cb.toggled.connect(self._on_checkbox_toggled) 
            self.checkbox_layout.addWidget(cb)
            self._checkboxes[b] = cb
            
        if item: self.checkbox_layout.addItem(item)
        
        # Load state after building
        if self._project_name:
            self._load_selection_state()

    @pyqtSlot(bool)
    def _on_checkbox_toggled(self, checked: bool):
        """Saves state and updates plots when a checkbox is toggled."""
        self._save_selection_state()
        self._setup_plots()
        self.selection_changed.emit(self.get_selected_behaviors())

    def _setup_plots(self):
        """
        HEAVY OPERATION: Clears and rebuilds the entire figure structure.
        Called ONLY when data changes or checkboxes are toggled.
        """
        self.figure.clear()
        self._cursor_lines.clear()
        self._axes = [] 
        
        active_behaviors = [b for b, cb in self._checkboxes.items() if cb.isChecked()]
        
        if not active_behaviors:
            self.figure.text(0.5, 0.5, "No behaviors selected", ha='center')
            self.canvas.draw_idle()
            return

        # --- Determine Master X-Axis ---
        # Prioritize Ground Truth or Binary Preds (likely aligned to video) over Raw Probs
        reference_df = None
        if self._gt_df is not None and not self._gt_df.empty:
            reference_df = self._gt_df
        elif self._pred_binary_df is not None and not self._pred_binary_df.empty:
            reference_df = self._pred_binary_df
        elif self._probs_df is not None and not self._probs_df.empty:
            reference_df = self._probs_df
        elif self._verification_df is not None:
             # Fallback for verification-only case
             max_frame = self._verification_df['end_frame'].max()
             # Create dummy ref
             reference_df = pd.DataFrame(index=pd.RangeIndex(int(max_frame) + 100))
        else:
            self.clear()
            return
            
        # Extract the master index
        if isinstance(reference_df.index, pd.RangeIndex) or pd.api.types.is_numeric_dtype(reference_df.index):
            master_index = reference_df.index
        else:
            # Handle string indices like "video_frame_001" if present
            try:
                # Attempt to extract numeric part
                numeric_idx = reference_df.index.str.split("_").str[-1].astype(int)
                master_index = pd.Index(numeric_idx, name="frame_index")
            except (AttributeError, ValueError):
                master_index = pd.RangeIndex(len(reference_df))

        x_values = master_index.values

        n_plots = len(active_behaviors)
        axs = self.figure.subplots(nrows=n_plots, ncols=1, sharex=True, squeeze=False)
        self._axes = axs.flatten() 

        # Helper to align series to master axis
        def _get_aligned_series(df, col_name, index):
            if df is None or col_name not in df.columns:
                return None
            
            # 1. Check if index alignment is needed
            if len(df) == len(index):
                # Optimization: if lengths match, assume alignment (fast path)
                return df[col_name].values
            
            # 2. Reindex (Robust path)
            # Ensure the source df has a compatible index type
            if not pd.api.types.is_numeric_dtype(df.index) and not isinstance(df.index, pd.RangeIndex):
                 # Try to fix index temporarily
                 try:
                     temp_idx = df.index.str.split("_").str[-1].astype(int)
                     df = df.set_index(temp_idx)
                 except:
                     pass # Fallback to existing index

            # Reindex and fill missing with 0
            return df[col_name].reindex(index, fill_value=0).values

        for i, behavior in enumerate(active_behaviors):
            ax = self._axes[i]
            
            # Verification Regions
            if self._verification_df is not None and not self._verification_df.empty:
                beh_bouts = self._verification_df[self._verification_df['behavior'] == behavior]
                for row in beh_bouts.itertuples():
                    status = getattr(row, 'status', 'unverified')
                    color = 'gray'
                    if status == 'correct': color = 'tab:green'
                    elif status == 'incorrect': color = 'tab:red'
                    elif status == 'merge': color = 'tab:blue'
                    # Clamp regions to current view to avoid drawing issues? No, matplotlib handles it.
                    ax.axvspan(row.start_frame, row.end_frame, color=color, alpha=0.3)

            # 1. Plot Ground Truth (Orange Fill)
            y_gt = _get_aligned_series(self._gt_df, behavior, master_index)
            if y_gt is not None:
                ax.fill_between(
                    x_values, 0, y_gt, 
                    color="tab:orange", alpha=0.3, label="GT", step='mid'
                )

            # 2. Plot Raw Probabilities or Dynamic Binary Variant
            is_binary_variant = behavior.endswith("_binary")
            base_behavior = behavior[:-7] if is_binary_variant else behavior
            
            y_prob = _get_aligned_series(self._probs_df, base_behavior, master_index)
            
            if is_binary_variant and y_prob is not None:
                # Calculate dynamic binary variant
                y_bin = (y_prob >= self._prob_threshold).astype(bool)
                
                # Apply minimum bout duration filter
                if self._min_bout_duration > 0 and self._fps > 0:
                    min_bout_frames = int(self._min_bout_duration * self._fps)
                    if min_bout_frames > 1:
                        y_bin = binary_opening(y_bin, structure=np.ones(min_bout_frames, dtype=bool))
                
                ax.plot(x_values, y_bin.astype(float), label="Binary", color="black", lw=1.5, alpha=0.8, drawstyle='steps-mid')
            elif y_prob is not None:
                # Plot raw probability
                ax.plot(x_values, y_prob, label="Prob", color="black", lw=1.5, alpha=0.6)

            # 3. Plot Discrete/Amended Predictions (Green Step)
            y_pred = _get_aligned_series(self._pred_binary_df, behavior, master_index)
            if y_pred is not None:
                ax.plot(x_values, y_pred, label="Pred", color="tab:green", lw=2.0, linestyle="-", drawstyle='steps-mid')

            ax.set_ylabel(behavior, rotation=0, ha='right', fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.grid(True, linestyle="--", alpha=0.3)
            
            if i < n_plots - 1:
                ax.tick_params(labelbottom=False)

            # Create Dynamic Cursor Line (init at 0)
            line = ax.axvline(x=0, color="red", alpha=0.8, lw=1)
            self._cursor_lines.append(line)

        # Setup Bottom Axis
        bottom_ax = self._axes[-1]
        if self._fps > 0:
            def time_fmt(x, pos):
                m, s = divmod(int(x / self._fps), 60)
                return f"{m:02d}:{s:02d}"
            bottom_ax.xaxis.set_major_formatter(FuncFormatter(time_fmt))
            bottom_ax.set_xlabel("Time (MM:SS)")
        else:
            bottom_ax.set_xlabel("Frame")

        try:
            self.figure.subplots_adjust(hspace=0.1)
            self._adjust_margins()
        except Exception:
            pass
        
        self.canvas.draw_idle()

    def _adjust_margins(self):
        """
        Dynamically adjusts the figure margins to ensure the plot area aligns 
        exactly with the 120px left / 80px right padding of the timeline and slider.
        """
        if len(self._axes) == 0:
            return

        canvas_width = self.canvas.width()
        if canvas_width <= 0:
            return

        # Target: 120px left, 80px right
        left_frac = 120.0 / canvas_width
        right_frac = 1.0 - (80.0 / canvas_width)

        if left_frac >= right_frac:
            left_frac = 0.0
            right_frac = 0.1

        self.figure.subplots_adjust(left=left_frac, right=right_frac)

    def get_current_frame(self) -> int:
        return self._current_frame

    def get_probabilities_at_frame(self, frame_index: int, behaviors: List[str]) -> Dict[str, float]:
        """Returns a mapping of behavior names to probabilities for the specified frame."""
        if self._probs_df is None or self._probs_df.empty:
            return {}
        
        try:
            # Most efficient: direct index access if numeric
            if frame_index in self._probs_df.index:
                row = self._probs_df.loc[frame_index]
            else:
                # Fallback: positional access if index is not numeric or out of sync
                if 0 <= frame_index < len(self._probs_df):
                    row = self._probs_df.iloc[frame_index]
                else:
                    return {}
            
            result = {}
            for b in behaviors:
                if b.endswith("_binary"):
                    base_b = b[:-7]
                    if base_b in self._probs_df.columns:
                        # For consistent display, we should apply filtering to the entire series
                        # and then pick the frame. (Inefficient but correct for this visualization)
                        col_probs = self._probs_df[base_b].values
                        y_bin = (col_probs >= self._prob_threshold)
                        
                        if self._min_bout_duration > 0 and self._fps > 0:
                            min_bout_frames = int(self._min_bout_duration * self._fps)
                            if min_bout_frames > 1:
                                y_bin = binary_opening(y_bin, structure=np.ones(min_bout_frames, dtype=bool))
                        
                        # Use iloc if we extracted .values earlier to avoid index mismatch
                        # Wait, we need to know the positional index of 'frame_index' in _probs_df
                        # If frame_index matches index directly:
                        try:
                            pos = self._probs_df.index.get_loc(frame_index)
                            result[b] = 1.0 if y_bin[pos] else 0.0
                        except (KeyError, IndexError):
                            result[b] = 0.0
                    elif b in row.index:
                        result[b] = float(row[b])
                elif b in row.index:
                    result[b] = float(row[b])
            return result
        except Exception:
            return {}

    def set_current_frame(self, frame_index: int):
        """
        LIGHTWEIGHT OPERATION: Updates only the cursor position.
        Called every frame during playback.
        """
        self._current_frame = frame_index
        
        if not self._cursor_lines:
            return

        # 1. Update Cursors (Fast)
        for line in self._cursor_lines:
            line.set_xdata([frame_index])

        # 2. Update Zoom (Only needs a redraw if bounds change)
        if self._axes is not None and len(self._axes) > 0:
            bottom_ax = self._axes[-1]
            
            if self._is_zoomed:
                if self._current_frame > self._zoom_radius:
                    min_x = self._current_frame - self._zoom_radius
                    max_x = self._current_frame + self._zoom_radius
                else:
                    min_x = 0
                    max_x = 2 * self._zoom_radius
                
                bottom_ax.set_xlim(min_x, max_x)
            else:
                # Reset to auto-scale (reverts zoom)
                bottom_ax.autoscale(enable=True, axis='x', tight=True)

        # 3. Redraw
        self.canvas.draw_idle()

    def clear(self):
        self._probs_df = None
        self.figure.clear()
        self._cursor_lines = []
        self._axes = []
        self.figure.text(0.5, 0.5, "Select a video...", ha='center')
        self.canvas.draw_idle()

    def export_to_image(self, file_path: str):
        """
        Exports the current plot to a PNG image, respecting zoom and visibility,
        but excluding the cursor line.
        """
        # 1. Hide Cursors
        for line in self._cursor_lines:
            line.set_visible(False)
        
        # 2. Save
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
        finally:
            # 3. Restore Cursors
            for line in self._cursor_lines:
                line.set_visible(True)
            self.canvas.draw_idle()

    def resizeEvent(self, event):
        """Ensure plot margins are adjusted correctly when the widget is resized."""
        super().resizeEvent(event)
        if len(self._axes) > 0:
            self._adjust_margins()
            self.canvas.draw_idle()


class ProbabilityPlotPanel(HorizontalCollapsiblePanel):
    """
    A collapsible panel wrapping the ProbabilityPlotWidget inside a HorizontalCollapsiblePanel.
    Exposes all public methods and attributes of the inner ProbabilityPlotWidget via delegation.
    """
    def __init__(self, title_text: str = "PROBABILITY PLOT", parent: QWidget | None = None):
        title = QLabel(title_text)
        title.setObjectName("HeaderLabel")
        super().__init__(title, parent=parent, drag_edges=Qt.Edge.TopEdge)

        self.plot_widget = ProbabilityPlotWidget()
        self.content_layout.addWidget(self.plot_widget)

        # Forward signals
        self.selection_changed = self.plot_widget.selection_changed
        self.bout_clicked = self.plot_widget.bout_clicked

        # Clear fixed height on startup so that it's resizable by splitters / layouts
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        plot_widget = self.__dict__.get("plot_widget")
        if plot_widget is not None:
            return getattr(plot_widget, name)
            
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")