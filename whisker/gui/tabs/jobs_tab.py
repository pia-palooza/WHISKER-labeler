from datetime import datetime
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QProgressBar, QPushButton, QHBoxLayout, 
    QLabel, QAbstractItemView
)

from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.job_manager import JobManager, JobStatus

# Column Indices
COL_NAME = 0
COL_STATUS = 1
COL_PROGRESS = 2
COL_TIME = 3
COL_END_TIME = 4
COL_ELAPSED = 5
COL_ACTION = 6

class JobsTab(BaseTab):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        
        # Setup Timer for Elapsed Time updates
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_timer_timeout)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header Controls
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Background Tasks</b>"))
        header.addStretch()
        
        self.clear_btn = QPushButton("Clear Finished")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        header.addWidget(self.clear_btn)
        
        layout.addLayout(header)

        # Job Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Task Name", "Status", "Progress", "Start Time", "End Time", "Elapsed", "Action"
        ])
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.table)
        
        # Map job_id -> row_index for fast updates
        self._row_map = {} 

    def _connect_signals(self):
        manager = JobManager.get()
        manager.job_added.connect(self._on_job_added)
        manager.job_updated.connect(self._on_job_updated)
        manager.job_removed.connect(self._on_job_removed)

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()

    def _on_timer_timeout(self):
        """Updates the elapsed time for all running jobs."""
        manager = JobManager.get()
        for job_id, row in self._row_map.items():
            record = manager.get_job(job_id)
            if record and record.status == JobStatus.RUNNING:
                item = self.table.item(row, COL_ELAPSED)
                if item:
                    item.setText(record.runtime_display)

    def _on_clear_clicked(self):
        JobManager.get().clear_finished_jobs()

    @pyqtSlot(str)
    def _on_job_added(self, job_id: str):
        record = JobManager.get().get_job(job_id)
        if not record: return

        row = self.table.rowCount()
        self.table.insertRow(row)
        self._row_map[job_id] = row

        # 1. Name
        self.table.setItem(row, COL_NAME, QTableWidgetItem(record.name))
        
        # 2. Status
        self.table.setItem(row, COL_STATUS, QTableWidgetItem(record.status.value))
        
        # 3. Progress (Widget)
        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setValue(0)
        pbar.setTextVisible(True)
        # Fix styling to fit cell
        pbar.setStyleSheet("QProgressBar { border: none; background: transparent; }")
        self.table.setCellWidget(row, COL_PROGRESS, pbar)
        
        # 4. Time
        time_str = datetime.fromtimestamp(record.start_time).strftime("%H:%M:%S")
        self.table.setItem(row, COL_TIME, QTableWidgetItem(time_str))

        # 5. End Time
        end_time_str = (
            "" if record.end_time is None 
            else datetime.fromtimestamp(record.end_time).strftime("%H:%M:%S")
        )
        self.table.setItem(row, COL_END_TIME, QTableWidgetItem(end_time_str))
        
        # 6. Elapsed
        self.table.setItem(row, COL_ELAPSED, QTableWidgetItem(record.runtime_display))

        # 7. Action (Cancel Button)
        cancel_btn = QPushButton("Cancel")
        # Use lambda capture carefully
        cancel_btn.clicked.connect(lambda _, jid=job_id: JobManager.get().cancel_job(jid))
        self.table.setCellWidget(row, COL_ACTION, cancel_btn)

    @pyqtSlot(str)
    def _on_job_updated(self, job_id: str):
        row = self._row_map.get(job_id)
        if row is None: return
        
        record = JobManager.get().get_job(job_id)
        if not record: return

        # Update Status
        status_item = self.table.item(row, COL_STATUS)
        if status_item:
            status_item.setText(record.status.value)
            
            # Color coding status
            if record.status == JobStatus.ERROR:
                status_item.setForeground(Qt.GlobalColor.red)
            elif record.status == JobStatus.FINISHED:
                status_item.setForeground(Qt.GlobalColor.green)
        
        # Update End Time if finished
        if record.end_time:
            end_time_str = datetime.fromtimestamp(record.end_time).strftime("%H:%M:%S")
            end_item = self.table.item(row, COL_END_TIME)
            if end_item:
                end_item.setText(end_time_str)
            else:
                self.table.setItem(row, COL_END_TIME, QTableWidgetItem(end_time_str))

        # Update Elapsed
        elapsed_item = self.table.item(row, COL_ELAPSED)
        if elapsed_item:
            elapsed_item.setText(record.runtime_display)

        # Update Progress
        pbar = self.table.cellWidget(row, COL_PROGRESS)
        if isinstance(pbar, QProgressBar):
            pbar.setValue(record.progress)
            # Show message in progress text if available
            pbar.setFormat(f"%p% - {record.message}" if record.message else "%p%")

        # Update Action Button State
        btn = self.table.cellWidget(row, COL_ACTION)
        if isinstance(btn, QPushButton):
            if record.status in (JobStatus.FINISHED, JobStatus.ERROR, JobStatus.CANCELLED):
                btn.setEnabled(False)
                btn.setText("Done")

    @pyqtSlot(str)
    def _on_job_removed(self, job_id: str):
        """Surgically removes the job row and updates the row map."""
        row = self._row_map.get(job_id)
        if row is None: return

        self.table.removeRow(row)
        del self._row_map[job_id]

        # Shift all subsequent row indices in the map
        for jid, r in self._row_map.items():
            if r > row:
                self._row_map[jid] = r - 1