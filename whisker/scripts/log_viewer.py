import logging
import sys
import ast
import json
import re
import os
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTableWidget,
                             QTableWidgetItem, QFileDialog, QHeaderView, QWidget,
                             QHBoxLayout, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QLabel, QSplitter)
from PyQt6.QtGui import QAction, QColor, QBrush, QActionGroup
from PyQt6.QtCore import Qt, QSettings, QTimer

from whisker.base import topics

HEADERS = ["Source File", "Level", "Logger", "Timestamp", "Message ID", "Sender \u2192 Target", "Message", "Routing Path"]

class ColumnIndices:
    SOURCE_FILE = 0
    LEVEL = 1
    LOGGER = 2
    TIMESTAMP = 3
    MESSAGE_ID = 4
    SENDER_TARGET = 5
    MESSAGE = 6
    ROUTING_PATH = 7

TARGET_DATA_ROLE = Qt.ItemDataRole.UserRole + 1
TOPIC_DATA_ROLE = Qt.ItemDataRole.UserRole + 2
PAYLOAD_DATA_ROLE = Qt.ItemDataRole.UserRole + 3
CORRELATION_DATA_ROLE = Qt.ItemDataRole.UserRole + 4

PALETTE = [
    QColor("#fce4ec"), QColor("#e8f5e9"), QColor("#fff9c4"), QColor("#e3f2fd"),
    QColor("#fff3e0"), QColor("#f3e5f5"), QColor("#e0f7fa"), QColor("#fce4ec"),
    QColor("#f1f8e9"), QColor("#fbe9e7"), QColor("#e0f2f1"), QColor("#ede7f6"),
    QColor("#e8eaf6"), QColor("#f9fbe7"), QColor("#efebe9"), QColor("#e1f5fe")
]

LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
NODE_UUID_PATTERN = r"(?P<label>[a-z_.]+)@(?P<uuid>[0-9a-f]{8})"
ENUM_REPR_PATTERN = re.compile(r"<[A-Za-z_][A-Za-z0-9_.]*:\s*'([^']*)'>"  )

def sanitize_enum_reprs(text: str) -> str:
    """Replace Python Enum repr strings like <Request.SHUTDOWN: 'value'> with 'value'."""
    return ENUM_REPR_PATTERN.sub(r"'\1'", text)

def get_contrast_color(bg: QColor) -> QColor:
    lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
    return QColor(Qt.GlobalColor.black) if lum > 128 else QColor(Qt.GlobalColor.white)

class LogViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisker Log Viewer")
        self.resize(1400, 700) # Width increased to host the filter panel comfortably
        self.settings = QSettings("WhiskerTeam", "LogViewer")
        
        self.color_map = {}
        self.next_color_idx = 0
        
        self.short_sender_names = {}
        self.prefix_counts = {}
        
        # Main Layout Setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        splitter.addWidget(self.tabs)

        # Right Side Filter Panel Setup
        filter_panel = QWidget()
        filter_layout = QVBoxLayout(filter_panel)
        
        filter_layout.addWidget(QLabel("<b>Filter Senders</b>"))
        self.sender_tree = QTreeWidget()
        self.sender_tree.setHeaderHidden(True)
        self.sender_tree.itemChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.sender_tree)
        
        filter_layout.addWidget(QLabel("<b>Filter Topics</b>"))
        self.topic_tree = QTreeWidget()
        self.topic_tree.setHeaderHidden(True)
        self.topic_tree.itemChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.topic_tree)

        filter_layout.addWidget(QLabel("<b>Filter Routing Paths</b>"))
        self.routing_path_tree = QTreeWidget()
        self.routing_path_tree.setHeaderHidden(True)
        self.routing_path_tree.itemChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.routing_path_tree)
        
        splitter.addWidget(filter_panel)
        splitter.setSizes([1000, 400])

        self.opened_files = []
        self.file_positions = {}

        self.setup_menus()
        
        stored_files = self.settings.value("open_files", [])
        if isinstance(stored_files, str):
            stored_files = [stored_files]
        self.opened_files = [f for f in stored_files if os.path.exists(f)]
        
        self.rebuild_views()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_files)
        self.timer.start(100)

    def setup_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open Log File", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        view_menu = self.menuBar().addMenu("&View")
        
        self.merge_action = QAction("Merge Files", self, checkable=True)
        self.merge_action.setChecked(self.settings.value("merge_mode", False, type=bool))
        self.merge_action.triggered.connect(self.toggle_merge)
        view_menu.addAction(self.merge_action)
        view_menu.addSeparator()
        
        log_level_menu = view_menu.addMenu("Log Level")
        self.level_group = QActionGroup(self)
        self.level_group.setExclusive(True)
        saved_level = self.settings.value("log_level", "DEBUG")
        self.current_log_level = LOG_LEVELS.get(saved_level, 10)

        for level_name in LOG_LEVELS.keys():
            act = QAction(level_name, self, checkable=True)
            if level_name == saved_level:
                act.setChecked(True)
            act.triggered.connect(lambda checked, lvl=level_name: self.change_log_level(lvl))
            self.level_group.addAction(act)
            log_level_menu.addAction(act)
            
        view_menu.addSeparator()
        
        cols_menu = view_menu.addMenu("Display Columns")
        self.col_actions = []
        for i, header in enumerate(HEADERS):
            act = QAction(header, self, checkable=True)
            act.setChecked(self.settings.value(f"col_{i}", True, type=bool))
            act.triggered.connect(lambda checked, idx=i: self.toggle_column(idx, checked))
            cols_menu.addAction(act)
            self.col_actions.append(act)

        # Color Key Submenu
        color_key_menu = view_menu.addMenu("Color Key")
        self.color_key_group = QActionGroup(self)
        self.color_key_group.setExclusive(True)
        self.current_color_key = self.settings.value("color_key_mode", "Sender ID")
        
        color_modes = ["Sender ID", "Source File column", "Target Node ID", "Topic"]
        # NOTE: "Sender ID" and "Target Node ID" color keys operate on the
        # sender/target stored in the merged Sender → Target column.
        for mode in color_modes:
            act = QAction(mode, self, checkable=True)
            if mode == self.current_color_key:
                act.setChecked(True)
            act.triggered.connect(lambda checked, m=mode: self.change_color_key(m))
            self.color_key_group.addAction(act)
            color_key_menu.addAction(act)

        self.compact_action = QAction("Compact UUID", self, checkable=True)
        self.compact_action.setChecked(self.settings.value("compact_uuid", False, type=bool))
        self.compact_action.triggered.connect(self.toggle_compact)
        view_menu.addAction(self.compact_action)

    def change_color_key(self, mode_str):
        self.current_color_key = mode_str
        self.settings.setValue("color_key_mode", mode_str)
        self.color_map.clear()
        self.next_color_idx = 0
        self.rebuild_views()

    def change_log_level(self, level_str):
        self.settings.setValue("log_level", level_str)
        self.current_log_level = LOG_LEVELS.get(level_str, 10)
        self.rebuild_views()

    def open_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Log Files", "", "All Files (*)")
        if file_paths:
            for path in file_paths:
                if path not in self.opened_files:
                    self.opened_files.append(path)
            self.rebuild_views()
            self.save_open_files()

    def create_table(self, file_path=None):
        table = QTableWidget()
        table.setColumnCount(len(HEADERS))
        table.setHorizontalHeaderLabels(HEADERS)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setWordWrap(True)
        table.file_path = file_path 

        for i, act in enumerate(self.col_actions):
            table.setColumnHidden(i, not act.isChecked())
            
        return table

    def rebuild_views(self):
        self.tabs.clear()
        self.sender_tree.clear()
        self.topic_tree.clear()
        self.file_positions = {p: 0 for p in self.opened_files}
        
        if not self.opened_files:
            return

        if self.merge_action.isChecked():
            table = self.create_table()
            self.tabs.addTab(table, "Merged Logs")
            for path in self.opened_files:
                self.read_new_lines(table, path)
            table.sortItems(ColumnIndices.TIMESTAMP, Qt.SortOrder.AscendingOrder)
        else:
            for path in self.opened_files:
                table = self.create_table(path)
                self.tabs.addTab(table, os.path.basename(path))
                self.read_new_lines(table, path)

    def toggle_merge(self):
        self.settings.setValue("merge_mode", self.merge_action.isChecked())
        self.rebuild_views()

    def refresh_files(self):
        if not self.opened_files:
            return

        if self.merge_action.isChecked():
            if self.tabs.count() == 0: return
            table = self.tabs.widget(0)
            any_added = False
            for path in self.opened_files:
                if self.read_new_lines(table, path):
                    any_added = True
            if any_added:
                table.sortItems(ColumnIndices.TIMESTAMP, Qt.SortOrder.AscendingOrder)
        else:
            for i in range(self.tabs.count()):
                table = self.tabs.widget(i)
                self.read_new_lines(table, table.file_path)

    def get_short_sender(self, sender_id):
        if sender_id in self.short_sender_names:
            return self.short_sender_names[sender_id]
        
        if '@' in sender_id:
            prefix = sender_id.split('@', 1)[0]
            self.prefix_counts[prefix] = self.prefix_counts.get(prefix, 0) + 1
            short_name = f"{prefix}-{self.prefix_counts[prefix]}"
        else:
            short_name = sender_id
            
        self.short_sender_names[sender_id] = short_name
        return short_name

    def get_row_colors(self, source_name, sender_id, target_node_id, topic):
        if self.current_color_key == "Target Node ID" and (not target_node_id or target_node_id.lower() == "none"):
            bg = QColor("#d3d3d3")
            return QBrush(bg), QBrush(get_contrast_color(bg))

        key = sender_id
        if self.current_color_key == "Source File column":
            key = source_name
        elif self.current_color_key == "Target Node ID":
            key = target_node_id or sender_id
        elif self.current_color_key == "Topic":
            key = topic.split('/')[0] if topic else ""

        if key not in self.color_map:
            bg = PALETTE[self.next_color_idx % len(PALETTE)]
            fg = get_contrast_color(bg)
            self.color_map[key] = (QBrush(bg), QBrush(fg))
            self.next_color_idx += 1
            
        return self.color_map[key]

    def add_hierarchical_filter(self, tree_widget, value_str, source_name=None):
        """Builds a tree configuration matching standard tokens separated by / or ."""
        if not value_str: return
        
        if tree_widget in (self.sender_tree, self.topic_tree):
            delimiters = r'[./]'
            tokens = [t for t in re.split(delimiters, value_str) if t]
            
            tree_widget.blockSignals(True)
            
            parent_item = None
            for i in range(len(tokens)):
                current_path = "/".join(tokens[:i+1])
                found = False
                
                root = tree_widget if parent_item is None else parent_item
                for idx in range(root.childCount() if parent_item else root.topLevelItemCount()):
                    item = root.child(idx) if parent_item else root.topLevelItem(idx)
                    if item.text(0) == tokens[i]:
                        parent_item = item
                        found = True
                        break
                
                if not found:
                    new_item = QTreeWidgetItem([tokens[i]])
                    new_item.setFlags(new_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    new_item.setCheckState(0, Qt.CheckState.Checked)
                    new_item.setData(0, Qt.ItemDataRole.UserRole, current_path)
                    if parent_item:
                        parent_item.addChild(new_item)
                    else:
                        tree_widget.addTopLevelItem(new_item)
                    parent_item = new_item
                    
            tree_widget.blockSignals(False)
        elif tree_widget == self.routing_path_tree:
            if not value_str or value_str == "[]" or not source_name:
                return
            
            try:
                path_list = ast.literal_eval(value_str)
                if not path_list or not isinstance(path_list, list):
                    return
            except (ValueError, SyntaxError):
                return

            tree_widget.blockSignals(True)
            
            # 1. Find or create the Top-Level Source File item
            source_item = None
            for idx in range(tree_widget.topLevelItemCount()):
                item = tree_widget.topLevelItem(idx)
                if item.text(0) == source_name:
                    source_item = item
                    break
                    
            if not source_item:
                source_item = QTreeWidgetItem([source_name])
                source_item.setFlags(source_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                source_item.setCheckState(0, Qt.CheckState.Checked)
                source_item.setData(0, Qt.ItemDataRole.UserRole, source_name)
                tree_widget.addTopLevelItem(source_item)
            
            # 2. Add individual hop nodes under this specific source file item
            for node in path_list:
                node_str = str(node)
                
                exists = False
                for idx in range(source_item.childCount()):
                    if source_item.child(idx).data(0, Qt.ItemDataRole.UserRole) == node_str:
                        exists = True
                        break
                        
                if not exists:
                    display_text = self.get_short_sender(node_str) if self.compact_action.isChecked() else node_str
                    child_item = QTreeWidgetItem([display_text])
                    child_item.setFlags(child_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child_item.setCheckState(0, Qt.CheckState.Checked)
                    child_item.setData(0, Qt.ItemDataRole.UserRole, node_str)
                    source_item.addChild(child_item)
                    
            tree_widget.blockSignals(False)
            

    def is_allowed_by_tree(self, tree_widget, value_str, source_name=None):
        """Evaluates row matching status based on parent tree checkbox state cascades."""
        if not value_str or tree_widget.topLevelItemCount() == 0:
            return True

        if tree_widget in (self.sender_tree, self.topic_tree):
            delimiters = r'[./]'
            tokens = [t for t in re.split(delimiters, value_str) if t]
            
            parent_item = None
            for i in range(len(tokens)):
                found = False
                root = tree_widget if parent_item is None else parent_item
                for idx in range(root.childCount() if parent_item else root.topLevelItemCount()):
                    item = root.child(idx) if parent_item else root.topLevelItem(idx)
                    if item.text(0) == tokens[i]:
                        if item.checkState(0) == Qt.CheckState.Unchecked:
                            return False
                        parent_item = item
                        found = True
                        break
                if not found:
                    return True
            return True
        elif tree_widget == self.routing_path_tree:
            if not value_str or value_str == "[]" or not source_name:
                return True
            
            # Locate the configuration tree node belonging to this log's file source
            for idx in range(tree_widget.topLevelItemCount()):
                source_item = tree_widget.topLevelItem(idx)
                if source_item.text(0) == source_name:
                    # If the entire source file checkbox is unchecked, hide the row
                    if source_item.checkState(0) == Qt.CheckState.Unchecked:
                        return False
                        
                    # Evaluate individual sub-node checkbox cascades inside this source file scope
                    for child_idx in range(source_item.childCount()):
                        child_item = source_item.child(child_idx)
                        if child_item.checkState(0) == Qt.CheckState.Unchecked:
                            node_in_tree = child_item.data(0, Qt.ItemDataRole.UserRole)
                            if f"'{node_in_tree}'" in value_str or f'"{node_in_tree}"' in value_str:
                                return False
                    break
            return True

    def on_filter_changed(self, item, column):
        """Propagates checkbox state changes downstream/upstream and updates row states."""
        self.sender_tree.blockSignals(True)
        self.topic_tree.blockSignals(True)
        self.routing_path_tree.blockSignals(True)
        
        # Propagate states downwards to children
        state = item.checkState(0)
        stack = [item.child(i) for i in range(item.childCount())]
        while stack:
            curr = stack.pop()
            curr.setCheckState(0, state)
            stack.extend([curr.child(i) for i in range(curr.childCount())])
            
        self.sender_tree.blockSignals(False)
        self.topic_tree.blockSignals(False)
        self.routing_path_tree.blockSignals(False)
        
        # Update visibility states across tables
        for i in range(self.tabs.count()):
            table = self.tabs.widget(i)
            for row in range(table.rowCount()):
                self.evaluate_row_visibility(table, row)

    def evaluate_row_visibility(self, table, row):
        sender_target_item = table.item(row, ColumnIndices.SENDER_TARGET)
        message_item = table.item(row, ColumnIndices.MESSAGE)
        routing_path_item = table.item(row, ColumnIndices.ROUTING_PATH)
        source_item = table.item(row, ColumnIndices.SOURCE_FILE)
        if not sender_target_item or not message_item: return
        
        sender_val = sender_target_item.data(Qt.ItemDataRole.UserRole)
        topic_val = message_item.data(TOPIC_DATA_ROLE) or ''
        routing_path_val = routing_path_item.data(Qt.ItemDataRole.UserRole) if routing_path_item else None
        source_val = source_item.data(Qt.ItemDataRole.UserRole) if source_item else None

        show_row = (self.is_allowed_by_tree(self.sender_tree, sender_val) and 
                    self.is_allowed_by_tree(self.topic_tree, topic_val) and
                    self.is_allowed_by_tree(self.routing_path_tree, routing_path_val, source_val))
        table.setRowHidden(row, not show_row)

    def read_new_lines(self, table, file_path):
        added_any = False
        try:
            current_size = os.path.getsize(file_path)
            file_pos = self.file_positions.get(file_path, 0)
            
            if current_size < file_pos:
                if self.merge_action.isChecked():
                    QTimer.singleShot(0, self.rebuild_views)
                    return False
                else:
                    self.file_positions[file_path] = 0
                    file_pos = 0
                    table.setRowCount(0)
                    table.clearContents()

            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(file_pos)
                lines = f.readlines()
                self.file_positions[file_path] = f.tell() 
                
                if not lines:
                    return False

                current_row = table.rowCount()
                table.setRowCount(current_row + len(lines))
                
                valid_lines_added = 0
                source_name = os.path.basename(file_path)

                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        # Parse the outer JSON wrapper emitted by JsonFormatter
                        outer_record = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        logging.debug(f"Skipping non-JSON line in {file_path}: {line[:120]}")
                        continue

                    log_level = outer_record.get('level', 'DEBUG')
                    log_level_num = LOG_LEVELS.get(log_level, 10)
                    if log_level_num < self.current_log_level:
                        continue

                    logger_name = outer_record.get('name', '')
                    log_timestamp = outer_record.get('timestamp', '')
                    message_text = outer_record.get('message', '')
                    func_name = outer_record.get('funcName', '')

                    try:
                        # Bus messages are logged as Python dict reprs in the 'message' field
                        if func_name == 'log_bus_message':
                            record = ast.literal_eval(sanitize_enum_reprs(message_text))
                            h = record.get('header', {})
                            p = record.get('payload', {})
                            
                            if h.get('topic') == topics.node.Telemetry.HEARTBEAT:
                                p["timestamp"] = datetime.fromtimestamp(float(p["timestamp"])).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                            ts = h.get('timestamp')
                            ts_str = datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if ts else log_timestamp
                            sender_id = str(h.get('sender_id', ''))
                            target_node_id = str(h.get('target_node_id', ''))
                            topic = str(h.get('topic', ''))
                            message_id = str(h.get('message_id', ''))
                            correlation_id = str(h.get('correlation_id', ''))
                            routing_path = str(h.get('routing_path', []))
                            payload_str = json.dumps(p, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o))
                        else:
                            # Non-bus-message log lines (INFO, WARNING, etc.)
                            ts_str = log_timestamp
                            sender_id = logger_name
                            target_node_id = ''
                            topic = ''
                            message_id = ''
                            correlation_id = ''
                            routing_path = '[]'
                            payload_str = message_text

                        # Dynamically expand UI hierarchical structure rules on discovery
                        self.add_hierarchical_filter(self.sender_tree, sender_id)
                        if topic:
                            self.add_hierarchical_filter(self.topic_tree, topic)
                        self.add_hierarchical_filter(self.routing_path_tree, routing_path, source_name)
                        row_data = [
                            source_name,
                            log_level,
                            logger_name,
                            ts_str,
                            message_id,
                            sender_id,  # UserRole stores sender_id for filtering
                            topic,      # UserRole stores topic for filtering; payload/correlation in extra roles
                            routing_path
                        ]
                        
                        bg_brush, fg_brush = self.get_row_colors(source_name, sender_id, target_node_id, topic)
                        
                        target_row = current_row + valid_lines_added
                        for col_idx, col_value in enumerate(row_data):
                            item = QTableWidgetItem()
                            item.setData(Qt.ItemDataRole.UserRole, col_value) 
                            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                            
                            item.setBackground(bg_brush)
                            item.setForeground(fg_brush)
                            
                            # Store target_node_id alongside sender in the merged column
                            if col_idx == ColumnIndices.SENDER_TARGET:
                                item.setData(TARGET_DATA_ROLE, target_node_id)
                            
                            # Store topic, payload, correlation in the merged Message column
                            if col_idx == ColumnIndices.MESSAGE:
                                item.setData(TOPIC_DATA_ROLE, topic)
                                item.setData(PAYLOAD_DATA_ROLE, payload_str)
                                item.setData(CORRELATION_DATA_ROLE, correlation_id)
                            
                            table.setItem(target_row, col_idx, item)
                            
                        self.format_row_uuids(table, target_row)
                        self.evaluate_row_visibility(table, target_row)
                        valid_lines_added += 1
                        added_any = True

                    except (ValueError, SyntaxError):
                        logging.exception(f"Failed to parse bus message in {file_path}: {message_text[:120]}")
                        continue
                
                table.setRowCount(current_row + valid_lines_added)

        except Exception:
            logging.exception(f"Error reading file {file_path}")
            
        return added_any

    def format_row_uuids(self, table, row):
        is_compact = self.compact_action.isChecked()
        
        sender_target_item = table.item(row, ColumnIndices.SENDER_TARGET)
        if not sender_target_item: return
        sender_val = sender_target_item.data(Qt.ItemDataRole.UserRole)
        target_val = sender_target_item.data(TARGET_DATA_ROLE) or ''
        
        sender_display = self.get_short_sender(sender_val) if is_compact else sender_val
        if target_val and target_val.lower() != 'none':
            target_display = self.get_short_sender(target_val) if is_compact else target_val
            sender_target_item.setText(f"{sender_display} => {target_display}")
        else:
            sender_target_item.setText(sender_display)
        
        msg_item = table.item(row, ColumnIndices.MESSAGE_ID)
        msg_val = msg_item.data(Qt.ItemDataRole.UserRole)
        msg_item.setText(msg_val[:8] if is_compact and len(msg_val) >= 8 else msg_val)
        
        # Compose the merged Message column: "<topic> [correlation_id]\npayload"
        message_item = table.item(row, ColumnIndices.MESSAGE)
        if message_item:
            topic_val = message_item.data(TOPIC_DATA_ROLE) or ''
            payload_val = message_item.data(PAYLOAD_DATA_ROLE) or ''
            correlation_val = message_item.data(CORRELATION_DATA_ROLE) or ''
            
            if is_compact:
                correlation_display = correlation_val[:8] if len(correlation_val) >= 8 else correlation_val
                matches = re.finditer(NODE_UUID_PATTERN, payload_val, re.I)
                for m in matches:
                    full_uuid = m.group("label") + "@" + m.group("uuid")
                    short_name = self.get_short_sender(full_uuid)
                    payload_val = payload_val.replace(full_uuid, short_name)
            else:
                correlation_display = correlation_val
            
            header_parts = []
            if topic_val:
                header_parts.append(f"<{topic_val}>")
            if correlation_display and correlation_display.lower() != 'none':
                header_parts.append(f"[{correlation_display}]")
            
            header_line = " ".join(header_parts)
            if header_line and payload_val:
                message_item.setText(f"{header_line}\n{payload_val}")
            elif header_line:
                message_item.setText(header_line)
            else:
                message_item.setText(payload_val)
    
        for col in [ColumnIndices.SOURCE_FILE, ColumnIndices.LEVEL, ColumnIndices.LOGGER, ColumnIndices.TIMESTAMP, ColumnIndices.ROUTING_PATH]:
            item = table.item(row, col)
            if item: item.setText(item.data(Qt.ItemDataRole.UserRole))

    def toggle_compact(self):
        self.settings.setValue("compact_uuid", self.compact_action.isChecked())
        for i in range(self.tabs.count()):
            table = self.tabs.widget(i)
            for row in range(table.rowCount()):
                self.format_row_uuids(table, row)
    
    def toggle_column(self, col_idx, is_checked):
        self.settings.setValue(f"col_{col_idx}", is_checked)
        for i in range(self.tabs.count()):
            self.tabs.widget(i).setColumnHidden(col_idx, not is_checked)

    def close_tab(self, index):
        if self.merge_action.isChecked():
            self.opened_files.clear()
        else:
            path = self.tabs.widget(index).file_path
            if path in self.opened_files:
                self.opened_files.remove(path)
                
        self.rebuild_views()
        self.save_open_files()

    def save_open_files(self):
        self.settings.setValue("open_files", self.opened_files)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = LogViewer()
    viewer.show()
    sys.exit(app.exec())