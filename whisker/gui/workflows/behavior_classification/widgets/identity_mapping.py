# FILE: whisker/gui/widgets/identity_mapping_widget.py
import logging
from typing import Dict, List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QComboBox,
    QLabel,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
)

from whisker.gui.workflows.behavior_classification.widgets.evaluation_summary import HelpIcon

ROOT_BP_HELP = (
    "The reference body part (usually body-center or nose) from which all other keypoint "
    "coordinates/movements are measured. It defines the coordinate origin for feature extraction."
)

ROOT_ID_HELP = (
    "For multi-animal classification, select the specific individual (e.g. the focal animal or resident mouse) "
    "to serve as the absolute (0,0) reference point for all other animals. If set to 'None', each animal is "
    "evaluated relative to its own coordinate frame (Local)."
)


class IdentityMappingWidget(QWidget):
    """
    A reusable widget that dynamically creates a UI for mapping
    a list of "model identities" (e.g., ['Mouse A', 'Mouse B']) to
    a list of "project identities" (e.g., ['ind_1', 'ind_2']).
    """
    
    # Emitted when the mapping is valid and complete
    mapping_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._model_identities: List[str] = []
        self._project_identities: List[str] = []
        self._project_bodyparts: List[str] = []

        # This will hold our dynamically created QComboBox widgets
        self._identity_combos: List[QComboBox] = []
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # We use a QFormLayout for a nice "Label: [Dropdown]" look
        self.form_layout = QFormLayout()
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        
        self.placeholder_label = QLabel("Load a multi-animal model to configure mapping.")
        self.placeholder_label.setStyleSheet("font-style: italic;")
        
        # A simple separator line
        self.line = QFrame()
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        
        # --- Top Controls (Root Body Part & Root Individual) ---
        top_controls_layout = QHBoxLayout()
        
        # 1. Root Body Part
        self.root_bodypart_combo = QComboBox()
        self.root_bodypart_combo.setPlaceholderText("Select root body part...")
        self.root_bodypart_combo.currentTextChanged.connect(self._on_mapping_changed)
        top_controls_layout.addWidget(QLabel("Root BP:"))
        top_controls_layout.addWidget(HelpIcon(ROOT_BP_HELP, self))
        top_controls_layout.addWidget(self.root_bodypart_combo, 1)

        # 2. Root Individual (NEW)
        self.root_individual_combo = QComboBox()
        self.root_individual_combo.setPlaceholderText("None (Local)")
        self.root_individual_combo.setToolTip(
            "Select which individual serves as the (0,0) reference point for everyone else.\n"
            "Useful for interaction datasets."
        )
        # We don't trigger validation on this change necessarily, but good to have.
        top_controls_layout.addWidget(QLabel("Root ID:"))
        top_controls_layout.addWidget(HelpIcon(ROOT_ID_HELP, self))
        top_controls_layout.addWidget(self.root_individual_combo, 1)

        self.form_layout.addRow(top_controls_layout)

        main_layout.addWidget(self.line)
        main_layout.addLayout(self.form_layout)
        main_layout.addWidget(self.placeholder_label)
        
        self.setVisible(False) # Start hidden

    def set_identities_and_bodyparts(
        self, 
        model_identities: List[str], 
        project_identities: List[str],
        project_bodyparts: List[str]
    ):
        """
        Configures the widget's UI based on the required model
        identities and available project identities.
        """
        self._model_identities = model_identities
        self._project_identities = project_identities
        self._project_bodyparts = project_bodyparts
        self._identity_combos = []

        # 1. Clear any existing widgets from the form layout (excluding row 0 which is Top Controls)
        while self.form_layout.rowCount() > 1:
            self.form_layout.removeRow(1)

        # 2. Update the root body part selector
        self.root_bodypart_combo.clear()
        if self._project_bodyparts:
            self.root_bodypart_combo.addItems(self._project_bodyparts)
            self.root_bodypart_combo.setCurrentIndex(0) # Default to the first one
        else:
             self.root_bodypart_combo.setPlaceholderText("No body parts available.")

        # 3. Update the root individual selector (Populated with MODEL identities)
        self.root_individual_combo.clear()
        self.root_individual_combo.addItem("None (Self-Referencing)")
        if model_identities:
            self.root_individual_combo.addItems(model_identities)
            self.root_individual_combo.setEnabled(True)
        else:
            # Single animal case usually
            self.root_individual_combo.setEnabled(False)
             
        # 4. If no model identities are required, stay hidden (but update root selector).
        if not model_identities:
            self.setVisible(True) 
            self.placeholder_label.setVisible(True)
            return
            
        # 5. Create and add new Identity Mapping widgets
        self.placeholder_label.setVisible(False)
        
        if not project_identities:
            error_label = QLabel("Project has no identities defined.")
            error_label.setStyleSheet("color: red;")
            self.form_layout.addRow(error_label)
            self.setVisible(True)
            return

        for i, model_id in enumerate(model_identities):
            label = QLabel(f"Model Identity '{model_id}':")
            combo = QComboBox()
            combo.addItems(project_identities)
            
            if model_id in project_identities:
                combo.setCurrentText(model_id)
            elif i < len(project_identities):
                 combo.setCurrentIndex(i)

            combo.currentTextChanged.connect(self._on_mapping_changed)
            
            self.form_layout.addRow(label, combo)
            self._identity_combos.append(combo)

        self.setVisible(True)
        self._on_mapping_changed()

    def get_identity_map(self) -> Optional[Dict[str, str]]:
        """
        Returns the current mapping if it's valid, otherwise None.
        """
        if not self._model_identities or not self._identity_combos:
            return None

        mapping: Dict[str, str] = {}
        mapped_to_set = set()

        for i, combo in enumerate(self._identity_combos):
            model_id = self._model_identities[i]
            project_id = combo.currentText()
            
            if not project_id:
                return None
            
            if project_id in mapped_to_set:
                return None
            
            mapped_to_set.add(project_id)
            mapping[model_id] = project_id

        return mapping

    def get_root_bodypart(self) -> str:
        return self.root_bodypart_combo.currentText()
        
    def get_root_individual(self) -> Optional[str]:
        """Returns the selected Model Identity to be used as root, or None."""
        txt = self.root_individual_combo.currentText()
        if not txt or txt == "None (Self-Referencing)":
            return None
        return txt

    def _on_mapping_changed(self):
        current_map = self.get_identity_map()
        
        if current_map:
            for combo in self._identity_combos:
                combo.setStyleSheet("")
            self.mapping_changed.emit(current_map)
        else:
            self._highlight_duplicates()

    def _highlight_duplicates(self):
        counts: Dict[str, int] = {}
        for combo in self._identity_combos:
            text = combo.currentText()
            counts[text] = counts.get(text, 0) + 1
            
        for combo in self._identity_combos:
            if counts.get(combo.currentText(), 0) > 1:
                combo.setStyleSheet("border: 1px solid red;")
            else:
                combo.setStyleSheet("")