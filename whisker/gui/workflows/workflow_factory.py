import logging
import os
from typing import Callable
from pathlib import Path

import cv2
import h5py
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QButtonGroup,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStackedWidget,
    QGroupBox,
    QFormLayout,
    QTextEdit,
)
from whisker.gui.tabs.base_workflow_tab import WorkflowItemHandler
from .pose_estimation.widgets.info_item_handler import PoseEstimationInfoItemHandlerWidget
from .behavior_classification.widgets.info_item_handler import BehaviorClassificationInfoItemHandlerWidget

def get_workflow_info_item_handlers() -> dict[str, WorkflowItemHandler]:
    return {
        workflow_name: (
            widget.show_workflow_item_info,
            widget,
        ) for workflow_name, widget in [
            ("pose_estimation", PoseEstimationInfoItemHandlerWidget()),
            ("behavior_classification", BehaviorClassificationInfoItemHandlerWidget()),
        ]
    }
