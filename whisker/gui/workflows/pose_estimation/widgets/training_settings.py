# UPDATE_FILE: whisker/gui/workflows/pose_estimation/widgets/training_settings.py
from typing import Any, Dict
import logging

from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QSpinBox, QWidget,
    QGroupBox, QVBoxLayout, QHBoxLayout, QCheckBox, QTabWidget,
    QButtonGroup, QRadioButton, QLabel
)
from whisker.gui.base.collapsible_panel import HelpIcon


class FormLabelWithHelp(QWidget):
    """A widget combining a QLabel and a HelpIcon side-by-side."""

    def __init__(self, label_text: str, help_text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QLabel(label_text)
        layout.addWidget(self.label)

        self.help_icon = HelpIcon(help_text, self)
        layout.addWidget(self.help_icon)

        layout.addStretch()


MODEL_ARCH_HELP = (
    "Select the prediction paradigm:\n\n"
    "• Global: Detects all keypoints directly across the entire frame. Good for single animals.\n"
    "• Top-Down: First runs an animal detector, then estimates pose within cropped animal boxes. Required for multiple animals."
)

MODEL_TYPE_HELP = (
    "Select the neural network model design:\n\n"
    "• vitpose: Vision Transformer for Pose Estimation. SOTA accuracy.\n"
    "• mobilenet: Lightweight, fast model optimized for resource-constrained environments.\n"
    "• hrnet: High-Resolution Network. Maintains high-res representations throughout the model."
)

BACKBONE_HELP = (
    "The pretrained feature extractor structure (e.g. vitpose-base vs. vitpose-large). "
    "Larger backbones are more accurate but slower to train/run."
)

IMG_WIDTH_HELP = (
    "The target width in pixels that input images are resized to before being passed to the network."
)

IMG_HEIGHT_HELP = (
    "The target height in pixels that input images are resized to before being passed to the network."
)

BASE_LOSS_HELP = (
    "The optimization objective function:\n\n"
    "• Standard MSE: Mean Squared Error. Standard distance penalty.\n"
    "• Adaptive Wing Loss: Focuses gradients on small errors, improving accuracy on fine-grained keypoints."
)

DECODER_ATTN_HELP = (
    "Applies spatial & channel attention (scSE) to refine features in the upsampling decoder, improving keypoint localization."
)

UDP_HELP = (
    "If checked, uses unbiased sub-pixel coordinate mapping during post-processing to eliminate coordinate shift errors."
)

VARIANCE_PENALTY_HELP = (
    "Penalty to encourage localized heatmap predictions. High values prevent spread-out heatmaps."
)

GEOM_LOSS_HELP = (
    "Penalty enforcing relative spatial skeleton distance constraints between keypoints to prevent anatomically impossible poses."
)

GEOM_WARMUP_HELP = (
    "The number of epochs over which the geometric skeleton loss weight is gradually increased, allowing the model to learn raw heatpoints first."
)

LIMB_THICKNESS_HELP = (
    "The thickness in pixels of the limb connection lines for Part Affinity Fields (multi-instance tracking)."
)

LIMB_LOSS_WEIGHT_HELP = (
    "Scale factor for Part Affinity Field loss optimization."
)

EPOCHS_HELP = (
    "Maximum training duration (passes over dataset)."
)

EPOCHS_PER_CHECKPOINT_HELP = (
    "Frequency of saving model weights and running full validation."
)

BATCH_SIZE_HELP = (
    "Number of samples processed concurrently in one training step."
)

LR_HELP = (
    "Starting optimization step size."
)

EARLY_STOPPING_HELP = (
    "Terminate training if validation loss does not improve for this many consecutive checkpoints."
)

SCHED_PATIENCE_HELP = (
    "Reduce learning rate if validation loss plateaus for this many epochs."
)

SCHED_FACTOR_HELP = (
    "Multiplier used to reduce the learning rate (e.g., 0.5 cuts it in half)."
)

MIN_LR_HELP = (
    "The lower limit to which the learning rate can be reduced by the scheduler."
)

ROTATION_LIMIT_HELP = (
    "Maximum angle (+/- degrees) to rotate images randomly during training."
)

SCALE_LIMIT_HELP = (
    "Maximum scale factor (+/- %) to resize images randomly during training."
)

GEOM_PROB_HELP = (
    "The probability (0 to 1) of applying geometric data augmentations to any training image."
)

BLUR_PROB_HELP = (
    "The probability of applying random visual blur to a training image."
)

NOISE_PROB_HELP = (
    "The probability of applying random Gaussian noise to a training image."
)

CONTRAST_PROB_HELP = (
    "The probability of applying random brightness/contrast adjustments to a training image."
)

DROPOUT_PROB_HELP = (
    "The probability of applying random rectangular cutouts (occlusions) to train the model to handle occluded keypoints."
)

MAX_HOLES_HELP = (
    "The maximum number of random occlusion cutouts per image."
)

REID_HELP = (
    "If checked, trains an embedder head on the backend model to visually re-identify individual animals across frames."
)

TD_MARGIN_HELP = (
    "Scale factor to expand cropped animal bounding boxes to ensure no body parts are cut off at the edges."
)

TD_MIN_CONF_HELP = (
    "Minimum confidence required for keypoints in the training dataset to be included in loss calculations."
)

TD_CROP_STRATEGY_HELP = (
    "How crop boxes are sized:\n\n"
    "• Dynamic: Crops scale to the size of the detected animal.\n"
    "• Fixed Size: Crops are resized to a fixed pixel width and height."
)

AUG_HELP_TEXTS = {
    "Rotation Limit (+/- deg):": ROTATION_LIMIT_HELP,
    "Scale Limit (+/-):": SCALE_LIMIT_HELP,
    "Apply Probability:": GEOM_PROB_HELP,
    "Blur Prob:": BLUR_PROB_HELP,
    "Noise Prob:": NOISE_PROB_HELP,
    "Contrast Prob:": CONTRAST_PROB_HELP,
    "Dropout Prob:": DROPOUT_PROB_HELP,
    "Max Holes:": MAX_HOLES_HELP
}


class PoseEstimationTrainingSettingsWidget(QWidget):
    """
    A widget to configure native WHISKER training parameters.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Helper Factory for Concise Instantiation ---
        def make_spin(cls, rng, val, step=None, decimals=None, suffix="", tooltip=""):
            w = cls()
            if isinstance(rng[0], float): w.setRange(*rng)
            else: w.setRange(*rng)
            w.setValue(val)
            if step is not None: w.setSingleStep(step)
            if decimals is not None: w.setDecimals(decimals)
            if suffix: w.setSuffix(suffix)
            if tooltip: w.setToolTip(tooltip)
            return w

        # ==========================================
        # TAB 1: GENERAL (Two-Column Layout)
        # ==========================================
        self.general_tab = QWidget()
        gen_main_layout = QHBoxLayout(self.general_tab)
        
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        
        # --- Left Column: Architecture & Losses ---
        arch_group = QGroupBox("Architecture")
        arch_layout = QFormLayout(arch_group)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["vitpose", "mobilenet", "hrnet"]) 
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        
        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems(["Global", "Top-Down"])

        self.backbone_combo = QComboBox()
        self._populate_backbones("vitpose")
        
        arch_layout.addRow(FormLabelWithHelp("Model Architecture:", MODEL_ARCH_HELP), self.architecture_combo)
        arch_layout.addRow(FormLabelWithHelp("Model Type:", MODEL_TYPE_HELP), self.model_type_combo)
        arch_layout.addRow(FormLabelWithHelp("Backbone:", BACKBONE_HELP), self.backbone_combo)
        left_col.addWidget(arch_group)

        img_size_group = QGroupBox("Model Input Dimensions")
        img_size_layout = QFormLayout(img_size_group)
        self.img_width_spin = QSpinBox()
        self.img_width_spin.setRange(1, 4096)
        self.img_width_spin.setValue(192)
        self.img_width_spin.setSingleStep(1)
        self.img_height_spin = QSpinBox()
        self.img_height_spin.setRange(1, 4096)
        self.img_height_spin.setValue(192)
        self.img_height_spin.setSingleStep(1)
        img_size_layout.addRow(FormLabelWithHelp("Image Width:", IMG_WIDTH_HELP), self.img_width_spin)
        img_size_layout.addRow(FormLabelWithHelp("Image Height:", IMG_HEIGHT_HELP), self.img_height_spin)
        left_col.addWidget(img_size_group)

        cv_group = QGroupBox("Loss Settings")
        cv_layout = QFormLayout(cv_group)
        self.loss_type_combo = QComboBox()
        self.loss_type_combo.addItem("Standard MSE", "mse")
        self.loss_type_combo.addItem("Adaptive Wing Loss (Robust)", "awing")
        self.udp_checkbox = QCheckBox("Unbiased Data Processing (UDP)")
        self.udp_checkbox.setToolTip(UDP_HELP)
        self.attention_combo = QComboBox()
        self.attention_combo.addItem("None", None)
        self.attention_combo.addItem("scSE (Spatial & Channel Attention)", "scse")
        
        cv_layout.addRow(FormLabelWithHelp("Base Loss Function:", BASE_LOSS_HELP), self.loss_type_combo)
        cv_layout.addRow(FormLabelWithHelp("Decoder Attention:", DECODER_ATTN_HELP), self.attention_combo)
        cv_layout.addRow(self.udp_checkbox)
        
        self.variance_loss_weight_spin = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.01, step=0.005, decimals=4)
        cv_layout.addRow(FormLabelWithHelp("Variance Penalty Weight:", VARIANCE_PENALTY_HELP), self.variance_loss_weight_spin)

        self.geom_loss_weight_spin = make_spin(QDoubleSpinBox, (0.0, 100.0), 0.05, step=0.01)
        cv_layout.addRow(FormLabelWithHelp("Geometric Loss Weight (Max):", GEOM_LOSS_HELP), self.geom_loss_weight_spin)

        self.geom_warmup_spin = make_spin(QSpinBox, (0, 100), 5, suffix=" epochs")
        cv_layout.addRow(FormLabelWithHelp("Geometric Warmup:", GEOM_WARMUP_HELP), self.geom_warmup_spin)
        left_col.addWidget(cv_group)

        self.paf_group = QGroupBox("Part Affinity Fields (Multi-Instance)")
        self.paf_group.setCheckable(True)
        self.paf_group.setChecked(False)
        paf_layout = QFormLayout(self.paf_group)
        self.paf_thickness_spin = make_spin(QSpinBox, (1, 20), 3, suffix=" px")
        self.paf_loss_weight_spin = make_spin(QDoubleSpinBox, (0.0, 100.0), 1.0, step=0.1)
        paf_layout.addRow(FormLabelWithHelp("Limb Thickness:", LIMB_THICKNESS_HELP), self.paf_thickness_spin)
        paf_layout.addRow(FormLabelWithHelp("Loss Weight:", LIMB_LOSS_WEIGHT_HELP), self.paf_loss_weight_spin)
        left_col.addWidget(self.paf_group)
        left_col.addStretch()

        # --- Right Column: Training & Scheduler ---
        train_group = QGroupBox("Training Hyperparameters")
        train_layout = QFormLayout(train_group)
        self.epochs_spin = make_spin(QSpinBox, (1, 1000), 50)
        self.epochs_per_checkpoint = make_spin(QSpinBox, (1, 1000), 5)
        self.batch_size_spin = make_spin(QSpinBox, (1, 128), 8)
        self.lr_spin = make_spin(QDoubleSpinBox, (0.00001, 0.1), 0.0001, step=0.0001, decimals=5)
        self.early_stopping_patience_spin = make_spin(QSpinBox, (1, 100), 10)
        
        train_layout.addRow(FormLabelWithHelp("Epochs (Max):", EPOCHS_HELP), self.epochs_spin)
        train_layout.addRow(FormLabelWithHelp("Epochs per Checkpoint:", EPOCHS_PER_CHECKPOINT_HELP), self.epochs_per_checkpoint)
        train_layout.addRow(FormLabelWithHelp("Batch Size:", BATCH_SIZE_HELP), self.batch_size_spin)
        train_layout.addRow(FormLabelWithHelp("Initial Learning Rate:", LR_HELP), self.lr_spin)
        train_layout.addRow(FormLabelWithHelp("Early Stopping Patience:", EARLY_STOPPING_HELP), self.early_stopping_patience_spin)
        right_col.addWidget(train_group)

        sched_group = QGroupBox("LR Scheduler")
        sched_layout = QFormLayout(sched_group)
        self.sched_patience_spin = make_spin(QSpinBox, (1, 50), 3)
        self.sched_factor_spin = make_spin(QDoubleSpinBox, (0.1, 0.9), 0.5, step=0.1)
        self.min_lr_spin = make_spin(QDoubleSpinBox, (1e-8, 1e-3), 1e-6, step=1e-6, decimals=8)
        
        sched_layout.addRow(FormLabelWithHelp("Patience (Epochs):", SCHED_PATIENCE_HELP), self.sched_patience_spin)
        sched_layout.addRow(FormLabelWithHelp("Decay Factor:", SCHED_FACTOR_HELP), self.sched_factor_spin)
        sched_layout.addRow(FormLabelWithHelp("Min LR:", MIN_LR_HELP), self.min_lr_spin)
        right_col.addWidget(sched_group)
        right_col.addStretch()

        gen_main_layout.addLayout(left_col)
        gen_main_layout.addLayout(right_col)
        self.tabs.addTab(self.general_tab, "General")

        # ==========================================
        # TAB 2: AUGMENTATION
        # ==========================================
        self.aug_tab = QWidget()
        aug_layout = QVBoxLayout(self.aug_tab)
        
        self.aug_enabled_cb = QCheckBox("Enable Data Augmentation")
        self.aug_enabled_cb.setChecked(True)
        self.aug_enabled_cb.toggled.connect(self._on_aug_toggled)
        aug_layout.addWidget(self.aug_enabled_cb)

        self.aug_content_widget = QWidget()
        aug_form = QHBoxLayout(self.aug_content_widget) # Using HBox for Augmentation groups too

        def make_aug_group(title: str, fields: list) -> QGroupBox:
            grp = QGroupBox(title)
            lay = QFormLayout(grp)
            for label, widget in fields:
                help_text = AUG_HELP_TEXTS.get(label)
                if help_text:
                    lay.addRow(FormLabelWithHelp(label, help_text), widget)
                else:
                    lay.addRow(label, widget)
            return grp

        self.aug_rotate_limit = make_spin(QSpinBox, (0, 180), 30)
        self.aug_scale_limit = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.2)
        self.aug_geom_prob = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.5)
        aug_form.addWidget(make_aug_group("Geometric", [
            ("Rotation Limit (+/- deg):", self.aug_rotate_limit),
            ("Scale Limit (+/-):", self.aug_scale_limit),
            ("Apply Probability:", self.aug_geom_prob)
        ]))

        self.aug_blur_prob = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.3)
        self.aug_noise_prob = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.3)
        self.aug_contrast_prob = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.3)
        
        self.aug_dropout_prob = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.2)
        self.aug_dropout_holes = make_spin(QSpinBox, (1, 20), 5)
        
        vis_occ_col = QVBoxLayout()
        vis_occ_col.addWidget(make_aug_group("Visual", [
            ("Blur Prob:", self.aug_blur_prob),
            ("Noise Prob:", self.aug_noise_prob),
            ("Contrast Prob:", self.aug_contrast_prob)
        ]))
        vis_occ_col.addWidget(make_aug_group("Occlusion", [
            ("Dropout Prob:", self.aug_dropout_prob),
            ("Max Holes:", self.aug_dropout_holes)
        ]))
        vis_occ_col.addStretch()
        aug_form.addLayout(vis_occ_col)

        aug_layout.addWidget(self.aug_content_widget)
        self.tabs.addTab(self.aug_tab, "Augmentation")

        # ==========================================
        # TAB 3: TOP-DOWN
        # ==========================================
        self.topdown_tab = QWidget()
        td_layout = QFormLayout(self.topdown_tab)
        
        self.td_margin_spin = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.1, step=0.05)
        self.td_min_conf_spin = make_spin(QDoubleSpinBox, (0.0, 1.0), 0.0, step=0.05)
        
        self.td_size_group = QButtonGroup(self)
        self.td_dynamic_radio = QRadioButton("Dynamic (Auto-Aspect Ratio)")
        self.td_fixed_radio = QRadioButton("Fixed Size")
        self.td_dynamic_radio.setChecked(True)
        self.td_size_group.addButton(self.td_dynamic_radio)
        self.td_size_group.addButton(self.td_fixed_radio)
        
        self.td_fixed_w_spin = make_spin(QSpinBox, (1, 4096), 256)
        self.td_fixed_w_spin.setEnabled(False)
        self.td_fixed_h_spin = make_spin(QSpinBox, (1, 4096), 256)
        self.td_fixed_h_spin.setEnabled(False)
        
        self.td_fixed_radio.toggled.connect(self.td_fixed_w_spin.setEnabled)
        self.td_fixed_radio.toggled.connect(self.td_fixed_h_spin.setEnabled)

        self.train_reid_chk = QCheckBox("Enable Visual Re-ID Training")
        self.train_reid_chk.setToolTip(REID_HELP)

        td_layout.addRow("", self.train_reid_chk)
        td_layout.addRow(FormLabelWithHelp("Bounding Box Margin:", TD_MARGIN_HELP), self.td_margin_spin)
        td_layout.addRow(FormLabelWithHelp("Min Keypoint Confidence (Training):", TD_MIN_CONF_HELP), self.td_min_conf_spin)
        td_layout.addRow(FormLabelWithHelp("Crop Sizing Strategy:", TD_CROP_STRATEGY_HELP), self.td_dynamic_radio)
        td_layout.addRow("", self.td_fixed_radio)
        td_layout.addRow("Fixed Width:", self.td_fixed_w_spin)
        td_layout.addRow("Fixed Height:", self.td_fixed_h_spin)
        
        self.tabs.addTab(self.topdown_tab, "Top-Down")
        
        # Disabled by default until architecture is selected
        self.set_top_down_enabled(False)
        self.architecture_combo.currentTextChanged.connect(lambda text: self.set_top_down_enabled(text == "Top-Down"))

    def set_top_down_enabled(self, enabled: bool):
        """Enables or disables the Top-Down tab based on architecture selection."""
        idx = self.tabs.indexOf(self.topdown_tab)
        self.tabs.setTabVisible(idx, enabled)

    def _on_model_type_changed(self, model_type: str):
        self._populate_backbones(model_type)

    def _on_aug_toggled(self, checked: bool):
        self.aug_content_widget.setEnabled(checked)

    def _populate_backbones(self, model_type: str):
        self.backbone_combo.clear()
        backbones = {
            "vitpose": ["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224", "vit_large_patch16_224"],
            "mobilenet": ["resnet50", "mobilenet_v2", "tu-mobilenetv3_large_100", "tu-mobilenetv3_small_100"],
            "hrnet": ["hrnet_w18", "hrnet_w32", "hrnet_w48"]
        }
        self.backbone_combo.addItems(backbones.get(model_type, []))

    def get_params(self) -> Dict[str, Any]:
        """Returns the current settings."""
        return {
            "model_architecture": self.architecture_combo.currentText(),
            "model_type": self.model_type_combo.currentText(),
            "backbone": self.backbone_combo.currentText(),
            "img_size": [
                self.img_height_spin.value(),
                self.img_width_spin.value(),
            ],
            "loss_type": self.loss_type_combo.currentData(),
            "use_udp": self.udp_checkbox.isChecked(),
            "decoder_attention_type": self.attention_combo.currentData(),
            "enable_paf": self.paf_group.isChecked(),
            "paf_thickness": self.paf_thickness_spin.value(),
            "paf_loss_weight": self.paf_loss_weight_spin.value(),
            "variance_loss_weight": self.variance_loss_weight_spin.value(),
            "geometric_loss_weight": self.geom_loss_weight_spin.value(),
            "geometric_warmup_epochs": self.geom_warmup_spin.value(),
            "epochs": self.epochs_spin.value(),
            "epochs_per_checkpoint": self.epochs_per_checkpoint.value(),
            "patience": self.early_stopping_patience_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "lr": self.lr_spin.value(),
            "scheduler_patience": self.sched_patience_spin.value(),
            "scheduler_factor": self.sched_factor_spin.value(),
            "min_lr": self.min_lr_spin.value(),
            "aug_enabled": self.aug_enabled_cb.isChecked(),
            "aug_rotate_limit": self.aug_rotate_limit.value(),
            "aug_scale_limit": self.aug_scale_limit.value(),
            "aug_geom_prob": self.aug_geom_prob.value(),
            "aug_blur_prob": self.aug_blur_prob.value(),
            "aug_noise_prob": self.aug_noise_prob.value(),
            "aug_contrast_prob": self.aug_contrast_prob.value(),
            "aug_dropout_prob": self.aug_dropout_prob.value(),
            "aug_dropout_holes": self.aug_dropout_holes.value(),
            "top_down_config": {
                "enable_reid": self.train_reid_chk.isChecked(),
                "bbox_margin": self.td_margin_spin.value(),
                "bbox_min_keypoint_confidence": self.td_min_conf_spin.value(),
                "bbox_fixed_size": (self.td_fixed_w_spin.value(), self.td_fixed_h_spin.value()) if self.td_fixed_radio.isChecked() else None
            }
        }

    def set_params(self, params: Dict[str, Any]):
        """Populates the widget with the provided settings."""
        if not params:
            return

        try:
            if "model_architecture" in params:
                idx = self.architecture_combo.findText(params["model_architecture"])
                if idx >= 0:
                    self.architecture_combo.setCurrentIndex(idx)
                    self.set_top_down_enabled(params["model_architecture"] == "Top-Down")

            if "model_type" in params:
                idx = self.model_type_combo.findText(params["model_type"])
                if idx >= 0:
                    self.model_type_combo.setCurrentIndex(idx)
                    self._populate_backbones(params["model_type"])

            if "backbone" in params:
                idx = self.backbone_combo.findText(params["backbone"])
                if idx >= 0: self.backbone_combo.setCurrentIndex(idx)

            if "img_size" in params:
                height, width = params["img_size"]
                self.img_width_spin.setValue(width)
                self.img_height_spin.setValue(height)

            if "loss_type" in params:
                idx = self.loss_type_combo.findData(params["loss_type"])
                if idx >= 0: self.loss_type_combo.setCurrentIndex(idx)

            if "use_udp" in params: self.udp_checkbox.setChecked(bool(params["use_udp"]))
            
            if "decoder_attention_type" in params:
                idx = self.attention_combo.findData(params["decoder_attention_type"])
                if idx >= 0: self.attention_combo.setCurrentIndex(idx)
            
            if "enable_paf" in params: self.paf_group.setChecked(bool(params["enable_paf"]))
            if "paf_thickness" in params: self.paf_thickness_spin.setValue(int(params["paf_thickness"]))
            if "paf_loss_weight" in params: self.paf_loss_weight_spin.setValue(float(params["paf_loss_weight"]))
            if "variance_loss_weight" in params: self.variance_loss_weight_spin.setValue(float(params["variance_loss_weight"]))
            if "geometric_loss_weight" in params: self.geom_loss_weight_spin.setValue(float(params["geometric_loss_weight"]))
            if "geometric_warmup_epochs" in params: self.geom_warmup_spin.setValue(int(params["geometric_warmup_epochs"]))

            if "epochs" in params: self.epochs_spin.setValue(int(params["epochs"]))
            if "epochs_per_checkpoint" in params: self.epochs_per_checkpoint.setValue(int(params["epochs_per_checkpoint"]))
            if "patience" in params: self.early_stopping_patience_spin.setValue(int(params["patience"]))
            if "batch_size" in params: self.batch_size_spin.setValue(int(params["batch_size"]))
            if "lr" in params: self.lr_spin.setValue(float(params["lr"]))
            if "scheduler_patience" in params: self.sched_patience_spin.setValue(int(params["scheduler_patience"]))
            if "scheduler_factor" in params: self.sched_factor_spin.setValue(float(params["scheduler_factor"]))
            if "min_lr" in params: self.min_lr_spin.setValue(float(params["min_lr"]))

            if "aug_enabled" in params: self.aug_enabled_cb.setChecked(bool(params["aug_enabled"]))

            aug_mapping = {
                "aug_rotate_limit": self.aug_rotate_limit, "aug_scale_limit": self.aug_scale_limit,
                "aug_geom_prob": self.aug_geom_prob, "aug_blur_prob": self.aug_blur_prob,
                "aug_noise_prob": self.aug_noise_prob, "aug_contrast_prob": self.aug_contrast_prob,
                "aug_dropout_prob": self.aug_dropout_prob, "aug_dropout_holes": self.aug_dropout_holes
            }

            for key, widget in aug_mapping.items():
                if key in params:
                    if isinstance(widget, QSpinBox): widget.setValue(int(params[key]))
                    elif isinstance(widget, QDoubleSpinBox): widget.setValue(float(params[key]))

            if "top_down_config" in params and params["top_down_config"]:
                td = params["top_down_config"]
                if "bbox_margin" in td: self.td_margin_spin.setValue(float(td["bbox_margin"]))
                if "bbox_min_keypoint_confidence" in td: self.td_min_conf_spin.setValue(float(td["bbox_min_keypoint_confidence"]))
                
                if td.get("bbox_fixed_size"):
                    self.td_fixed_radio.setChecked(True)
                    w, h = td["bbox_fixed_size"] if isinstance(td["bbox_fixed_size"], (list, tuple)) else (td["bbox_fixed_size"], td["bbox_fixed_size"])
                    self.td_fixed_w_spin.setValue(int(w))
                    self.td_fixed_h_spin.setValue(int(h))
                else:
                    self.td_dynamic_radio.setChecked(True)
                    
        except Exception as e:
            logging.error(f"Error setting params in PoseEstimationTrainingSettingsWidget: {e}")