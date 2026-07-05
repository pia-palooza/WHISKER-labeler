from PyQt6.QtWidgets import QMessageBox, QWidget
from PyQt6.QtCore import Qt
import logging

def check_gpu_availability(check_torch: bool = True, check_tf: bool = True) -> tuple[bool, list[str]]:
    details = []
    gpu_found = True

    if check_torch:
        try:
            import torch
            if not torch.cuda.is_available():
                gpu_found = False
                details.append("• PyTorch: CUDA is not initialized or no compatible NVIDIA GPU was found.")
        except ImportError:
            details.append("• PyTorch: Library not installed.")

    if check_tf:
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                gpu_found = False
                details.append("• TensorFlow: No GPU devices detected via the current environment.")
        except ImportError:
            details.append("• TensorFlow: Library not installed.")

    return gpu_found, details

def warn_if_no_gpu(parent: QWidget, check_torch: bool = True, check_tf: bool = True) -> bool:
    has_gpu, issues = check_gpu_availability(check_torch, check_tf)
    
    if has_gpu or not issues:
        return True

    # Detailed technical but accessible message
    issue_text = "\n".join(issues)
    msg = (
        f"<b>Hardware Acceleration Unvailable</b><br><br>"
        f"{issue_text}<br><br>"
        f"<b>Why is this happening?</b><br>"
        f"1. <b>Drivers:</b> Your NVIDIA drivers may be outdated or missing.<br>"
        f"2. <b>CUDA Toolkit:</b> The required CUDA libraries might not be installed or match your software version.<br>"
        f"3. <b>Hardware:</b> An NVIDIA GPU may not be present or is currently disabled.<br><br>"
        f"<b>Troubleshooting:</b><br>"
        f"Check that your GPU is recognized in the Windows Task Manager (Performance tab) "
        f"and ensure you have the latest drivers from the NVIDIA website.<br><br>"
        f"<i>Running on the CPU will be significantly slower. Proceed anyway?</i>"
    )
    
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Warning)
    msg_box.setWindowTitle("GPU Acceleration Warning")
    msg_box.setTextFormat(Qt.TextFormat.RichText)
    msg_box.setText(msg)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    msg_box.setDefaultButton(QMessageBox.StandardButton.No)
    
    return msg_box.exec() == QMessageBox.StandardButton.Yes