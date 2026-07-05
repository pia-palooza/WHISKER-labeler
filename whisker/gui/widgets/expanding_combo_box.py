from PyQt6.QtWidgets import QComboBox

class ExpandingComboBox(QComboBox):
    """
    A QComboBox that expands its dropdown list (popup) width to fit its contents,
    rather than limiting it to the width of the base widget.
    """
    def showPopup(self):
        width = self.view().sizeHintForColumn(0)

        if self.view().verticalScrollBar().isVisible():
            width += self.view().verticalScrollBar().sizeHint().width()

        width += 20 # Padding

        width = max(width, self.width())
        self.view().setMinimumWidth(width)
        super().showPopup()
