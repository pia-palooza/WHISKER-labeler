"""Make sure label edits are on disk before something reads them from disk.

An export copies the label files as they are saved, so edits still held in a labeling tab
would silently be left out of the package. Before an export, :func:`settle_unsaved_labels`
finds the tabs holding such edits, asks what to do, and saves them if asked.
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping

SAVE = "save"
SKIP = "skip"
CANCEL = "cancel"


def settle_unsaved_labels(views: Mapping[str, object], ask: Callable[[list], str]) -> bool:
    """True if the export may go ahead.

    ``views`` maps a view's name to the view. Those that report ``has_unsaved_labels()`` are
    listed to ``ask`` (by name), which answers :data:`SAVE`, :data:`SKIP` (export what is saved) or
    :data:`CANCEL`. On SAVE each one's ``save_labels()`` runs, and one failing (it reports why
    itself) stops the export, since going ahead would export a stale copy without saying so.
    """
    pending = {
        name: view for name, view in views.items()
        if hasattr(view, "has_unsaved_labels") and view.has_unsaved_labels()
    }
    if not pending:
        return True
    answer = ask(list(pending))
    if answer == SKIP:
        logging.info("Exporting without the unsaved label edits in: %s", ", ".join(pending))
        return True
    if answer != SAVE:
        return False
    for name, view in pending.items():
        if not view.save_labels():
            logging.warning("Export stopped: could not save the label edits in '%s'.", name)
            return False
    return True
