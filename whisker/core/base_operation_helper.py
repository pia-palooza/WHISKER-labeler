import logging
import os
import shutil
from pathlib import Path
from typing import  Optional, Callable

class BaseOperationHelper:
    """Base class for all operation helpers to share common utility methods."""
    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        logging.info(f"Initializing {self.__class__.__name__} at {self._base_dir}")
        os.makedirs(self._base_dir, exist_ok=True)
    
    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @staticmethod
    def _delete_if_existing(
        base_path: Path,
        dir_type_name: str,
        warn_if_exists: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """Utility function for deletion logic."""
        if base_path.exists():
            logging.info(f"Base path for {dir_type_name} ({base_path}) already exists!")
            if warn_if_exists:
                logging.info(f"Prompting user to confirm deletion of existing {dir_type_name}.")
                if not warn_if_exists(
                    f"Overwrite existing {dir_type_name} at {base_path}?\n"
                    "It will be completely deleted and all data will be lost.\n"
                ):
                    logging.info(f"Cancelled creation of {dir_type_name}.")
                    return False
            
            if base_path.is_dir():
                shutil.rmtree(base_path)
            else:
                os.remove(base_path)

            logging.info(f"Removed existing {dir_type_name} at {base_path}.")

        return True

