import logging
import os
from pathlib import Path
from typing import Any, Dict

from whisker.core.constants import BackendEnum

class ThirdPartyManager:
    def __init__(self, base_dir: Path):     
        logging.debug(f"Initializing ThirdPartyManager at {base_dir}")
        self._base_dir = base_dir
        self._backends: Dict[BackendEnum, Any] = {}
        os.makedirs(self._base_dir, exist_ok=True)

    def get_backend_manager(
        self,
        backend: BackendEnum,
        manager_class: Any
    ):
        if not backend in self._backends:
            backend_base_dir = self._base_dir / backend.value
            logging.info(
                f"Initializing {manager_class} for {backend} "
                f"backend at {backend_base_dir}"
            )
            self._backends[backend] = manager_class(backend_base_dir)
        return self._backends[backend]