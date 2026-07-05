import json
import os
from pathlib import Path
from typing import Optional

class SessionManager:
    SESSION_FILE = Path.home() / ".whisker" / "session.json"

    @classmethod
    def get_active_workspace_path(cls) -> Optional[Path]:
        if not cls.SESSION_FILE.exists():
            return None
        try:
            with open(cls.SESSION_FILE, "r") as f:
                data = json.load(f)
                path = data.get("active_workspace")
                return Path(path) if path else None
        except Exception:
            return None

    @classmethod
    def set_active_workspace_path(cls, path: Path):
        cls.SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"active_workspace": str(path.resolve())}
        with open(cls.SESSION_FILE, "w") as f:
            json.dump(data, f)

    @classmethod
    def clear_active_workspace(cls):
        if cls.SESSION_FILE.exists():
            cls.SESSION_FILE.unlink()
