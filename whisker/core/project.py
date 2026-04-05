from typing import Dict, List, Optional, Tuple

import json
from pydantic import BaseModel, Field


class Project(BaseModel):
    name: str
    body_parts: List[str] = Field(default_factory=list)
    skeleton: List[Tuple[str, str]] = Field(default_factory=list[tuple[str, str]])
    identities: List[str] = Field(default_factory=list)
    behaviors: List[str] = Field(default_factory=list)
    template_coords: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    heading_axis: Optional[Tuple[str, str]] = None

    @classmethod
    def from_json(cls, json_str: str) -> "Project":
        """
        Create a Project object from a JSON string.
        """
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            body_parts=data.get("body_parts", []),
            skeleton=data.get("skeleton", []),
            identities=data.get("identities", []),
            behaviors=data.get("behaviors", []),
            template_coords=data.get("template_coords", {}),
            heading_axis=tuple(data["heading_axis"]) if data.get("heading_axis") else None,
        )
