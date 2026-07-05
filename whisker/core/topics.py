import enum
from typing import Optional
from pydantic import BaseModel, Field

class SamplingTechnique(str, enum.Enum):
    UNIFORM = "Uniform"
    KMEANS_VISUAL = "K-Means (Visual Diversity)"
    POSE_KMEANS = "K-Means (Pose/Morphological)"

class SubsampleParams(BaseModel):
    source_dataset_name: str = Field(..., description="Name of the dataset to subsample")
    num_frames: int = Field(..., description="Number of frames to sample")
    technique: SamplingTechnique = Field(default=SamplingTechnique.UNIFORM, description="Sampling technique to use")
    target_dataset_name: Optional[str] = Field(None, description="Name of the new dataset. Auto-generated if not provided.")
