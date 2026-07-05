from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple

class PoseTrainingParams(BaseModel):
    run_name: str
    dataset_names: List[str]
    individuals: List[str]
    bodyparts: List[str]
    skeleton: List[Tuple[str, str]] = []
    backend: str = "WHISKER"
    backend_params: Dict[str, Any] = {}
    resume_run_name: Optional[str] = None

class PosePredictionParams(BaseModel):
    run_name: str
    video_paths: List[str] = []
    dataset_names: List[str] = []
    backend: str = "WHISKER"
    detector_run_name: Optional[str] = None
    detector_conf: float = 0.3
    max_candidates: int = 10
    store_debug_info: bool = False
    batch_size: int = 16

class PoseEvaluationParams(BaseModel):
    run_name: str
    dataset_name: str
    pck_threshold: float = 0.05
    data_split: str = "all"
    swap_identities: bool = False
    calculate_purity: bool = False

class PoseExportParams(BaseModel):
    pose_labels_set_name: str
    output_path: str
    flatten: bool = False

class LabelImportParams(BaseModel):
    dataset_name: str
    project_name: str
    backend: str
    pose_labels_path: str
    force_overwrite: bool = False

