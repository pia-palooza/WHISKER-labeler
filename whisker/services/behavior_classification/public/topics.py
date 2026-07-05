from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple

class BehaviorModelConfig(BaseModel):
    """
    Configuration for a trained behavior model.
    This is saved as model_config.json alongside the model.h5 file.
    """
    model_type: str = Field(
        ..., 
        description="Type of model, e.g., 'single-animal-lstm' or 'multi-animal-lstm'"
    )
    window_size: int = Field(
        ..., 
        description="Number of frames in an input window (T)"
    )
    roi_indicator: Optional[str] = Field(
        default=None,
        description="Optional behavior name used to indicate a Region of Interest. If provided, training and inference will only occur during bouts of this behavior."
    )
    behaviors: List[str] = Field(
        ..., 
        description="Ordered list of behavior names in the model's output layer"
    )
    model_identities: List[str] = Field(
        default_factory=list,
        description="Ordered list of model identities (e.g., ['Mouse A', 'Mouse B'])."
    )
    root_individual_model_id: Optional[str] = Field(
        default=None,
        description="The Model Identity that serves as the coordinate origin (0,0)."
    )
    root_bodypart: Optional[str] = Field(
        default=None,
        description="The body part used as the reference point for spatial features."
    )
    heading_axis: Optional[Tuple[str, str]] = Field(
        default=None,
        description="The (From, To) body parts defining the forward heading."
    )
    skeleton: Optional[List[Tuple[str, str]]] = Field(
        default=None,
        description="List of (BP_A, BP_B) pairs defining the edges for distance features."
    )
    template_coords: Optional[Dict[str, Tuple[float, float]]] = Field(
        default=None,
        description="The {body_part: (x,y)} coordinates of the template skeleton used for scaling."
    )

    # MS-TCN Specific Hyperparameters
    mstcn_stages: int = Field(
        default=2,
        description="Number of refinement stages in the MS-TCN architecture."
    )
    mstcn_layers_per_stage: int = Field(
        default=7,
        description="Number of dilated residual blocks per stage."
    )
    mstcn_filters: int = Field(
        default=64,
        description="Number of feature channels in the convolutional layers."
    )
    epochs: int = Field(
        default=50,
        description="Maximum number of training epochs."
    )
    patience: int = Field(
        default=10,
        description="Number of epochs to wait for improvement before early stopping."
    )
    learning_rate: float = Field(
        default=0.0001,
        description="The initial learning rate for the optimizer."
    )
    focal_alpha: float = Field(
        default=0.5,
        description="Alpha parameter for Focal Loss (balances precision/recall)."
    )
    warmup_epochs: int = Field(
        default=5,
        description="Number of epochs for linear learning rate warmup."
    )
    max_vel: float = Field(
        default=50.0,
        description="Maximum allowed pixel velocity before a jump is filtered as an anomaly."
    )
    confidence_smoothing_window: int = Field(
        default=5,
        description="Window size for the rolling median filter applied to the confidence channel."
    )
    proximity_threshold: float = Field(
        default=300.0,
        description="Distance threshold (in pixels) below which subjects are considered in proximity."
    )
    proximity_neg_ratio: float = Field(
        default=0.5,
        description="Fraction of negative samples that must be sampled from proximity (hard negatives)."
    )

class BehaviorTrainingParams(BaseModel):
    run_name: str
    config: BehaviorModelConfig
    identity_map: Dict[str, str]
    root_bodypart: str
    source_run_name: str
    training_targets: List[Tuple[str, str]]
    resume_run_name: Optional[str] = None
    replicate_run_name: Optional[str] = None
