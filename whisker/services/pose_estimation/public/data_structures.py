# UPDATE_FILE: whisker/core/workflows/pose_estimation/data_structures.py
import enum
from pathlib import Path
from typing import Optional, List, Tuple, Union, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator
import logging

class VisibilityFlag(int, enum.Enum):
    NOT_VISIBLE = 0
    VISIBLE = 1

class PoseDataset:
    _HDF_KEY = "keypoints"

    def __init__(
        self,
        keypoint_data: Optional[pd.DataFrame] = None,
        body_parts: List[str] = [],
        individuals: List[str] = [],
    ):
        self.keypoint_data = (
            keypoint_data if keypoint_data is not None
            else self._create_empty_df()
        )
        self.body_parts = body_parts
        self.individuals = individuals

    def _create_empty_df(self) -> pd.DataFrame:
        index_names = ['frame_index', 'individual_id', 'body_part']
        columns = ['x', 'y', 'c'] 
        
        empty_index = pd.MultiIndex.from_tuples([], names=index_names)
        
        column_dtypes = {
            'x': 'float32',
            'y': 'float32',
            'c': 'float32',
        }
        
        df = pd.DataFrame(index=empty_index, columns=columns).astype(column_dtypes)
        
        new_levels = []
        for name in index_names:
            level_dtype = 'category' if name in ['individual_id', 'body_part'] else 'object'
            new_levels.append(df.index.get_level_values(name).astype(level_dtype))

        df.index = pd.MultiIndex.from_arrays(new_levels, names=index_names)
        
        df.index = df.index.set_levels(
            df.index.get_level_values('individual_id').astype('category'),
            level='individual_id'
        )
        df.index = df.index.set_levels(
            df.index.get_level_values('body_part').astype('category'),
            level='body_part'
        )
        
        return df

    @classmethod
    def from_file(cls, file_path: Path):
        try:
            df = pd.read_hdf(file_path, key=cls._HDF_KEY, mode='r')
            
            body_parts = []
            individuals = []
            
            with pd.HDFStore(file_path, 'r') as store:
                if 'metadata/body_parts' in store:
                    body_parts = store.get('metadata/body_parts').tolist()
                elif not df.empty and 'body_part' in df.index.names:
                    logging.warning("Metadata 'body_parts' not found. Inferring.")
                    body_parts = df.index.get_level_values('body_part').unique().tolist()
                
                if 'metadata/individuals' in store:
                    individuals = store.get('metadata/individuals').tolist()
                elif not df.empty and 'individual_id' in df.index.names:
                    logging.warning("Metadata 'individuals' not found. Inferring.")
                    individuals = df.index.get_level_values('individual_id').unique().tolist()

        except FileNotFoundError:
            raise ValueError(f"File not found at {file_path}")
        except Exception as e:
            raise IOError(f"Error reading HDF5 file: {e}")

        return cls(keypoint_data=df, body_parts=body_parts, individuals=individuals)

    def to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.keypoint_data.to_hdf(
            file_path, 
            key=self._HDF_KEY, 
            mode='w',            
            format='table',      
            data_columns=True,   
            complevel=9,         
            complib='blosc:lz4', 
        )
        with pd.HDFStore(file_path, 'a') as store: 
            pd.Series(self.body_parts).to_hdf(store, key='metadata/body_parts')
            pd.Series(self.individuals).to_hdf(store, key='metadata/individuals')

    def has_labeled_keypoints(self, threshold: float = 0.0) -> bool:
        if self.keypoint_data.empty: return False
        if 'c' not in self.keypoint_data.columns: return True
        return bool((self.keypoint_data['c'] > threshold).any())

    def dump(self, output_path: str, flatten: bool = False):
        metadata_lines = [
            f"# body_parts:{','.join(self.body_parts)}",
            f"# individuals:{','.join(self.individuals)}"
        ]
        metadata_header = "\n".join(metadata_lines) + "\n"

        df_to_save: pd.DataFrame
        if flatten:
            if self.keypoint_data.empty:
                df_to_save = pd.DataFrame(columns=[])
            else:
                df_flat = self.keypoint_data.unstack(level=['individual_id', 'body_part'])
                df_flat.columns = df_flat.columns.reorder_levels([1, 2, 0])
                df_flat.columns.names = ['individual_id', 'body_part', 'coords']
                df_flat = df_flat.sort_index(axis=1)
                df_flat.columns = ['_'.join(col) for col in df_flat.columns]
                df_to_save = df_flat
        else:
            df_to_save = self.keypoint_data

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(metadata_header)
                is_completely_empty = (df_to_save.empty and len(df_to_save.columns) == 0)                
                if not is_completely_empty:
                    df_to_save.to_csv(f, index=True, lineterminator='\n')
        except IOError as e:
            logging.error(f"Error writing file: {e}")
            raise

    @property
    def frame_indices(self) -> list[str]:
        if self.keypoint_data.empty: return []
        return list(self.keypoint_data.index.get_level_values('frame_index').unique())

class PoseEstimationTopDownModelConfig(BaseModel):
    bbox_margin: float = 0.1
    bbox_min_keypoint_confidence: Optional[float] = 0.0
    bbox_fixed_size: Optional[Union[int, Tuple[int, int]]] = None

class PoseEstimationModelConfig(BaseModel):
    model_type: str = "vitpose"
    backbone: str = "vit_base_patch16_224"
    img_size: Tuple[int, int] = (192, 256)  # (height, width)
    use_udp: bool = False
    output_stride: int = 4
    decoder_attention_type: Optional[str] = None
    individuals: List[str] = []
    bodyparts: List[str] = []
    skeleton: List[Tuple[str, str]] = [] # Moved up from TrainingConfig
    model_architecture: Literal["Global", "Top-Down"] = "Global"
    top_down_config: Optional[PoseEstimationTopDownModelConfig] = None
    
    # --- PAF Configuration ---
    enable_paf: bool = False
    paf_thickness: int = 3

    @property
    def is_top_down(self) -> bool:
        return self.model_architecture == "Top-Down"

    @property
    def num_keypoints(self):
        if self.model_architecture == "Top-Down":
            return len(self.bodyparts)
        else:
            return len(self.individuals) * len(self.bodyparts)
            
    @property
    def num_limbs(self):
        """Returns the number of connections in the skeleton."""
        return len(self.skeleton)

class PoseEstimationModelTrainingConfig(PoseEstimationModelConfig):
    epochs: int = 20
    epochs_per_checkpoint: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    loss_type: str = "mse"
    
    # --- SoftArgmax Configuration ---
    soft_argmax_beta: float = 100.0
    soft_argmax_warmup_epochs: int = 5
    
    variance_loss_weight: float = 0.01
    geometric_loss_weight: float = 0.0
    geometric_warmup_epochs: int = 5
    
    # --- PAF Loss Config ---
    paf_loss_weight: float = 1.0
    
    aug_enabled: bool = False
    aug_geom_prob: float = 0.5
    aug_rotate_limit: int = 30
    aug_scale_limit: float = 0.2
    aug_blur_prob: float = 0.3
    aug_noise_prob: float = 0.3
    aug_contrast_prob: float = 0.3
    aug_dropout_prob: float = 0.2
    aug_dropout_holes: int = 5
    debug_mode: bool = False
    
    @model_validator(mode='after')
    def check_paf_requirements(self) -> 'PoseEstimationModelTrainingConfig':
        if self.enable_paf:
            if not self.skeleton:
                raise ValueError("PAFs are enabled ('enable_paf=True') but no skeleton structure was provided.")
            
            # Verify skeleton body parts exist
            if self.bodyparts:
                for bp1, bp2 in self.skeleton:
                    if bp1 not in self.bodyparts or bp2 not in self.bodyparts:
                        raise ValueError(f"Skeleton limb ({bp1}, {bp2}) contains parts not in 'bodyparts' list.")
        return self

class PoseEstimationModelInferenceConfig(PoseEstimationModelConfig):
    detector_run_name: Optional[str] = None

class AmendmentParams(BaseModel):
    # --- Global Graph Tracking ---
    enable_global_tracking: bool = False
    tracking_max_gap: int = 30
    tracking_motion_weight: float = 0.5
    edge_reward: float = 10000.0
    
    # Cost scaling and speeds
    tracking_max_speed: float = 150.0
    tracking_dist_cost_scale: float = 20.0
    tracking_dt_cost_scale: float = 100.0

    # --- Visual Re-ID ---
    enable_visual_reid: bool = False
    reid_visual_weight: float = 0.5    
    reid_sim_threshold: float = 0.4    
    reid_sample_step: int = 10  

    # --- Debugging ---
    store_debug_info: bool = False

    # --- Confidence Masking ---
    low_confidence_threshold: float = 0.3

    # --- Legacy Swap Detective Params ---
    correct_swaps: bool = False
    swap_min_distance: float = 50.0
    swap_window_size: int = 5        
    swap_cooldown: int = 30          
    
    clean_outliers: bool = True
    jump_threshold: float = 30.0

    # --- Kalman Smoothing Params ---
    smooth_kalman: bool = False
    kalman_process_noise: float = 0.01  # Q
    kalman_measurement_noise: float = 0.1  # R
    kalman_predict_gaps: bool = True

    fill_gaps: bool = True