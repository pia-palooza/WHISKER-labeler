import enum

class Workflow(enum.Enum):
    POSE_ESTIMATION = "Pose Estimation"
    BEHAVIOR_CLASSIFICATION = "Behavior Classification"
    ROI_TRACKING = "ROI Tracking"

    def to_display_name(self) -> str:
        return self.value

    def to_var_name(self) -> str:
        return self.name.lower().replace(' ', '_')
