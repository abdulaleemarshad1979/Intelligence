"""Re-export of Pose Estimation adapters for backward compatibility."""

from app.adapters.pose.rtmpose import RTMPoseAdapter, COCO_KEYPOINTS
from app.adapters.pose.mmpose import MMPoseAdapter

MMPoseRTMPoseAdapter = RTMPoseAdapter

__all__ = ["RTMPoseAdapter", "MMPoseAdapter", "MMPoseRTMPoseAdapter", "COCO_KEYPOINTS"]
