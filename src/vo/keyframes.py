import numpy as np


def should_make_keyframe(T_kf_to_curr, n_tracked, n_kf_features,
                        trans_thresh=1.0, rot_thresh_deg=10.0, feature_ratio_thresh=0.7):
    """
    Decide whether to make a new keyframe based on motion and tracking quality.
    Args:
        T_kf_to_curr: SE(3) transform from keyframe to current frame (4x4 numpy array)
        n_tracked: number of features successfully tracked from keyframe to current frame
        n_kf_features: number of features detected at last keyframe
    """

    translation = np.linalg.norm(T_kf_to_curr[:3, 3]) # euclidean distance of translation vector

    # rotation magnitude from rotation matrix:
    # angle = arccos((trace(R) - 1) / 2)
    R = T_kf_to_curr[:3, :3]
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # avoid arccos NaN from float noise
    rotation_deg = np.degrees(np.arccos(cos_angle))

    feature_ratio = n_tracked / max(n_kf_features, 1)  # avoid div by zero

    return (translation > trans_thresh or
            rotation_deg > rot_thresh_deg or
            feature_ratio < feature_ratio_thresh)
    