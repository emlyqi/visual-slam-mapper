import cv2
import numpy as np


# module-level ORB detector, reused across frames (faster than recreating per call)
_orb_detector = None


def _get_orb(n_features):
    """Lazy-init module-level ORB detector. Recreates if n_features changes."""
    global _orb_detector
    if _orb_detector is None or _orb_detector.getMaxFeatures() != n_features:
        _orb_detector = cv2.ORB_create(n_features)
    return _orb_detector


def detect_features(image, n_features=2000, return_descriptors=False):
    """
    Detects ORB keypoints in the given image.
    Returns:
        points: (N, 2) float32 array of keypoint pixel coordinates
        descriptors: (N, 32) uint8 array of ORB descriptors (if return_descriptors=True)
    """
    
    orb = _get_orb(n_features)

    if return_descriptors:
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None:
            # no keypoints found, return empty arrays
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 32), dtype=np.uint8)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points, descriptors
    else: 
        keypoints = orb.detect(image, None)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points


def track_features(prev_image, curr_image, prev_points):
    """
    Tracks features from the previous image to the current image using Lucas-Kanade optical flow.
    Returns (M, 2) tracked points + (M, 2) corresponding original points,
    after filtering. M <= N.
    """
    if len(prev_points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    
    p0 = prev_points.reshape(-1, 1, 2).astype(np.float32) # OpenCV wants (N, 1, 2) for points

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    # forward: prev -> curr
    p1, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, p0, None, **lk_params)

    # backward: curr -> prev
    p0_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_image, prev_image, p1, None, **lk_params)

    # round-trip (forward-backward) error: euclidean distance between p0 and p0_back
    fb_error = np.linalg.norm(p0 - p0_back, axis=2).flatten()

    # combined mask: valid forward, valid backward, low fb error
    good = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1) & (fb_error < 1.0)

    prev_kept = p0[good].reshape(-1, 2)
    curr_kept = p1[good].reshape(-1, 2)
    indices = np.where(good)[0]

    return prev_kept, curr_kept, indices