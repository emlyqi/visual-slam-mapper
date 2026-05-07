"""
Persistent landmark tracking for bundle adjustment.

A Landmark represents a single physical 3D point in the world that may be observed
across multiple keyframes. The LandmarkMap maintains the set of all landmarks and
their observations as VO runs.
"""

import json
from pathlib import Path
import numpy as np


class LandmarkMap:
    """
    Maintains a set of persistent 3D landmarks and their 2D observations.

    Each landmark has:
        - id: unique integer
        - position_world: (3,) float32, position in world frame
        - observations: list of (kf_frame_idx, pixel_xy) tuples

    Also maintains, per-keyframe, the local-feature-index → landmark_id mapping so
    that descriptor-matching against the previous keyframe can carry landmark IDs forward.
    """

    def __init__(self):
        self._landmarks = {} # landmark_id -> dict
        self._next_id = 0
        # per-keyframe: local feature index -> landmark_id
        self._kf_index_to_landmark = {} # kf_frame_idx -> {local_idx: landmark_id}

    def _create_landmark(self, position_world):
        lid = self._next_id
        self._next_id += 1
        self._landmarks[lid] = {
            'id': lid,
            'position_world': np.asarray(position_world, dtype=np.float32),
            'observations': [],
        }
        return lid

    def add_observation(self, landmark_id, kf_frame_idx, pixel_xy):
        self._landmarks[landmark_id]['observations'].append(
            (int(kf_frame_idx), np.asarray(pixel_xy, dtype=np.float32))
        )

    def add_keyframe(self, kf_frame_idx, kf_pose_world, 
                    points_2d, points_3d_cam, inherited_landmark_ids=None):
        """
        Register a keyframe and its features with the map.
        Args:
            kf_frame_idx: KITTI frame index
            kf_pose_world: (4, 4) world-from-camera pose
            points_2d: (N, 2) pixel coords in this keyframe
            points_3d_cam: (N, 3) 3D points in this keyframe's camera frame
            inherited_landmark_ids: optional list of length N. Entry i is:
                - landmark_id (>=0) if feature i matches an existing landmark
                - -1 if feature i is a new landmark
                - None (entire arg) means all features are new
        """
        n = len(points_2d)
        if inherited_landmark_ids is None:
            inherited_landmark_ids = [-1] * n

        # transform points to world frame for new-landmark initialization
        pts_h = np.concatenate([points_3d_cam, np.ones((n, 1), dtype=np.float32)], axis=1) # (N, 4)
        pts_world = (kf_pose_world @ pts_h.T).T[:, :3] 

        local_to_landmark = {}
        for i in range(n):
            lid = inherited_landmark_ids[i]
            if lid is None or lid < 0:
                # new landmark
                lid = self._create_landmark(pts_world[i])
            self.add_observation(lid, kf_frame_idx, points_2d[i])
            local_to_landmark[i] = lid

        self._kf_index_to_landmark[int(kf_frame_idx)] = local_to_landmark

    def get_landmark_ids_for_keyframe(self, kf_frame_idx):
        """
        Return a dict {local_index -> landmark_id} for the given keyframe.
        Used to look up which landmarks each feature in a keyframe corresponds to.
        """
        return self._kf_index_to_landmark.get(int(kf_frame_idx), {})

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def n_landmarks(self):
        return len(self._landmarks)
    
    def filter_by_observations(self, min_observations=3):
        """
        Return a new LandmarkMap with only landmarks observed in at least
        min_observations keyframes. Filters out weakly-constrained landmarks.
        """
        filtered = LandmarkMap()
        old_to_new_id = {}
        for lid, lm in self._landmarks.items():
            if len(lm['observations']) >= min_observations:
                new_id = filtered._create_landmark(lm['position_world'])
                old_to_new_id[lid] = new_id
                for kf, pixel in lm['observations']:
                    filtered.add_observation(new_id, kf, pixel)

        # also need to filter the per-keyframe local_idx -> landmark_id mappings
        for kf_idx, mapping in self._kf_index_to_landmark.items():
            new_mapping = {
                local: old_to_new_id[old_lid]
                for local, old_lid in mapping.items()
                if old_lid in old_to_new_id
            }
            if new_mapping:
                filtered._kf_index_to_landmark[kf_idx] = new_mapping
        return filtered

    def save(self, output_path):
            """Save the landmark map to disk as a compressed npz file."""
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # sorted list of all landmark IDs and their world positions in the same order
            ids = sorted(self._landmarks.keys())
            positions = np.stack([self._landmarks[lid]['position_world'] for lid in ids])  # (L, 3)

            # observations stored as 3 parallel flat arrays
            # (landmark_id, kf_frame_idx, pixel_xy) per row
            obs_landmark_id, obs_kf_idx, obs_pixels = [], [], []
            for lid in ids:
                for kf_idx, pixel in self._landmarks[lid]['observations']:
                    obs_landmark_id.append(lid)
                    obs_kf_idx.append(kf_idx)
                    obs_pixels.append(pixel)

            # per-keyframe local_idx -> landmark_id mappings, also flattened
            # one row = "in keyframe X, local feature Y is landmark Z"
            kf_idx_arr, local_idx_arr, landmark_id_arr = [], [], []
            for kf_idx, mapping in self._kf_index_to_landmark.items():
                for local_idx, lid in mapping.items():
                    kf_idx_arr.append(kf_idx)
                    local_idx_arr.append(local_idx)
                    landmark_id_arr.append(lid)

            npz_data = {
                'ids': np.array(ids, dtype=np.int32),
                'positions_world': positions.astype(np.float32),
                'obs_landmark_id': np.array(obs_landmark_id, dtype=np.int32),
                'obs_kf_idx': np.array(obs_kf_idx, dtype=np.int32),
                'obs_pixels': np.stack(obs_pixels).astype(np.float32) if obs_pixels else np.empty((0, 2), dtype=np.float32),
                'kf_idx_arr': np.array(kf_idx_arr, dtype=np.int32),
                'local_idx_arr': np.array(local_idx_arr, dtype=np.int32),
                'landmark_id_arr': np.array(landmark_id_arr, dtype=np.int32),
            }
            np.savez_compressed(str(output_path) + '.npz', **npz_data)

            meta = {
                'n_landmarks': len(ids),
                'n_observations': len(obs_landmark_id),
                'mean_obs_per_landmark': float(len(obs_landmark_id) / max(1, len(ids))),
                'max_obs_per_landmark': max((len(self._landmarks[lid]['observations']) for lid in ids), default=0),
            }
            with open(str(output_path) + '.json', 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"Saved {len(ids)} landmarks with {len(obs_landmark_id)} observations to {output_path}.npz (+ .json)")


def load_landmarks(npz_path):
    """Load landmarks saved by LandmarkMap.save. Returns a LandmarkMap instance."""
    path = Path(npz_path)
    if path.suffix != '.npz':
        path = path.with_suffix('.npz')

    data = np.load(str(path))

    lm_map = LandmarkMap()

    # restore landmarks with their original IDs (no remapping needed)
    for lid, pos in zip(data['ids'], data['positions_world']):
        lm_map._landmarks[int(lid)] = {
            'id': int(lid),
            'position_world': pos.astype(np.float32),
            'observations': [],
        }

    # advance _next_id past the highest existing ID so future creations don't collide
    lm_map._next_id = int(data['ids'].max()) + 1 if len(data['ids']) > 0 else 0

    # restore observations (each observation row appends to its landmark's list)
    for lid, kf_idx, pixel in zip(data['obs_landmark_id'], data['obs_kf_idx'], data['obs_pixels']):
        lm_map._landmarks[int(lid)]['observations'].append(
            (int(kf_idx), pixel.astype(np.float32))
        )

    # restore per-keyframe local_idx -> landmark_id mappings
    for kf_idx, local_idx, lid in zip(data['kf_idx_arr'], data['local_idx_arr'], data['landmark_id_arr']):
        kf_idx_int = int(kf_idx)
        if kf_idx_int not in lm_map._kf_index_to_landmark:
            lm_map._kf_index_to_landmark[kf_idx_int] = {}
        lm_map._kf_index_to_landmark[kf_idx_int][int(local_idx)] = int(lid)

    return lm_map