"""The ring's images grouped by trigger, from their header stamps.

The cameras are triggered together (slam_sensors FrameSync), and each stamps
the trigger edge with its own clock read. On the boards so far the stamps of
one trigger were equal to ~1 us; in stretches of 2026-10-19's bags they are
1-7 ms apart, with no camera nearer the others than the rest. So a trigger's
images are each camera's nearest to camera 0's within TOL, and its stamp is
the median of theirs: what open_vins' callback_ring stamps the frame with
(ov_msckf/src/ros/ROS1Visualizer.cpp), so that a pose (trigger stamp + td)
leads back to its images.
"""
import numpy as np

TOL = 0.010     # s: under half the frame interval (33 ms at 30 Hz), so a trigger never takes its neighbour's image


def group(stamps, tol=TOL):
    """stamps: per camera, its images' header stamps in ns (ints).
    -> (t, members): t the triggers' stamps in s (float64, ascending),
    members[i, c] camera c's stamp (ns) of trigger i. Only triggers that every
    camera has an image of."""
    ref = np.sort(np.asarray(stamps[0], np.int64))
    cols = [ref]
    for s in stamps[1:]:
        s = np.sort(np.asarray(s, np.int64))
        if not len(s):
            return np.zeros(0), np.zeros((0, len(stamps)), np.int64)
        j = np.searchsorted(s, ref)
        lo, hi = s[np.maximum(j - 1, 0)], s[np.minimum(j, len(s) - 1)]
        cols.append(np.where(np.abs(hi - ref) < np.abs(lo - ref), hi, lo))
    m = np.stack(cols, 1)
    m = m[np.all(np.abs(m - ref[:, None]) <= round(tol * 1e9), axis=1)]
    return np.median(m * 1e-9, axis=1), m
