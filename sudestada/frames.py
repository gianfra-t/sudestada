"""Body and wind frames: FLU <-> FRD and the flow angles of the freestream."""

from __future__ import annotations

import numpy as np

from sudestada.fits.splines import smooth_norm

FRD_FROM_FLU = np.diag([1.0, -1.0, -1.0])
SPEED_EPSILON_MPS = 1e-12


def body_from_wind(alpha, beta):
    """``C_BW`` mapping ``[-CD, CY, -CL]`` in wind axes to body FRD axes."""

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array(
        [
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0.0],
            [sa * cb, -sa * sb, ca],
        ]
    )


def flow_angles(velocity_flu):
    """``(speed, alpha, beta)`` of the freestream with airspeed ``velocity_flu`` (body FLU; rad, FRD convention)."""

    velocity_flu = np.asarray(velocity_flu, dtype=float)
    speed = smooth_norm(velocity_flu, SPEED_EPSILON_MPS)
    u, v, w = FRD_FROM_FLU @ velocity_flu
    alpha = np.arctan2(w, u)
    beta = np.arctan2(v, np.sqrt(u * u + w * w + SPEED_EPSILON_MPS**2))
    return speed, alpha, beta
