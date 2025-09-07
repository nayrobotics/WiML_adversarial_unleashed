import torch

import numpy as np


_TOL = np.finfo(float).eps * 4.0

def slerp_quat(quat_a, quat_b, t, extra_spin=0, use_shortest=True):
    """Batch spherical linear interpolation (SLERP) between two quaternions."""

    result = torch.zeros_like(quat_a)
    mask_start = torch.isclose(t, torch.zeros_like(t)).squeeze()
    mask_end = torch.isclose(t, torch.ones_like(t)).squeeze()
    result[mask_start] = quat_a[mask_start]
    result[mask_end] = quat_b[mask_end]

    dot_val = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    mask_close = (torch.abs(torch.abs(dot_val) - 1.0) < _TOL).squeeze()
    result[mask_close] = quat_a[mask_close]

    if use_shortest:
        dot_copy = dot_val.clone()
        dot_val = torch.where(dot_copy < 0, -dot_val, dot_val)
        quat_b = torch.where(dot_copy < 0, -quat_b, quat_b)

    theta = torch.acos(dot_val) + extra_spin * torch.pi
    mask_small = (torch.abs(theta) < _TOL).squeeze()
    result[mask_small] = quat_a[mask_small]

    mask_interp = ~(mask_start | mask_end | mask_close | mask_small)

    inv_theta = 1.0 / theta
    qa_term = torch.sin((1.0 - t) * theta) * inv_theta
    qb_term = torch.sin(t * theta) * inv_theta

    interp = quat_a * qa_term + quat_b * qb_term
    result[mask_interp] = interp[mask_interp]

    return result
