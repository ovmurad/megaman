# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

import time

import numpy as np

from ..geometry import RiemannMetric


def compute_tangent_plane(embedding, geom):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    L = geom.laplacian_matrix
    rmetric = RiemannMetric(embedding, L)
    rmetric.get_dual_rmetric()
    HH = rmetric.H
    evalues, evects = map(np.array, zip(*[eigsorted(HHi) for HHi in HH]))
    return evects
