import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform, pdist

from src.megaman.utils.estimate_radius import run_estimate_radius

RANDOM_STATE = np.random.RandomState(42)
def test_radius_serial_vs_parallel():

    X = RANDOM_STATE.randn(100, 10)
    dists = csr_matrix(squareform(pdist(X)))
    sample = range(100)
    d = 3
    rmin = 2
    rmax = 10.0
    ntry = 10
    run_parallel = True
    results_parallel = run_estimate_radius(X, dists, sample, d, rmin, rmax, ntry, run_parallel)
    print(results_parallel)
    results_serial = run_estimate_radius(X, dists, sample, d, rmin, rmax, ntry, False)
    print(results_serial)
    np.testing.assert_array_almost_equal(results_parallel, results_serial)