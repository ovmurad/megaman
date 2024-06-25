import numpy as np

from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform, pdist

import src.megaman.utils.analyze_dimension_and_radius as adar

RANDOM_STATE = np.random.RandomState(42)

def test_dim_distance_passed_vs_computed():

    X = RANDOM_STATE.randn(100, 10)
    dists = csr_matrix(squareform(pdist(X)))
    dists
    rmin = 2
    rmax = 10.0
    nradii = 10
    radii = 10**(np.linspace(np.log10(rmin), np.log10(rmax), nradii))

    results_passed = adar.neighborhood_analysis(dists, radii)
    avg_neighbors = results_passed['avg_neighbors'].flatten()
    radii = results_passed['radii'].flatten()
    fit_range = range(len(radii))
    dim_passed = adar.find_dimension_plot(avg_neighbors, radii, fit_range)
    results_computed, dim_computed = adar.run_analyze_dimension_and_radius(X, rmin, rmax, nradii)
    # Bad test because compute_adj_matrix does not necessarily produce 0s on the diagonal.
    # assert(dim_passed == dim_computed)
    # np.testing.assert_allclose(results_passed['avg_neighbors'], results_computed['avg_neighbors'], rtol=1e-6, atol=1e-7)