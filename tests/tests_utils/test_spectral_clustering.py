
import numpy as np
import pytest

from src.megaman.utils.eigendecomp import EIGEN_SOLVERS
from src.megaman.utils.spectral_clustering import SpectralClustering


@pytest.mark.parametrize("stabalize", [True, False])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("eigen_solver", EIGEN_SOLVERS)  # Assuming EIGEN_SOLVERS is a list of solvers
def test_spectral_clustering(stabalize, renormalize, eigen_solver):
    K = 3
    num_per_cluster = 100
    c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    X = np.repeat(c, np.repeat(num_per_cluster, K), axis=0)
    radius = 5
    rng = np.random.RandomState(36)

    if eigen_solver in ['dense', 'auto']:
        solver_kwds = {}
    else:
        solver_kwds = {'maxiter': 100000, 'tol': 1e-5}
    SC = SpectralClustering(K=K, radius=radius, stabalize=stabalize, renormalize=renormalize,
                            eigen_solver=eigen_solver, solver_kwds=solver_kwds, random_state=rng,
                            additional_vectors=0)
    labels = SC.fit_transform(X, input_type='data')
    for k in range(K):
        cluster_labs = labels[range((k * num_per_cluster), ((k + 1) * num_per_cluster))]
        first_lab = cluster_labs[0]
        assert (np.all(cluster_labs == first_lab))