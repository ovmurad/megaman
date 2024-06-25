# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import pytest

import os

import numpy as np

from scipy import io
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist, squareform

from src.megaman.geometry import (compute_adjacency_matrix,
                                  compute_affinity_matrix, Affinity,
                                  affinity_methods)

random_state = np.random.RandomState(36)
n_sample = 10
d = 2
X = random_state.randn(n_sample, d)
D = squareform(pdist(X))
D[D > 1/d] = 0


TEST_DATA = os.path.join(os.path.dirname(__file__),
                        'testmegaman_laplacian_rad0_2_lam1_5_n200.mat')



def test_affinity_methods():
    np.testing.assert_equal(set(affinity_methods()), {'auto', 'gaussian'})


def test_affinity_input_validation():
    X = np.random.rand(20, 3)
    D = compute_adjacency_matrix(X, radius=1)
    np.testing.assert_raises(ValueError, compute_affinity_matrix, X)


def test_affinity_sparse_vs_dense():
    """
    Test that A_sparse is the same as A_dense for a small A matrix
    """
    rad = 2.
    n_samples = 6
    X = np.arange(n_samples)
    X = X[ :,np.newaxis]
    X = np.concatenate((X,np.zeros((n_samples,1),dtype=float)),axis=1)
    X = np.asarray( X, order="C" )
    test_dist_matrix = compute_adjacency_matrix( X, method = 'auto', radius = rad )
    A_dense = compute_affinity_matrix(test_dist_matrix.toarray(), method = 'auto',
                                      radius = rad, symmetrize = False )
    A_sparse = compute_affinity_matrix(csr_matrix(test_dist_matrix),
                                       method = 'auto', radius = rad, symmetrize = False)
    A_spdense = A_sparse.toarray()
    A_spdense[ A_spdense == 0 ] = 1.
    np.testing.assert_allclose(A_dense, A_spdense)


def test_affinity_vs_matlab():
    """Test that the affinity calculation matches the matlab result"""
    matlab = io.loadmat(TEST_DATA)

    D = np.sqrt(matlab['S'])  # matlab outputs squared distances
    A_matlab = matlab['A']
    radius = matlab['rad'][0]

    # check dense affinity computation
    A_dense = compute_affinity_matrix(D, radius=radius)
    np.testing.assert_allclose(A_dense, A_matlab)

    # check sparse affinity computation
    A_sparse = compute_affinity_matrix(csr_matrix(D), radius=radius)
    np.testing.assert_allclose(A_sparse.toarray(), A_matlab)


@pytest.fixture
def setup_data():
    rand = np.random.RandomState(42)
    x = rand.rand(20, 3)
    d = cdist(x, x)
    return x, d


@pytest.mark.parametrize('adjacency_radius', [0.5, 1.0, 5.0])
@pytest.mark.parametrize('affinity_radius', [0.1, 0.5, 1.0])
@pytest.mark.parametrize('symmetrize', [True, False])
def test_affinity(adjacency_radius, affinity_radius, symmetrize, setup_data):
    x, d = setup_data

    adj = compute_adjacency_matrix(x, radius=adjacency_radius)
    aff = compute_affinity_matrix(adj, radius=affinity_radius, symmetrize=symmetrize)

    a = np.exp(-(d / affinity_radius) ** 2)
    a[d > adjacency_radius] = 0
    np.testing.assert_allclose(aff.toarray(), a)


def test_custom_affinity():
    class CustomAffinity(Affinity):
        name = "custom"
        def affinity_matrix(self, adjacency_matrix):
            return np.exp(-abs(adjacency_matrix.toarray()))

    rand = np.random.RandomState(42)
    X = rand.rand(10, 2)
    D = compute_adjacency_matrix(X, radius=10)
    A = compute_affinity_matrix(D, method='custom', radius=1)
    np.testing.assert_allclose(A, np.exp(-abs(D.toarray())))

    Affinity._remove_from_registry("custom")
