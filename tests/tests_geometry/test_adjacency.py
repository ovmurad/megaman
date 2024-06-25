# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
import pytest
from scipy.sparse import isspmatrix
from scipy.spatial.distance import cdist, pdist, squareform

from src.megaman.geometry import (compute_adjacency_matrix, Adjacency,
                                  adjacency_methods)

RANDOM_STATE = np.random.RandomState(42)
EXACT_METHODS = set(Adjacency.methods())
X = RANDOM_STATE.rand(100, 3)

GTRUE = {}

for n_neighbors in [5, 10, 15]:
    GTRUE[n_neighbors] = compute_adjacency_matrix(X, method='brute', n_neighbors=n_neighbors)
for radius in [0.1, 0.5, 1.0]:
    GTRUE[radius] = compute_adjacency_matrix(X, method='brute', radius=radius)

def test_adjacency_methods():
    np.testing.assert_equal(set(adjacency_methods()),
                 {'auto', 'ball_tree', 'brute', 'kd_tree'})


def test_adjacency_input_validation():
    x = RANDOM_STATE.rand(20, 3)
    # need to specify radius or n_neighbors
    np.testing.assert_raises(ValueError, compute_adjacency_matrix, x)
    # cannot specify both radius and n_neighbors
    np.testing.assert_raises(ValueError, compute_adjacency_matrix, x, radius=1, n_neighbors=10)


@pytest.mark.parametrize('n_neighbors', [5, 10, 15])
@pytest.mark.parametrize('method', Adjacency.methods())
def test_kneighbors(n_neighbors, method):

    G = compute_adjacency_matrix(X, method=method, n_neighbors=n_neighbors)
    assert isspmatrix(G)
    assert G.shape == (X.shape[0], X.shape[0])
    if method in EXACT_METHODS:
        np.testing.assert_allclose(G.toarray(), GTRUE[n_neighbors].toarray(), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('radius', [0.1, 0.5, 1.0])
@pytest.mark.parametrize('method', Adjacency.methods())
def test_radius(radius, method):

    G = compute_adjacency_matrix(X, method=method, radius=radius)
    assert isspmatrix(G)
    assert G.shape == (X.shape[0], X.shape[0])

    if method in EXACT_METHODS:
        np.testing.assert_allclose(G.toarray(), GTRUE[radius].toarray(), rtol=1e-5, atol=1e-7)


def test_unknown_method():
    x = np.arange(20).reshape((10, 2))
    np.testing.assert_raises(ValueError, compute_adjacency_matrix, x, 'foo')


@pytest.fixture
def adj_data():
    x = RANDOM_STATE.randn(10, 2)
    D_true = squareform(pdist(x))
    D_true[D_true > 0.5] = 0
    return x, D_true

@pytest.mark.parametrize('method', ['auto', 'brute'])
def test_all_methods_close(method, adj_data):

    x, D_true = adj_data

    this_D = compute_adjacency_matrix(x, method=method, radius=0.5)
    np.testing.assert_allclose(this_D.toarray(), D_true, rtol=1e-5, atol=1e-7)


def test_custom_adjacency():
    class CustomAdjacency(Adjacency):
        name = "custom"
        def adjacency_graph(self, x):
            return squareform(pdist(x))

    x = RANDOM_STATE.rand(10, 2)
    D = compute_adjacency_matrix(x, method='custom', radius=1)
    np.testing.assert_allclose(D, cdist(x, x))

    Adjacency._remove_from_registry("custom")

@pytest.fixture
def adj_data_2():
    x = RANDOM_STATE.randn(10, 2)
    D_true = squareform(pdist(x))
    D_true[D_true > 1.5] = 0
    return x, D_true
