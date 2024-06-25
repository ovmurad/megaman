# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import os

import numpy as np

import pytest

from scipy import io
from scipy.sparse import isspmatrix, csr_matrix

from src.megaman.geometry import (compute_adjacency_matrix,
                                  compute_affinity_matrix,
                                  Laplacian, compute_laplacian_matrix,
                                  laplacian_methods)

RANDOM_STATE = np.random.RandomState(42)

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__),
                        'testmegaman_laplacian_rad0_2_lam1_5_n200.mat')
MATLAB_DATA = io.loadmat(TEST_DATA_PATH)
LAPLACIANS = {'unnormalized': MATLAB_DATA['Lunnorm'],
              'symmetricnormalized': MATLAB_DATA['Lsymnorm'],
              'geometric': MATLAB_DATA['Lgeom'],
              'randomwalk': MATLAB_DATA['Lrw'],
              'renormalized': MATLAB_DATA['Lreno1_5']}
RADIUS = MATLAB_DATA['rad'][0]
ADJACENCY = np.sqrt(MATLAB_DATA['S'])

def test_laplacian_methods():
    np.testing.assert_equal(set(laplacian_methods()),
                 {'auto', 'renormalized', 'symmetricnormalized',
                  'geometric', 'randomwalk', 'unnormalized'})


@pytest.mark.parametrize('input_type', [np.array, csr_matrix])
@pytest.mark.parametrize('laplacian_method', ['unnormalized', 'symmetricnormalized', 'geometric', 'randomwalk', 'renormalized'])
def test_laplacian_vs_matlab(input_type, laplacian_method):

    kwargs = {'scaling_epps': RADIUS}
    if laplacian_method == 'renormalized':
        kwargs['renormalization_exponent'] = 1.5

    adjacency = input_type(ADJACENCY)
    affinity = compute_affinity_matrix(adjacency, radius=RADIUS)
    laplacian = compute_laplacian_matrix(affinity, method=laplacian_method, **kwargs)

    if input_type is csr_matrix:
        laplacian = laplacian.toarray()

    np.testing.assert_allclose(laplacian, LAPLACIANS[laplacian_method], rtol=1e-6, atol=1e-7)


@pytest.fixture
def setup_matrices():

    x = RANDOM_STATE.rand(20, 2)
    adj = compute_adjacency_matrix(x, radius=0.5)
    aff = compute_affinity_matrix(adj, radius=0.1)

    return x, aff


@pytest.mark.parametrize('method', Laplacian.asymmetric_methods())
def test_laplacian_smoketest(method, setup_matrices):

    x, aff = setup_matrices

    lap = compute_laplacian_matrix(aff, method=method)

    assert isspmatrix(lap)
    np.testing.assert_equal(lap.shape, (x.shape[0], x.shape[0]))


def test_laplacian_unknown_method():
    """Test that laplacian fails with an unknown method type"""
    A = np.array([[ 5, 2, 1 ], [ 2, 3, 2 ],[1,2,5]])
    np.testing.assert_raises(ValueError, compute_laplacian_matrix, A, method='<unknown>')


@pytest.mark.parametrize('method', Laplacian.asymmetric_methods())
@pytest.mark.parametrize('adjacency_radius', [0.5, 1.0])
@pytest.mark.parametrize('affinity_radius', [0.1, 0.3])
def test_laplacian_full_output(method, adjacency_radius, affinity_radius):

    x = RANDOM_STATE.rand(20, 2)

    adj = compute_adjacency_matrix(x, radius=adjacency_radius)
    aff = compute_affinity_matrix(adj, radius=affinity_radius)
    lap, lapsym, w = compute_laplacian_matrix(aff, method=method, full_output=True)

    sym = w[:, np.newaxis] * (lap.toarray() + np.eye(*lap.shape))

    np.testing.assert_allclose(lapsym.toarray(), sym, rtol=1e-6, atol=1e-7)
