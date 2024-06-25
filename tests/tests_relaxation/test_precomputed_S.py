import pytest

import numpy as np

from src.megaman.relaxation.precomputed import makeA
from utils import generate_toy_laplacian


RANDOM_STATE = np.random.RandomState(42)


class BaseTestARkNeighbors(object):

    @classmethod
    def generate_laplacian(cls):
        raise NotImplementedError()

    @classmethod
    def setup_message(cls):
        raise NotImplementedError()

    @classmethod
    def setup_class(cls):

        cls.generate_laplacian_and_range()
        cls.setup_message()
        cls.A, cls.pairs = makeA(cls.laplacian)

        # HACK: A is somehow sorted by column, so here I'll change it manually.
        sortbyrow = np.lexsort((cls.pairs[:,1],cls.pairs[:,0]))
        cls.A = cls.A[sortbyrow]
        cls.pairs = cls.pairs[sortbyrow]

        # cls.Rk_tensor, cls.nbk = compute_Rk(cls.laplacian,cls.A,cls.n)
        cls.correct_S, cls.correct_pairs = cls.project_S_from_laplacian()

    @classmethod
    def generate_laplacian_and_range(cls):
        cls.laplacian = cls.generate_laplacian()
        cls.n = cls.laplacian.shape[0]
        cls.range = np.arange(cls.n)
        cls.Y = cls.generate_toy_Y()

    @classmethod
    def generate_toy_Y(cls):
        return np.random.uniform(size=cls.n)

    @classmethod
    def ij_is_neighbors(cls,i,j):
        return cls.laplacian[i,j] != 0

    @classmethod
    def project_S_from_laplacian(cls):
        # TODO: make the test process faster!
        S = [ cls.Y[i]-cls.Y[j] for i in np.arange(cls.n) \
              for j in np.arange(i+1,cls.n) \
              if cls.ij_is_neighbors(i,j) ]
        pairs = [ [i,j] for i in np.arange(cls.n) \
                  for j in np.arange(i+1,cls.n) \
                  if cls.ij_is_neighbors(i,j) ]
        return np.array(S), np.array(pairs)

    @classmethod
    def test_A_length_equality(cls):
        A_length = cls.A.shape[0]
        correct_A_length = cls.correct_S.shape[0]
        assert A_length == correct_A_length, 'The first dimension of A is calculated wrong.'

    @classmethod
    def test_pairs(cls):
        np.testing.assert_array_equal(
            cls.pairs, cls.correct_pairs,
            err_msg='Sorted pairs should be the same.'
        )

    @classmethod
    def test_A(cls):
        testing_S = cls.A.dot(cls.Y)
        np.testing.assert_allclose(
            testing_S, cls.correct_S,
            err_msg='A*y should be the same as yj-yi for all j>i'
        )

    @classmethod
    def _test_ATAinv(cls):
        # TODO: why this test will running out of the memory?
        ATAinv = np.linalg.pinv(cls.A.T.dot(cls.A).todense())
        S = cls.A.dot(cls.Y)
        testing_Y = ATAinv.dot(cls.A.T).dot(S)
        np.testing.assert_allclose(
            testing_Y, cls.Y,
            err_msg='ATAinv * AT * S should be the same as original Y'
        )

    @classmethod
    def _test_Rk(cls):
        # TODO: Need to understand what Rk means.
        pass

class TestAkRkNbkFromToyLaplacian(BaseTestARkNeighbors):
    @classmethod
    def generate_laplacian(cls):
        return generate_toy_laplacian(n=200)
    @classmethod
    def setup_message(cls):
        print ('Tesking Rk properties for toy laplacian.')
