from functools import wraps

import numpy as np

from src.megaman.relaxation.precomputed import compute_Lk
from utils import generate_toy_laplacian


class BaseTestLkNeighbors(object):

    @classmethod
    def generate_laplacian(cls):
        raise NotImplementedError()

    @classmethod
    def generate_subset(cls, n):
        raise NotImplementedError()

    @classmethod
    def setup_message(cls):
        raise NotImplementedError()

    @classmethod
    def setup_class(cls):
        cls.generate_laplacian_and_subset()
        cls.setup_message()
        cls.Lk_tensor, cls.nbk, cls.si_map = compute_Lk(cls.laplacian, cls.n, cls.subset)

    @classmethod
    def generate_laplacian_and_subset(cls):
        cls.laplacian = cls.generate_laplacian()
        cls.n = cls.laplacian.shape[0]
        cls.subset = cls.generate_subset(cls.n)

    @classmethod
    def get_rk(cls, k):
        idx_lk_space = cls.si_map[k]
        return (cls.nbk[idx_lk_space] == k).nonzero()[0]

    @staticmethod
    def _test_array_close_deco(func):
        @wraps(func)
        def wrapper(cls):
            test_func, true_func, err_msg = func(cls)

            def properties_to_test():
                for k in cls.subset:
                    Lk = cls.Lk_tensor[cls.si_map[k]]
                    yield test_func(Lk, k)
            def properties_is_true():
                for k in cls.subset:
                    yield true_func(cls.laplacian, k)

            testing_list = np.concatenate([np.asarray(item).reshape(-1) for item in properties_to_test()])
            correct_list = np.concatenate([np.asarray(item).reshape(-1) for item in properties_is_true()])

            try:
                np.testing.assert_allclose(testing_list, correct_list, err_msg=err_msg)
            except TypeError:
                for idx, totest in enumerate(testing_list):
                    np.testing.assert_allclose(
                        totest, correct_list[idx],
                        err_msg=err_msg + ", fails at k == {}".format(idx)
                    )
        return wrapper

    @_test_array_close_deco
    def test_nonzero_counts(cls):
        def test_func(Lk,k):
            return Lk.nonzero()[0].shape[0]
        def true_func(laplacian,k):
            nonzero_count = laplacian[k,:].nonzero()[0].shape[0]
            return 3*nonzero_count-2 if nonzero_count != 0 else 0
        err_msg = 'The nonzero count should be the same for Lk and laplacian[k,:]'
        return test_func, true_func, err_msg

    @_test_array_close_deco
    def test_diagonal_Lk(cls):
        def test_func(Lk,k):
            return Lk.diagonal()
        def true_func(laplacian,k):
            nnz_axis = laplacian[k,:].nonzero()
            rk = cls.get_rk(k)
            true_Lk = np.squeeze(np.asarray(laplacian[k,:][nnz_axis]))
            true_Lk[rk] *= -1
            return true_Lk
        err_msg = 'The diagonal of Lk should be the same as nonzeros laplacian[k,:] term'
        return test_func, true_func, err_msg

    @_test_array_close_deco
    def test_row_Lk(cls):
        def test_func(Lk,k):
            rk = cls.get_rk(k)
            return np.squeeze(Lk[rk,:].toarray())
        def true_func(laplacian,k):
            nnz_axis = laplacian[k,:].nonzero()
            return np.squeeze(np.array(-laplacian[k,:][nnz_axis]))
        err_msg = 'The kth row of Lk should be the same as laplacian[k,:] term.'
        return test_func, true_func, err_msg

    @_test_array_close_deco
    def test_col_Lk(cls):
        def test_func(Lk,k):
            rk = cls.get_rk(k)
            return np.squeeze(Lk[:,rk].toarray())
        def true_func(laplacian,k):
            nnz_axis = laplacian[k,:].nonzero()
            return np.squeeze(np.array(-laplacian[k,:][nnz_axis]))
        err_msg = 'The kth col of Lk should be the same as laplacian[k,:] term.'
        return test_func, true_func, err_msg

    @classmethod
    def test_Lk_symmetric(cls):
        if_symmetric = [np.allclose(Lk.toarray(), Lk.T.toarray()) for Lk in cls.Lk_tensor]
        assert np.all(if_symmetric), 'One or more Lk is not symmetric'

    @classmethod
    def test_neighbors(cls):
        true_neighbors = [cls.laplacian[k, :].nonzero()[1] for k in cls.subset]
        for idx, nb in enumerate(cls.nbk):
            np.testing.assert_array_equal(
                nb, true_neighbors[idx],
                err_msg='calculated nbk should be the same as the non zero term in laplacian'
            )

    @classmethod
    def test_si_map_index(cls):
        keys, values = zip(*cls.si_map.items())
        sorted_key = np.sort(keys)
        sorted_subset = np.sort(cls.subset)
        np.testing.assert_array_equal(
            sorted_key, sorted_subset,
            err_msg='The index in subset should be identical to that in si_map'
        )

    @classmethod
    def test_Lk_nbk_size_match(cls):
        if_size_match = [Lk.shape[0] == cls.nbk[k].shape[0] for k, Lk in enumerate(cls.Lk_tensor)]
        assert np.all(if_size_match), 'One or more size of Lk and nbk does not match'


class TestLkWithWholeSubset(BaseTestLkNeighbors):

    @classmethod
    def generate_laplacian(cls):
        return generate_toy_laplacian()

    @classmethod
    def generate_subset(cls, n):
        return np.arange(n)

    @classmethod
    def setup_message(cls):
        print('Testing Lk properties for whole subset')


class TestLkWithHalfIncrementalSubset(TestLkWithWholeSubset):

    @classmethod
    def generate_subset(cls, n):
        return np.arange(0, n, 2)

    @classmethod
    def setup_message(cls):
        print('Testing Lk properties for half incremental subsets')


class TestLkWithQuarterRandomSubset(TestLkWithWholeSubset):

    @classmethod
    def generate_subset(cls, n):
        size = int(round(n / 4))
        return np.random.choice(np.arange(n), size, replace=False)

    @classmethod
    def setup_message(cls):
        print('Testing Lk properties with a quarter random generated subset')
