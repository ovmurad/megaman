# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from sklearn import neighbors

from .utils import RegisterSubclasses


def compute_adjacency_matrix(X, method='auto', **kwargs):
    """Compute an adjacency matrix with the given method"""
    if method == 'auto':
        method = 'kd_tree'
    return Adjacency.init(method, **kwargs).adjacency_graph(X.astype('float'))


def adjacency_methods():
    """Return the list of valid adjacency methods"""
    return ['auto'] + list(Adjacency.methods())


class Adjacency(RegisterSubclasses):
    """Base class for computing adjacency matrices"""
    def __init__(self, radius=None, n_neighbors=None, mode='distance'):
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.mode = mode

        if (radius is None) == (n_neighbors is None):
           raise ValueError("Must specify either radius or n_neighbors, "
                            "but not both.")

    def adjacency_graph(self, X):
        if self.n_neighbors is not None:
            return self.knn_adjacency(X)
        elif self.radius is not None:
            return self.radius_adjacency(X)

    def knn_adjacency(self, X):
        raise NotImplementedError()

    def radius_adjacency(self, X):
        raise NotImplementedError()


class BruteForceAdjacency(Adjacency):
    name = 'brute'

    def radius_adjacency(self, X):
        model = neighbors.NearestNeighbors(algorithm=self.name).fit(X)
        # pass X so that diagonal will have explicit zeros
        return model.radius_neighbors_graph(X, radius=self.radius,
                                            mode=self.mode)

    def knn_adjacency(self, X):
        model = neighbors.NearestNeighbors(algorithm=self.name).fit(X)
        # pass X so that diagonal will have explicit zeros
        return model.kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                      mode=self.mode)


class KDTreeAdjacency(BruteForceAdjacency):
    name = 'kd_tree'


class BallTreeAdjacency(BruteForceAdjacency):
    name = 'ball_tree'
