"""General tests for embeddings"""

# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
import pytest

from src.megaman.embedding import (Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding)
from src.megaman.geometry.geometry import Geometry

EMBEDDINGS = [Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding]

# # TODO: make estimator_checks pass!
# def test_estimator_checks():
#     from sklearn.utils.estimator_checks import check_estimator
#     for Embedding in EMBEDDINGS:
#         yield check_estimator, Embedding


@pytest.mark.parametrize("Embedding", EMBEDDINGS)
@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_embeddings_fit_vs_transform(Embedding, n_components):
    rand = np.random.RandomState(42)
    X = rand.rand(100, 5)
    geom = Geometry(adjacency_kwds={'radius': 1.0}, affinity_kwds={'radius': 1.0})

    model = Embedding(n_components=n_components, geom=geom, random_state=rand)
    embedding = model.fit_transform(X)
    assert model.embedding_.shape == (X.shape[0], n_components)
    np.testing.assert_allclose(embedding, model.embedding_)

@pytest.mark.parametrize("Embedding", EMBEDDINGS)
def test_embeddings_bad_arguments(Embedding):
    rand = np.random.RandomState(32)
    X = rand.rand(100, 3)

    # no radius set
    embedding = Embedding()
    with pytest.raises(ValueError):
        embedding.fit(X)

    # unrecognized geometry
    embedding = Embedding(radius=2, geom='blah')
    with pytest.raises(ValueError):
        embedding.fit(X)
