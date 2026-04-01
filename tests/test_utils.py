import pytest
from sage.all import vector, QQ, matrix
from pgcnn.utils import (
    group_matrix,
    group_action_left,
    group_action_right,
    tensor_power
)
#from tests.conftest import cyclic2


def test_group_matrix_shape(cyclic3):
    """Group matrix should have size |G| x |G|."""
    G = list(cyclic3)
    v = vector(QQ, [1, 2, 3])

    M = group_matrix(G, v)

    assert M.nrows() == len(G)
    assert M.ncols() == len(G)


def test_group_action_left_identity(cyclic3):
    """Left action by identity should not change vector."""
    G = list(cyclic3)
    v = vector(QQ, [1, 2, 3])
    g = cyclic3.one()

    result = group_action_left(G, v, g)

    assert result == v


def test_group_action_right_identity(cyclic3):
    """Right action by identity should not change vector."""
    G = list(cyclic3)
    v = vector(QQ, [1, 2, 3])
    g = cyclic3.one()

    result = group_action_right(G, v, g)

    assert result == v


def test_tensor_power_basic():
    """Tensor power should produce correct matrix size."""

    M = matrix(QQ, [[1, 2], [3, 4]])
    T = tensor_power(M, 2)
    assert T.nrows() == 4
    assert T.ncols() == 4
    assert T == matrix(QQ, [
        [1, 2, 2, 4],
        [3, 4, 6, 8],
        [3, 6, 4, 8],
        [9, 12, 12, 16]
    ])