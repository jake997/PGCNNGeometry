import pytest
from sage.all import CyclicPermutationGroup, DihedralGroup
from pgcnn import PGCNN


@pytest.fixture
def cyclic2():
    """Very small group for fast tests."""
    return CyclicPermutationGroup(2)


@pytest.fixture
def cyclic3():
    """Small nontrivial group."""
    return CyclicPermutationGroup(3)


@pytest.fixture
def dihedral4():
    """Non-abelian test group."""
    return DihedralGroup(4)


@pytest.fixture
def simple_network(cyclic2):
    """Minimal network."""
    return PGCNN(cyclic2, [1])


@pytest.fixture
def two_layer_network(cyclic2):
    """Two layer network."""
    return PGCNN(cyclic2, [2, 1])