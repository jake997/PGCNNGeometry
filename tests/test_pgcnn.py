import pytest
from pgcnn import PGCNN


def test_pgcnn_initialization(cyclic2):
    net = PGCNN(cyclic2, [2, 2, 1])
    assert isinstance(net, PGCNN)
    assert net.group == cyclic2
    assert net.L == 3
    assert net.n == 2
    assert net.r == [2, 2, 1]

def test_pgcnn_image_dimension_cyclic2(cyclic2):
    net = PGCNN(cyclic2, [2, 2, 1])
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 4

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 4

def test_pgcnn_image_dimension_cyclic3(cyclic3):
    net = PGCNN(cyclic3, [2, 2, 1])
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 7

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 7

@pytest.mark.slow
def test_pgcnn_image_dimension_dihedral4(dihedral4):
    net = PGCNN(dihedral4, [2, 2, 1])
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 22

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 22


def test_pgcnn_image_dimension_cyclic2_finite_ring(cyclic2,):
    net = PGCNN(cyclic2, [2, 2, 1], finite_ring=True)
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 4

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 4

def test_pgcnn_image_dimension_cyclic3_finite_ring(cyclic3):
    net = PGCNN(cyclic3, [2, 2, 1], finite_ring=True)
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 7

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 7

@pytest.mark.slow
def test_pgcnn_image_dimension_dihedral4_finite_ring(dihedral4):
    net = PGCNN(dihedral4, [2, 2, 1], finite_ring=True)
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 22

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 22