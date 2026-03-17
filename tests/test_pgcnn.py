import pytest
from pgcnn import PGCNN


def test_pgcnn_initialization(cyclic2):
    net = PGCNN(cyclic2, [2, 2, 1])
    assert isinstance(net, PGCNN)
    assert net.group == cyclic2
    assert net.L == 3
    assert net.n == 2
    assert net.r == [2, 2, 1]

def test_pgcnn_image_dimension(cyclic2, cyclic3, dihedral4):
    net = PGCNN(cyclic2, [2, 2, 1])
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 4

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 4

    net = PGCNN(cyclic3, [2, 2, 1])
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 7

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 7

    net = PGCNN(dihedral4, [2, 2, 1])
    dim = net.compute_dim_image_Phi_sage()
    assert isinstance(dim, int)
    assert dim == 22

    dim = net.compute_dim_image_varphi_sage()
    assert isinstance(dim, int)
    assert dim == 22

