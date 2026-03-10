import pytest
from pgcnn import PGCNN



def test1_pgcnn_fiber_size(cyclic2):
    net = PGCNN(cyclic2, [2, 2, 1])
    size = net.compute_size_fiber_Phi_m2()
    assert isinstance(size, int)
    assert size == 4

    size = net.compute_size_fiber_varphi_m2()
    assert isinstance(size, int)
    assert size == 4

@pytest.mark.slow
def test2_pgcnn_fiber_size(cyclic3, dihedral4):
    net = PGCNN(cyclic3, [2, 2, 1])
    size = net.compute_size_fiber_Phi_m2()
    assert isinstance(size, int)
    assert size == 9

    size = net.compute_size_fiber_varphi_m2()
    assert isinstance(size, int)
    assert size == 9

    net = PGCNN(dihedral4, [2, 2, 1])
    size = net.compute_size_fiber_Phi_m2()
    assert isinstance(size, int)
    assert size == 64

    size = net.compute_size_fiber_varphi_m2()
    assert isinstance(size, int)
    assert size == 64   