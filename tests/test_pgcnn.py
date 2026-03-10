import pytest


def test_pgcnn_initialization(simple_network):
    net = simple_network

    assert net.L == 1
    assert net.n == 2
    assert len(net.x_vars) == 2
    assert len(net.params) == 1


def test_parameter_count(two_layer_network):
    net = two_layer_network

    expected = net.L * net.n
    assert len(net.all_params) == expected


def test_generic_point_structure(simple_network):
    point, filters = simple_network._generic_point()

    assert len(point) == simple_network.L * simple_network.n
    assert len(filters) == simple_network.L


def test_expected_fiber(simple_network):
    _, filters = simple_network._generic_point()
    fiber = simple_network._expected_generic_fiber(filters)

    assert isinstance(fiber, list)
    assert len(fiber) > 0


def test_phi_coefficients(simple_network):
    coeffs = simple_network.Phi_coeffs

    assert isinstance(coeffs, list)
    assert len(coeffs) > 0


def test_sage_image_dimension(simple_network):
    dim = simple_network.compute_dim_image_Phi_sage()

    assert isinstance(dim, int)
    assert dim >= 0