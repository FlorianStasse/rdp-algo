"""Test module"""
import numpy as np
import numpy.testing as npt
import pytest

from rdp_algo import rdp


@pytest.fixture(name="seesaw")
def fixture_seesaw() -> list:
    """Seesaw fixture"""
    return [[0, 0], [1, 1], [2, 0], [3, 1]]


def test_single_point():
    """Testing output for a single point"""
    npt.assert_array_equal(rdp([[0, 0]], epsilon=0.1), [[0, 0]])


def test_two_points():
    """Testing output for two points"""
    npt.assert_array_equal(rdp([[0, 0], [1, 1]], epsilon=0.1), [[0, 0], [1, 1]])


def test_2d():
    """Testing 2-D case"""
    points = np.random.random((10, 2))
    rdp(points, epsilon=0.1)


def test_3d():
    """Testing 3-D case (should check for all N-D case"""
    points = np.random.random((10, 3))
    rdp(points, epsilon=0.1)


def test_list(seesaw):
    """Testing return type if input is list"""
    assert isinstance(rdp(seesaw, epsilon=0.1), list)
    assert isinstance(rdp([[0, 0, 0], [1, 1, 1]], epsilon=0.1), list)


def test_array(seesaw):
    """Testing return type if input is numpy array"""
    assert isinstance(rdp(np.array(seesaw), epsilon=0.1), np.ndarray)
    assert isinstance(rdp(np.array([[0, 0, 0], [1, 1, 1]]), epsilon=0.1), np.ndarray)


def test_small_epsilon(seesaw: list):
    """Testing small epsilon (no simplification)"""
    npt.assert_array_equal(rdp(seesaw, epsilon=0.1), seesaw)


def test_big_epsilon(seesaw: list):
    """Testing big epsilon which simplifies everything"""
    npt.assert_array_equal(rdp(seesaw, epsilon=10), [[0, 0], [3, 1]])
