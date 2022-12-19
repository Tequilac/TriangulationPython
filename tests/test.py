from unittest import TestCase

import numpy as np
import pytest
import cv2

from .fixtures import setup_horizontal_configuration, setup_second_camera_rotated_right_configuration, \
    setup_general_camera_configuration, setup_second_camera_rotated_left_configuration
from iterative_eigen.iterative_eigen import IterativeEigen
from iterative_ls.iterative_ls import IterativeLS
from linear_eigen.linear_eigen import LinearEigen
from linear_ls.linear_ls import LinearLS
from poly.poly import Poly
from poly_abs.poly_abs import PolyAbs


def evaluate_result(result: tuple[float, float, float], expected_result: tuple[float, float, float], max_percentage_error=0.001):
    tolerance = cv2.norm(expected_result) * max_percentage_error
    distance = cv2.norm(np.subtract(result, expected_result))
    assert pytest.approx(0, abs=tolerance) == distance


class Test(TestCase):
    def test_poly_general_setup(self):
        P0, P1 = setup_general_camera_configuration()
        p = Poly(P0, P1)
        result = p.triangulate_point((146, 642.288), (1137.31, 385.201))
        expected_result = (0.0, 100.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_rotated_left(self):
        P0, P1 = setup_second_camera_rotated_left_configuration()
        p = Poly(P0, P1)
        result = p.triangulate_point((878.821, 634.619), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_rotated_right(self):
        P0, P1 = setup_second_camera_rotated_right_configuration()
        p = Poly(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (150.068, 634.618))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = Poly(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_abs_general_setup(self):
        P0, P1 = setup_general_camera_configuration()
        p = PolyAbs(P0, P1)
        result = p.triangulate_point((146, 642.288), (1137.31, 385.201))
        expected_result = (0.0, 100.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_abs_rotated_left(self):
        P0, P1 = setup_second_camera_rotated_left_configuration()
        p = PolyAbs(P0, P1)
        result = p.triangulate_point((878.821, 634.619), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_abs_rotated_right(self):
        P0, P1 = setup_second_camera_rotated_right_configuration()
        p = PolyAbs(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (150.068, 634.618))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_poly_abs_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = PolyAbs(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_ls_general_setup(self):
        P0, P1 = setup_general_camera_configuration()
        p = LinearLS(P0, P1)
        result = p.triangulate_point((146, 642.288), (1137.31, 385.201))
        expected_result = (0.0, 100.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_ls_rotated_left(self):
        P0, P1 = setup_second_camera_rotated_left_configuration()
        p = LinearLS(P0, P1)
        result = p.triangulate_point((878.821, 634.619), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_ls_rotated_right(self):
        P0, P1 = setup_second_camera_rotated_right_configuration()
        p = LinearLS(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (150.068, 634.618))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_ls_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = LinearLS(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_ls_general_setup(self):
        P0, P1 = setup_general_camera_configuration()
        p = IterativeLS(P0, P1)
        result = p.triangulate_point((146, 642.288), (1137.31, 385.201))
        expected_result = (0.0, 100.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_ls_rotated_left(self):
        P0, P1 = setup_second_camera_rotated_left_configuration()
        p = IterativeLS(P0, P1)
        result = p.triangulate_point((878.821, 634.619), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_ls_rotated_right(self):
        P0, P1 = setup_second_camera_rotated_right_configuration()
        p = IterativeLS(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (150.068, 634.618))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_ls_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = IterativeLS(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_eigen_general_setup(self):
        P0, P1 = setup_general_camera_configuration()
        p = LinearEigen(P0, P1)
        result = p.triangulate_point((146, 642.288), (1137.31, 385.201))
        expected_result = (0.0, 100.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_eigen_rotated_left(self):
        P0, P1 = setup_second_camera_rotated_left_configuration()
        p = LinearEigen(P0, P1)
        result = p.triangulate_point((878.821, 634.619), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_eigen_rotated_right(self):
        P0, P1 = setup_second_camera_rotated_right_configuration()
        p = LinearEigen(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (150.068, 634.618))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_linear_eigen_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = LinearEigen(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_eigen_general_setup(self):
        P0, P1 = setup_general_camera_configuration()
        p = IterativeEigen(P0, P1)
        result = p.triangulate_point((146, 642.288), (1137.31, 385.201))
        expected_result = (0.0, 100.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_eigen_rotated_left(self):
        P0, P1 = setup_second_camera_rotated_left_configuration()
        p = IterativeEigen(P0, P1)
        result = p.triangulate_point((878.821, 634.619), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_eigen_rotated_right(self):
        P0, P1 = setup_second_camera_rotated_right_configuration()
        p = IterativeEigen(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (150.068, 634.618))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)

    def test_iterative_eigen_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = IterativeEigen(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)
