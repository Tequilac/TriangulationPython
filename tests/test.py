from unittest import TestCase

import numpy as np
import pytest
from fixtures import setup_horizontal_configuration
from iterative_eigen.iterative_eigen import IterativeEigen


def evaluate_result(result: tuple[float, float, float], expected_result: tuple[float, float, float], max_percentage_error=0.001):
    tolerance = np.linalg.norm(expected_result) * max_percentage_error
    distance = np.linalg.norm(np.array(result) - np.array(expected_result))
    print(result)
    print(expected_result)
    print(pytest.approx(distance, 0.0, tolerance))
    print(distance)
    assert pytest.approx(distance, 0.0, tolerance) == distance


class Test(TestCase):
    # TEST(PolyTest, GeneralSetup)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupGeneralCameraConfiguration();
    # Triangulation::Poly p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(146, 642.288), cv::Point2d(1137.31, 385.201));
    # cv::Point3d expected_result(0.0, 100.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyTest, RotatedLeft)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedLeftConfiguration();
    # Triangulation::Poly p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(878.821, 634.619), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyTest, RotatedRight)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedRightConfiguration();
    # Triangulation::Poly p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(150.068, 634.618));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyTest, HorizontalStereo)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupHorizontalConfiguration();
    # Triangulation::Poly p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyAbsTest, GeneralSetup)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupGeneralCameraConfiguration();
    # Triangulation::PolyAbs p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(146, 642.288), cv::Point2d(1137.31, 385.201));
    # cv::Point3d expected_result(0.0, 100.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyAbsTest, RotatedLeft)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedLeftConfiguration();
    # Triangulation::PolyAbs p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(878.821, 634.619), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyAbsTest, RotatedRight)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedRightConfiguration();
    # Triangulation::PolyAbs p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(150.068, 634.618));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(PolyAbsTest, HorizontalStereo)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupHorizontalConfiguration();
    # Triangulation::PolyAbs p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(LinearLSTest, GeneralSetup)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupGeneralCameraConfiguration();
    # Triangulation::LinearLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(146, 642.288), cv::Point2d(1137.31, 385.201));
    # cv::Point3d expected_result(0.0, 100.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(LinearLSTest, RotatedLeft)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedLeftConfiguration();
    # Triangulation::LinearLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(878.821, 634.619), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(LinearLSTest, RotatedRight)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedRightConfiguration();
    # Triangulation::LinearLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(150.068, 634.618));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(LinearLSTest, HorizontalStereo)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupHorizontalConfiguration();
    # Triangulation::LinearLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(IterativeLSTest, GeneralSetup)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupGeneralCameraConfiguration();
    # Triangulation::IterativeLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(146, 642.288), cv::Point2d(1137.31, 385.201));
    # cv::Point3d expected_result(0.0, 100.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(IterativeLSTest, RotatedLeft)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedLeftConfiguration();
    # Triangulation::IterativeLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(878.821, 634.619), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(IterativeLSTest, RotatedRight)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupSecondCameraRotatedRightConfiguration();
    # Triangulation::IterativeLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(150.068, 634.618));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(IterativeLSTest, HorizontalStereo)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupHorizontalConfiguration();
    # Triangulation::IterativeLS p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }
    #
    # TEST(LinearEigenTest, HorizontalStereo)
    # {
    # cv::Mat P0, P1;
    # std::tie(P0, P1) = SetupHorizontalConfiguration();
    # Triangulation::LinearEigen p(P0, P1);
    # cv::Point3d result = p.triangulate(cv::Point2d(1004.08, 511.5), cv::Point2d(274.917, 511.5));
    # cv::Point3d expected_result(500.0, 0.0, 10000.0);
    # EvaluateResult(result, expected_result);
    # }

    def test_iterative_eigen_horizontal_stereo(self):
        P0, P1 = setup_horizontal_configuration()
        p = IterativeEigen(P0, P1)
        result = p.triangulate_point((1004.08, 511.5), (274.917, 511.5))
        expected_result = (500.0, 0.0, 10000.0)
        evaluate_result(result, expected_result)
