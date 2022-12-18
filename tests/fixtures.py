

# std::pair<cv::Mat, cv::Mat> SetupGeneralCameraConfiguration()
# {
#     cv::Mat P0 = cv::Mat::eye(3, 4, CV_64F);
# cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
#
# P0.at<double>(0, 0) = 0.999701;
# P0.at<double>(0, 1) = 0.0174497;
# P0.at<double>(0, 2) = -0.017145;
# P0.at<double>(0, 3) = -500;
# P0.at<double>(1, 0) = -0.0171452;
# P0.at<double>(1, 1) = 0.999695;
# P0.at<double>(1, 2) = 0.0177517;
# P0.at<double>(1, 3) = -100;
# P0.at<double>(2, 0) = 0.0174497;
# P0.at<double>(2, 1) = -0.0174524;
# P0.at<double>(2, 2) = 0.999695;
# P0.at<double>(2, 3) = -100;
#
# P1.at<double>(0, 0) = 0.99969;
# P1.at<double>(0, 1) = -0.0174497;
# P1.at<double>(0, 2) = 0.0177543;
# P1.at<double>(0, 3) = 500;
# P1.at<double>(1, 0) = 0.0177543;
# P1.at<double>(1, 1) = 0.999695;
# P1.at<double>(1, 2) = -0.0171425;
# P1.at<double>(1, 3) = -100;
# P1.at<double>(2, 0) = -0.0174497;
# P1.at<double>(2, 1) = 0.0174524;
# P1.at<double>(2, 2) = 0.999695;
# P1.at<double>(2, 3) = -100;
#
# cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
# K.at<double>(0, 0) = 7291.67;
# K.at<double>(1, 1) = 7291.67;
# K.at<double>(0, 2) = 639.5;
# K.at<double>(1, 2) = 511.5;
#
# P0 = K * P0;
# P1 = K * P1;
# return std::make_pair(P0, P1);
# }
#
# std::pair<cv::Mat, cv::Mat> SetupSecondCameraRotatedLeftConfiguration()
# {
# cv::Mat P0 = cv::Mat::eye(3, 4, CV_64F);
# cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
#
# P0.at<double>(0, 0) = 0.999701;
# P0.at<double>(0, 1) = 0.0174497;
# P0.at<double>(0, 2) = -0.017145;
# P1.at<double>(0, 3) = -1000;
# P0.at<double>(1, 0) = -0.0171452;
# P0.at<double>(1, 1) = 0.999695;
# P0.at<double>(1, 2) = 0.0177517;
# P0.at<double>(1, 3) = 0;
# P0.at<double>(2, 0) = 0.0174497;
# P0.at<double>(2, 1) = -0.0174524;
# P0.at<double>(2, 2) = 0.999695;
# P0.at<double>(2, 3) = 0;
#
# cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
# K.at<double>(0, 0) = 7291.67;
# K.at<double>(1, 1) = 7291.67;
# K.at<double>(0, 2) = 639.5;
# K.at<double>(1, 2) = 511.5;
#
# P0 = K * P0;
# P1 = K * P1;
# return std::make_pair(P0, P1);
# }
#
# std::pair<cv::Mat, cv::Mat> SetupSecondCameraRotatedRightConfiguration()
# {
# cv::Mat P0 = cv::Mat::eye(3, 4, CV_64F);
# cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
#
# P1.at<double>(0, 0) = 0.999701;
# P1.at<double>(0, 1) = 0.0174497;
# P1.at<double>(0, 2) = -0.017145;
# P1.at<double>(0, 3) = -1000;
# P1.at<double>(1, 0) = -0.0171452;
# P1.at<double>(1, 1) = 0.999695;
# P1.at<double>(1, 2) = 0.0177517;
# P1.at<double>(1, 3) = 0;
# P1.at<double>(2, 0) = 0.0174497;
# P1.at<double>(2, 1) = -0.0174524;
# P1.at<double>(2, 2) = 0.999695;
# P1.at<double>(2, 3) = 0;
#
# cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
# K.at<double>(0, 0) = 7291.67;
# K.at<double>(1, 1) = 7291.67;
# K.at<double>(0, 2) = 639.5;
# K.at<double>(1, 2) = 511.5;
#
# P0 = K * P0;
# P1 = K * P1;
# return std::make_pair(P0, P1);
# }
import numpy as np


def setup_horizontal_configuration():
    P0 = np.eye(3, 4)
    P1 = np.eye(3, 4)

    P1[0, 3] = -1000

    K = np.eye(3, 3)
    K[0, 0] = 7291.67
    K[1, 1] = 7291.67
    K[0, 2] = 639.5
    K[1, 2] = 511.5

    P0 = np.matmul(K, P0)
    P1 = np.matmul(K, P1)
    return P0, P1
