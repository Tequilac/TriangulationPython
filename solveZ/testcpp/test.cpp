#include <opencv2/opencv.hpp>
#include <iostream>


constexpr int size = 3;

void
print_mat(const cv::Mat& x)
{
    const int rows = x.rows;
    const int cols = x.cols;

    for (int ii = 0; ii < rows; ii++) {
        for (int jj = 0; jj < cols; jj++) {
            std::cout << x.at<double>(ii, jj) << ' ';
        }
        std::cout << std::endl;
    }
}

int
main()
{
    double m[size][size] = {{0, 1, 2},
                            {3, 4, 2},
                            {1, 6, 3}};

    cv::Mat A = cv::Mat(size, size, CV_64F, m);
    cv::Mat x;

    print_mat(A);

    cv::SVD::solveZ(A, x);

    print_mat(x);
}
