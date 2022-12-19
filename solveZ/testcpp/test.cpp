#include <opencv2/opencv.hpp>
#include <iostream>


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

void
mainSolveZ(cv::Mat m, cv::Mat& dst)
{
    cv::SVD svd(m, (m.rows >= m.cols ? 0 : cv::SVD::FULL_UV));
    dst.create(svd.vt.cols, 1, svd.vt.type());
    svd.vt.row(svd.vt.rows-1).reshape(1,svd.vt.cols).copyTo(dst);
    print_mat(svd.u);
    print_mat(svd.vt);
    print_mat(svd.w);
}

int
main()
{
    int rows, cols;
    double **m;

    std::cin >> rows >> cols;
    m = new double *[rows];

    for (int ii = 0; ii < rows; ii++) {
        m[ii] = new double[cols];
	for (int jj = 0; jj < cols; jj++) {
            std::cin >> m[ii][jj];
	}
    }

    cv::Mat A = cv::Mat(rows, cols, CV_64F);
    cv::Mat x;
    for (int ii = 0; ii < rows; ii++) {
        for (int jj = 0; jj < cols; jj++) {
            A.at<double>(ii, jj) = m[ii][jj];
        }
    }

//    cv::Mat y;

//    print_mat(A);

    cv::SVD::solveZ(A, x);

    print_mat(x);

//    mainSolveZ(A, y);
//    std::cout << std::endl;
//    print_mat(y);

    for (int ii = 0; ii < rows; ii++) {
        delete m[ii];
    }
    delete m;
}
