#ifndef GABOR_HPP
#define GABOR_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

Mat gabor(Mat src) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_32F);

	int size = 10;
	double sigma = 5;
	double theta = 0;
	double lambd = 7;
	double gamma = 1;
	double psi = CV_PI / 2;

	int iter = 90;
	for (int i = 0; i < iter; i++) {
		theta = CV_PI / iter * i;
		Mat temp;
		Mat gabor1 = getGaborKernel({ size,size }, sigma, theta, lambd, gamma, psi);
		filter2D(src, temp, CV_32F, gabor1);
		cvtColor(temp, temp, COLOR_BGR2GRAY);

		addWeighted(dst, 1, temp, 1, 0, dst, CV_32F);
	}
	dst.convertTo(dst, CV_8U);

	return dst;
}

#endif