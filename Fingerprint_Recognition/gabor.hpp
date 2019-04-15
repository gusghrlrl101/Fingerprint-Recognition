#ifndef GABOR_HPP
#define GABOR_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

Mat gabor(Mat src, vector<pair<float, float>>& vec, int block_size) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_32F);

	int size = 10;
	double sigma = 7;
	double theta = 0;
	double lambd = 7;
	double gamma = 1;
	double psi = CV_PI / 2;

	int height = src.rows;
	int width = src.cols;
	int index = 0;
	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			if ((m % block_size) == 0 && (n % block_size) == 0) {
				float dx = vec[index].first;
				float dy = vec[index].second;
				theta = atan2f(dy, dx) + CV_PI / 2;
				
				Mat temp;
				Mat gabor = getGaborKernel({ size, size }, sigma, theta, lambd, gamma, psi);
				filter2D(src, temp, CV_32F, gabor);
				cvtColor(temp, temp, COLOR_BGR2GRAY);

				int temp_size = block_size - 1;
				if (width < n + temp_size)
					temp_size = (width - 1) - n;
				if (height < m + block_size - 1 && temp_size > (height - 1) - m)
					temp_size = (height - 1) - m;

				Mat ttemp = temp;

				for (int i = 0; i < height; i++) {
					for (int j = 0; j < width; j++) {
						if(m <= i && i <= m + temp_size && n <= j && j <= n + temp_size)
							ttemp.at<float>(i, j) = 1000*  temp.at<float>(i, j);
					}
				}
				dst += ttemp;

				index++;
			}
		}
	}
	dst.convertTo(dst, CV_8U);
	threshold(dst, dst, 80, 255, THRESH_BINARY_INV);

	return dst;
	dst = Mat::zeros(src.rows, src.cols, CV_32F);

	int iter = 8;
	for (int i = 0; i < iter; i++) {
		theta = CV_PI / iter * i;
		Mat temp;
		Mat gabor1 = getGaborKernel({ size,size }, sigma, theta, lambd, gamma, psi);
		filter2D(src, temp, CV_32F, gabor1);
		cvtColor(temp, temp, COLOR_BGR2GRAY);

		addWeighted(dst, 1, temp, 1, 0, dst, CV_32F);
	}
	dst.convertTo(dst, CV_8U);
	threshold(dst, dst, 80, 255, THRESH_BINARY_INV);

	return dst;
}

#endif