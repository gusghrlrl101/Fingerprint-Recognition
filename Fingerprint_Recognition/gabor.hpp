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

	int size = 15;
	double sigma = 5;
	double theta = 0;
	double lambd = 7;
	double gamma = 1;
	double psi = 0;

	int height = src.rows;
	int width = src.cols;
	int index = 0;

	for (int m = 0; m < height; m++){
		for (int n = 0; n < width; n++){
			if ((m % block_size) == 0 && (n % block_size) == 0) {
				float dx = vec[index].first;
				float dy = vec[index].second;

				// 해당 Block의 방향대로 theta를 설정해줌
				theta = atan2f(dy, dx) + CV_PI / 2;

				Mat temp;
				Mat gabor = getGaborKernel({ size, size }, sigma, theta, lambd, gamma, psi);
				filter2D(src, temp, CV_32F, gabor);

				int temp_size = block_size - 1;
				if (width < n + temp_size)
					temp_size = (width - 1) - n;
				if (height < m + block_size - 1 && temp_size >(height - 1) - m)
					temp_size = (height - 1) - m;

				// 해당 block만 이미지를 저장
				for (int i = 0; i < height; i++) {
					for (int j = 0; j < width; j++) {
						if (m <= i && i <= m + temp_size && n <= j && j <= n + temp_size)
							dst.at<float>(i, j) = temp.at<float>(i, j);
					}
				}

				index++;
			}
		}
	}

	dst.convertTo(dst, CV_8U);
	adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);

	return dst;
}

#endif