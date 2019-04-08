#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "gabor.hpp"
using namespace std;
using namespace cv;

int main() {
	Mat src = imread("image/etc/1.bmp");
	Mat dst = gabor(src);
	pyrUp(src, src);
	pyrUp(dst, dst);
	imshow("SRC", src);
	imshow("gabor", dst);

	waitKey(0);
	return 0;
}