
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat segmentation(Mat &src, Mat &dst) {

	src.convertTo(dst, CV_8UC1);

	Mat mask = getStructuringElement(1, Size(3, 3), Point(1, 1));
//	adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 13, 3);

	imshow("before dst", dst);
	erode(dst, dst, mask);

	Mat seg;

	morphologyEx(dst, seg, MORPH_OPEN, mask, Point(-1, -1), 12);
	imshow("seg", seg);
	imshow("dst", dst);
	Mat seg_inv;
	threshold(seg, seg_inv, 70, 255, THRESH_BINARY_INV);


	return seg;
}