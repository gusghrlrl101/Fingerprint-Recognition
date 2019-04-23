/*
2019 DIP Team Project #1
FingerPrint Reconition
by Team1 (권동현, 임정혁, 임현호, 최진우)
*/

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gabor.hpp"
#include "orientation.hpp"
#include "segmentation.hpp"
#include "thinning.hpp"
#include "Minutiae.hpp"
#include "distance.hpp"

using namespace std;
using namespace cv;

int main() {
	// orientation block size
	// rows, cols must be devided by block size
	int block_size = 7;

	Mat src = imread("image/etc/50.bmp");
	Size size = { 154,203 };
	cvtColor(src, src, COLOR_RGB2GRAY);

	// resize image
	resize(src, src, size);

	Mat segmented;
	// segmantation image
	Mat segmented2 = segmentation(src, segmented);
	// normalize image
	equalizeHist(src, src);


	// block oriented
	pair<Mat, vector<pair<float, float>>> returned = orientation(src, block_size);
	Mat show = returned.first;
	vector<pair<float, float>> vec = returned.second;

	pair<Mat, vector<pair<float, float>>> returned2 = orientation(src, 7, true);
	Mat coredelta = returned2.first;

	// gabor filter
	Mat gabored = gabor(src, vec, block_size) + segmented2;

	Mat gabored_end;
	// binarization
	threshold(gabored, gabored_end, 1, 255, THRESH_BINARY_INV);

	// thinning
	Mat imgt = thinning(gabored_end);

	// find minutiae and visual them
	Mat result = printMinutiae(imgt, segmented2, vec, block_size, size, src);
	// measure distance between ridge
	calculate(imgt, segmented2);

	pyrUp(src, src);
	imshow("src", src);

	pyrUp(show, show);
	imshow("show", show);
	
	pyrUp(segmented2, segmented2);
	imshow("segmented area", segmented2);

	gabored.convertTo(gabored, CV_8U);
	pyrUp(gabored, gabored);
	imshow("gabored", gabored);

	imgt.convertTo(imgt, CV_8U);
	pyrUp(imgt, imgt);
	imshow("thinned", imgt);

	pyrUp(result, result);
	imshow("check", result);

	pyrUp(coredelta, coredelta);
	imshow("coredelta", coredelta);

	cout << "끝!" << endl;

	waitKey(0);
	return 0;
}