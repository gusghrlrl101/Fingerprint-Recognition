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
	int block_size = 7;

	Mat src = imread("image/etc/1.bmp");

	// rows, cols must be devided by block size
	resize(src, src, { 154, 203 });

	Mat pyup_src;
	pyrUp(src, pyup_src);
	pyrUp(pyup_src, pyup_src);
	imshow("pyup_src", pyup_src);

//	Mat segmented;
//	cvtColor(src, src, COLOR_BGR2GRAY);
//	adaptiveThreshold(src, segmented, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 2);
//	threshold(src, segmented, 180, 255, THRESH_BINARY);
//	cvtColor(segmented, segmented, COLOR_GRAY2BGR);

	Mat segmented = segmentation(src);
	imshow("segmented", segmented);

	pair<Mat, vector<pair<float, float>>> returned = orientation(segmented, block_size);
	Mat show = returned.first;
	vector<pair<float, float>> vec = returned.second;

	Mat gabored = gabor(segmented, vec, block_size);

	Mat imgt = thinning(gabored);

	Mat result = printMinutiae(imgt, gabored);
//	calculate(imgt, src);

	pyrUp(src, src);
	imshow("src", src);

	segmented.convertTo(segmented, CV_8U);
	pyrUp(segmented, segmented);
	imshow("segmented", segmented);

	pyrUp(show, show);
	imshow("show", show);

	gabored.convertTo(gabored, CV_8U);
	pyrUp(gabored, gabored);
	imshow("gabored", gabored);

	imgt.convertTo(imgt, CV_8U);
	pyrUp(imgt, imgt);
	imshow("thinned", imgt);

	pyrUp(result, result);
	imshow("check", result);

	waitKey(0);
	return 0;
}