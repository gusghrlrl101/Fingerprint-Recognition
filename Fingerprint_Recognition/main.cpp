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

	Mat src = imread("image/team1/2019_1_1_L_I_1.bmp");
	Size size = { 154,203 };
	cvtColor(src, src, COLOR_RGB2GRAY);

	// rows, cols must be devided by block size
	resize(src, src, size);

	Mat pyup_src;
	pyrUp(src, pyup_src);
	imshow("pyup_src", pyup_src);

	Mat segmented;
	Mat segmented2 = segmentation(src, segmented);
	equalizeHist(src, src);

	imshow("segmented", segmented);
	imshow("segmented2", segmented2);

	
	pair<Mat, vector<pair<float, float>>> returned = orientation(src, block_size);
	Mat show = returned.first;
	vector<pair<float, float>> vec = returned.second;

	pair<Mat, vector<pair<float, float>>> returned2 = orientation(src, 7, true);
	Mat coredelta = returned2.first;
	pyrUp(coredelta, coredelta);
	imshow("coredelta", coredelta);


	Mat gabored = gabor(src, vec, block_size) + segmented2;

	Mat gabored_end;
	threshold(gabored, gabored_end, 1, 255, THRESH_BINARY_INV);


	Mat imgt = thinning(gabored_end);

	Mat result = printMinutiae(imgt, segmented2, vec, block_size, size, src);
	calculate(imgt, segmented2);

	pyrUp(src, src);
	imshow("src", src);

//	segmented.convertTo(segmented, CV_8U);
//	pyrUp(segmented, segmented);
//	imshow("segmented", segmented);

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
	
	cout << "³¡!" << endl;
	waitKey(0);
	return 0;
}