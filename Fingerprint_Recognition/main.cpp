#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gabor.hpp"
#include "orientation.hpp"
#include "segmentation.hpp"
#include "thinning.hpp"

using namespace std;
using namespace cv;

Scalar white = CV_RGB(255, 255, 255);
Scalar green = CV_RGB(0, 255, 0);

int main() {
	Mat color;
	Mat src = imread("image/etc/1.bmp");
	cvtColor(src, color, COLOR_RGB2GRAY);

	Mat segmented = segmentation(src);

	pair<Mat, Mat> returned = orientation(segmented, 7);

	Mat show = returned.first;

	Mat orientationMap = returned.second;

	Mat gabored = gabor(segmented);

	Mat imgt = thinning(gabored);

	Mat harris_corners;

	pyrUp(src, src);
	imshow("src", src);

	segmented.convertTo(segmented, CV_8U);
	pyrUp(segmented, segmented);
	imshow("segmented", segmented);

	pyrUp(show, show);
	imshow("show", show);

	pyrUp(orientationMap, orientationMap);
	imshow("orientationMap", orientationMap);

	gabored.convertTo(gabored, CV_8U);
	pyrUp(gabored, gabored);
	imshow("gabored", gabored);

	imgt.convertTo(imgt, CV_8U);
	pyrUp(imgt, imgt);
	imshow("thinned", imgt);
	
	waitKey(0);
}