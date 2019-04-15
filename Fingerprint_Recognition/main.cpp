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

int main() {
	Mat src = imread("image/etc/1.bmp");

	Mat segmented = segmentation(src);

	pair<Mat, Mat> returned = orientation(segmented, 7);

	Mat show = returned.first;

	Mat orientationMap = returned.second;

	Mat gabored = gabor(segmented);

	Mat thinned = thinning(gabored);

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

	thinned.convertTo(thinned, CV_8U);
	pyrUp(thinned, thinned);
	imshow("thinned", thinned);

	waitKey(0);
	return 0;
}