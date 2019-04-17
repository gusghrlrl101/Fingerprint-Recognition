#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

struct Minutiae {
	int x;
	int y;
	int type; //ending:1  bifurcation:2  core:3  delta:4 
};

vector<Minutiae> findMinutiae(cv::Mat& img)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int ending = 0;
	int bifurcation = 0;
	vector<Minutiae> mVector;
	Minutiae minutiae;

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *a, *b, *c;    // north (pAbove)
	uchar *d, *e, *f;
	uchar *g, *h, *i;    // south (pBelow)

	uchar *pDst;

	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		b = &(pAbove[0]);
		c = &(pAbove[1]);
		e = &(pCurr[0]);
		f = &(pCurr[1]);
		h = &(pBelow[0]);
		i = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			a = b;   b = c;   c = &(pAbove[x + 1]);
			d = e;   e = f;   f = &(pCurr[x + 1]);
			g = h;   h = i;   i = &(pBelow[x + 1]);

			int sum = *a + *b + *c + *d + *f + *g + *h + *i;
			int xor_ = (*a ^ *b) + (*b ^ *c) + (*c ^ *f) + (*f ^ *i) + (*d ^ *g) + (*g ^ *h) + (*h ^ *i) + (*d ^ *a);
			int and_ = (*a & *b) + (*b & *c) + (*c & *f) + (*f & *i) + (*d & *g) + (*g & *h) + (*h & *i) + (*d & *a);

			if (*e == 1 && (sum == 1)) {
				ending++;
				minutiae.x = x; minutiae.y = y;
				minutiae.type = 1;
				mVector.push_back(minutiae);
			}
			if (*e == 1 && (xor_ == 6 || (xor_ == 6 && and_ == 2))) {
				bifurcation++;
				minutiae.x = x; minutiae.y = y;
				minutiae.type = 2;
				mVector.push_back(minutiae);
			}
		}
	}
	cout << "ending: " << ending << ", bifurcation: " << bifurcation << endl;

	return mVector;

}

Mat printMinutiae(const cv::Mat& src) {
	Mat dst = src.clone();
	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	dst /= 255;
	vector<Minutiae> mVector = findMinutiae(dst);
	dst *= 255;

	cvtColor(dst, dst, COLOR_GRAY2RGB);

	Scalar end = Scalar(255, 255, 000);
	Scalar bif = Scalar(000, 255, 255);


	for (int i = 0; i < mVector.size(); i++) {
		if (mVector[i].type == 1)
			circle(dst, Point(mVector[i].x, mVector[i].y), 5, end, 1, 8);
		else if (mVector[i].type == 2)
			rectangle(dst, Point(mVector[i].x - 4, mVector[i].y - 4), Point(mVector[i].x + 4, mVector[i].y + 4), bif, 1);
	}

	return dst;
}