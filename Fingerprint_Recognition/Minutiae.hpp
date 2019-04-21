#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>


using namespace std;
using namespace cv;


struct Minutiae {
	int x;
	int y;
	int angle;
	int type; //ending:1  bifurcation:2
};


vector<Minutiae> findMinutiae(Mat& img, Mat& seg) {
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);


	Mat area;
	seg.convertTo(area, CV_8UC1);
	Mat mask = getStructuringElement(1, Size(3, 3), Point(1, 1));
	dilate(area, area, mask, Point(-1, -1), 7);

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

			int thr = 5;
			if (*e == 1 && (sum == 1)) {
				uchar* segVal = &(area.ptr<uchar>(y))[x];
				//				if (*segVal == 0) {
				bool isAlready = false;
				for (auto mnt : mVector) {
					int distt = abs(mnt.x - x) + abs(mnt.y - y);
					if (distt <= thr) {
						isAlready = true;
						break;
					}
				}

				if (!isAlready) {
					if (0 <= x - 3 && x + 3 < img.cols && 0 <= y - 3 && y + 3 < img.rows) {
						ending++;
						minutiae.x = x; minutiae.y = y;
						minutiae.type = 1;
						mVector.push_back(minutiae);
					}
				}
				//				}
			}
			if (*e == 1 && (xor_ == 6 || (xor_ == 6 && and_ == 2))) {
				uchar* segVal = &(area.ptr<uchar>(y))[x];
				//				if (*segVal == 0) {
				bool isAlready = false;
				for (auto mnt : mVector) {
					int distt = abs(mnt.x - x) + abs(mnt.y - y);
					if (distt <= thr) {
						isAlready = true;
						break;
					}
				}

				if (!isAlready) {
					if (0 <= x - 3 && x + 3 < img.cols && 0 <= y - 3 && y + 3 < img.rows) {
						bifurcation++;
						minutiae.x = x; minutiae.y = y;
						minutiae.type = 2;
						mVector.push_back(minutiae);
					}
				}
				//				}
			}

		}
	}
	cout << "ending: " << ending << ", bifurcation: " << bifurcation << endl;

	return mVector;
}


float angle(vector<pair<float, float>>& vec, int& u, int& v, int& block_size, Size size) {
	cout << "angle: " << u << ", " << v << endl;
	float fi = 0.0;

	int width = size.width / block_size;
	int height = size.height / block_size;

	cout << width << ", " << height << endl;



	return fi;
}


Mat printMinutiae(Mat src, Mat& srcc, vector<pair<float,float>>& vec, int& block_size, Size size) {
	Mat temp;
	Mat dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);

	vector<Minutiae> mVector = findMinutiae(dst, srcc);
	dst *= 255;
	cvtColor(dst, dst, COLOR_GRAY2RGB);
	threshold(dst, dst, 127, 255, THRESH_BINARY_INV);

	Scalar end = Scalar(255, 255, 000);
	Scalar bif = Scalar(000, 255, 255);

	for (int i = 0; i < mVector.size(); i++) {
		if (mVector[i].type == 1)
			circle(dst, Point(mVector[i].x, mVector[i].y), 5, end, 1, 8);
		else if (mVector[i].type == 2)
			rectangle(dst, Point(mVector[i].x - 4, mVector[i].y - 4), Point(mVector[i].x + 4, mVector[i].y + 4), bif, 1);

		cout << mVector[i].x << ", " << mVector[i].y << ": " << angle(vec, mVector[i].x, mVector[i].y, block_size, size) << endl;
		imshow("drawing", dst);
		waitKey(0);
	}

	return dst;
}