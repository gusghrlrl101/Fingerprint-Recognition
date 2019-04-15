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

vector<Minutiae> FindMinutiae(cv::Mat& img)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);


	int ending = 0;
	int bifurcation = 0;
	int core = 0;
	int delta = 0;
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
			a = b;
			b = c;
			c = &(pAbove[x + 1]);
			d = e;
			e = f;
			f = &(pCurr[x + 1]);
			g = h;
			h = i;
			i = &(pBelow[x + 1]);

			int endcheck = 0;
			int bifcheck = 0;
			int corecheck = 0;

			//         if(*e == 1 && *a+*b+ *c + *d+*e+*f+*g + *h + *i==1) ending++;
			if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 1 && *h == 1 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) endcheck++, ending++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) endcheck++, ending++;


			else if (*a == 1 && *b == 1 && *c == 0 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 0) bifcheck++, bifurcation++;
			else if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 0 && *h == 1 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 0 && *h == 1 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 1 && *h == 0 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 1 && *c == 0 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) bifcheck++, bifurcation++;

			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 1 && *h == 0 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 0) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 0 && *e == 0 && *f == 0 && *g == 1 && *h == 0 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 0) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 1) bifcheck++, bifurcation++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 0 && *f == 0 && *g == 1 && *h == 1 && *i == 1) bifcheck++, bifurcation++;

			else if (*a == 1 && *b == 0 && *c == 0 && *d == 0 && *e == 1 && *f == 0 && *g == 0 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 0 && *e == 1 && *f == 0 && *g == 0 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 1 && *d == 0 && *e == 1 && *f == 0 && *g == 0 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 0 && *d == 1 && *e == 1 && *f == 0 && *g == 0 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 0 && *d == 0 && *e == 1 && *f == 1 && *g == 0 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 0 && *d == 0 && *e == 1 && *f == 0 && *g == 1 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 0 && *d == 0 && *e == 1 && *f == 0 && *g == 0 && *h == 1 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 0 && *d == 0 && *e == 1 && *f == 0 && *g == 0 && *h == 0 && *i == 1) corecheck++, core++;

			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 1 && *f == 0 && *g == 0 && *h == 0 && *i == 1) corecheck++, core++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 0 && *e == 1 && *f == 1 && *g == 0 && *h == 1 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 1 && *d == 0 && *e == 1 && *f == 0 && *g == 1 && *h == 0 && *i == 1) corecheck++, core++;
			else if (*a == 0 && *b == 0 && *c == 0 && *d == 1 && *e == 1 && *f == 1 && *g == 0 && *h == 1 && *i == 0) corecheck++, core++;
			else if (*a == 1 && *b == 0 && *c == 0 && *d == 0 && *e == 1 && *f == 0 && *g == 1 && *h == 0 && *i == 1) corecheck++, core++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 1 && *f == 0 && *g == 0 && *h == 1 && *i == 0) corecheck++, core++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 1 && *f == 0 && *g == 1 && *h == 0 && *i == 0) corecheck++, core++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 1 && *f == 1 && *g == 0 && *h == 0 && *i == 0) corecheck++, core++;

			if (endcheck == 1) {
				minutiae.x = x;
				minutiae.y = y;
				minutiae.type = 1;
				mVector.push_back(minutiae);
			}
			else if (bifcheck == 1) {
				minutiae.x = x;
				minutiae.y = y;
				minutiae.type = 2;
				mVector.push_back(minutiae);
			}
			else if (corecheck == 1) {
				minutiae.x = x;
				minutiae.y = y;
				minutiae.type = 3;
				mVector.push_back(minutiae);
			}
		}

	}
	Mat dimg;
	pyrDown(img, dimg);

	pAbove = NULL;
	pCurr = dimg.ptr<uchar>(0);
	pBelow = dimg.ptr<uchar>(1);

	for (y = 1; y < dimg.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = dimg.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		b = &(pAbove[0]);
		c = &(pAbove[1]);
		e = &(pCurr[0]);
		f = &(pCurr[1]);
		h = &(pBelow[0]);
		i = &(pBelow[1]);

		for (x = 1; x < dimg.cols - 1; ++x) {
			a = b;
			b = c;
			c = &(pAbove[x + 1]);
			d = e;
			e = f;
			f = &(pCurr[x + 1]);
			g = h;
			h = i;
			i = &(pBelow[x + 1]);

			int deltacheck = 0;

			if (*a == 1 && *b == 1 && *c == 0 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) deltacheck++, delta++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 0) deltacheck++, delta++;
			else if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 0 && *h == 1 && *i == 1) deltacheck++, delta++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) deltacheck++, delta++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 0 && *h == 1 && *i == 1) deltacheck++, delta++;
			else if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 1 && *h == 0 && *i == 1) deltacheck++, delta++;
			else if (*a == 1 && *b == 1 && *c == 0 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) deltacheck++, delta++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) deltacheck++, delta++;

			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 1 && *h == 1 && *i == 0) deltacheck++, delta++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 1 && *e == 0 && *f == 0 && *g == 1 && *h == 0 && *i == 1) deltacheck++, delta++;
			else if (*a == 1 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 0) deltacheck++, delta++;
			else if (*a == 1 && *b == 1 && *c == 1 && *d == 0 && *e == 0 && *f == 0 && *g == 1 && *h == 0 && *i == 1) deltacheck++, delta++;
			else if (*a == 0 && *b == 1 && *c == 1 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 0) deltacheck++, delta++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 0 && *f == 1 && *g == 1 && *h == 0 && *i == 1) deltacheck++, delta++;
			else if (*a == 0 && *b == 1 && *c == 0 && *d == 1 && *e == 0 && *f == 1 && *g == 0 && *h == 1 && *i == 1) deltacheck++, delta++;
			else if (*a == 1 && *b == 0 && *c == 1 && *d == 0 && *e == 0 && *f == 0 && *g == 1 && *h == 1 && *i == 1) deltacheck++, delta++;

			if (deltacheck == 1) {
				minutiae.x = x * 8;
				minutiae.y = y * 8;
				minutiae.type = 4;
				int check = 0;

				for (int j = 0; j < mVector.size(); j++) {
					if (mVector[j].x == minutiae.x && mVector[j].y == minutiae.y) {
						mVector[j].type = 4;
						check++;
						break;
					}
				}
				if (check == 0) mVector.push_back(minutiae);
			}
		}
	}

	cout << "ending: " << ending << ", bifurcation: " << bifurcation << ", core: " << core << endl;

	for (int i = 0; i < mVector.size(); i++) {
		if (mVector[i].type == 4) cout << "delta" << endl;
	}
	img &= ~marker;

	return mVector;
}




void MinutiaeCheck(const cv::Mat& src, cv::Mat& dst)
{
	dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;


	vector<Minutiae> mVector = FindMinutiae(dst);
	dst *= 255;
	cvtColor(dst, dst, COLOR_GRAY2RGB);

	Scalar end = Scalar(000, 000, 255);
	Scalar bif = Scalar(204, 051, 000);
	Scalar core = Scalar(000, 255, 255);
	Scalar delta = Scalar(000, 204, 204);



	for (int i = 0; i < mVector.size(); i++) {
		if (mVector[i].type == 1) circle(dst, Point(mVector[i].x, mVector[i].y), 5, end, 1, 8);
		else if (mVector[i].type == 2) circle(dst, Point(mVector[i].x, mVector[i].y), 5, bif, 1, 8);
		else if (mVector[i].type == 3) circle(dst, Point(mVector[i].x, mVector[i].y), 5, core, 1, 8);
		else if (mVector[i].type == 4) circle(dst, Point(mVector[i].x, mVector[i].y), 5, delta, 1, 8);
	}

}