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

Scalar white = CV_RGB(255, 255, 255);
Scalar green = CV_RGB(0, 255, 0);


int main() {
	int block_size = 7;
	Mat src = imread("E:\\good_quality_FP.jpg");
	resize(src, src, { 154, 203 });

	Mat temp_src;
	pyrUp(src, temp_src);
	pyrUp(temp_src, temp_src);
	imshow("temp_src", temp_src);

//	Mat segmented;
	//cvtColor(src, src, COLOR_BGR2GRAY);
//	adaptiveThreshold(src, segmented, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 2);
//	threshold(src, segmented, 180, 255, THRESH_BINARY);
//	cvtColor(segmented, segmented, COLOR_GRAY2BGR);

	Mat segmented = segmentation(src);
	imshow("segmented", segmented);

	pair<Mat, vector<pair<pair<float, float>,int>>> returned = orientation(segmented, block_size);
	Mat show = returned.first;
	vector<pair<pair<float, float>,int>> vec = returned.second;

	Mat gabored = gabor(segmented, vec, block_size);

	vector<Minutiae> minutiaes;
	vector<Point> ending;
	vector<Point> bif;
	vector<Point> core;
	vector<Point> delta;
	int endingN = 0, bifN = 0, coreN = 0, deltaN = 0;

	Mat imgt = thinning(gabored);

	//cvtColor(imgt, imgt, COLOR_BGR2GRAY);
	//threshold(imgt, imgt, 127, 255, THRESH_BINARY_INV);
	imgt /= 255;
	minutiaes = FindMinutiae(imgt);
	imgt *= 255;

	cout << minutiaes.size() << endl;

	for (int i = 0; i < minutiaes.size(); i++) {
		if (minutiaes[i].type == 1) {
			Point temp = { minutiaes[i].x, minutiaes[i].y };
			ending.push_back(temp);
			endingN++;
		}
		else if (minutiaes[i].type == 2) {
			Point temp = { minutiaes[i].x, minutiaes[i].y };
			bif.push_back(temp);
			bifN++;
		}
		else if (minutiaes[i].type == 3) {
			Point temp = { minutiaes[i].x, minutiaes[i].y };
			core.push_back(temp);
			coreN++;
		}
		else if (minutiaes[i].type == 4) {
			Point temp = { minutiaes[i].x, minutiaes[i].y };
			delta.push_back(temp);
			deltaN++;
		}
	}
	cout << "ending : " << endingN << " bif : " << bifN << " core : " << coreN << " delta : " << deltaN << endl;
	int count = 1;
	int distanceMean = 0, distanceMax = 0, distanceMin = 987654321;
	int distanceN = 0;
	for (int i = 0; i < ending.size(); i++) {
		for (int j = 0; j < core.size(); j++) {
			cout << "#" << count << "¹øÂ°" << endl;
			Point temp1 = { ending[i].x, ending[i].y };
			Point temp2 = { core[j].x, core[j].y };
			if(temp1.x<=temp2.x)
				distanceN = distance(imgt, temp1, temp2);
			else
				distanceN = distance(imgt, temp2, temp1);
			count++;
			if (distanceN > distanceMax)
				distanceMax = distanceN;
			if (distanceN < distanceMin)
				distanceMin = distanceN;
			distanceMean += distanceN;
		}
	}

	distanceMean /= count;

	Mat result;

	MinutiaeCheck(imgt, result);

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

//	imgt.convertTo(imgt, CV_8U);
	pyrUp(result, result);
	imshow("check", result);

	
	
	waitKey(0);
}