/*
2019 DIP Team Project #1
FingerPrint Reconition
by Team1 (권동현, 임정혁, 임현호, 최진우)
*/

#include <iostream>
#include <vector>
#include <fstream>
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
	int W = 154;                     //width
	int H = 203;                     //height
	int M = 1;                        //number of extracted minutiae
	int SP = 1;                        //singular point
	int X[50];                     //x coordinates of minutiae 
	int Y[50];                     //y coordinates of minutiae
	unsigned char O[50];            //Orientation of minutiae (0~359)
	unsigned char T[50];                     //type of minutiae (1:ending   3:bifurcation   10:core   11:delta)

	ofstream output("2019_1_1_L_T_1.bin", ios::out | ios::binary);


	Mat src = imread("image/etc/29.bmp");
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

	pair<Mat, vector<pair<float, float>>> returned2 = orientation(src, 7, true, &SP, X, Y, O, T);
	Mat coredelta = returned2.first;

	// gabor filter
	Mat gabored = gabor(src, vec, block_size) + segmented2;

	Mat gabored_end;
	// binarization
	threshold(gabored, gabored_end, 1, 255, THRESH_BINARY_INV);

	// thinning
	Mat imgt = thinning(gabored_end);

	// find minutiae and visual them
	Mat result = printMinutiae(imgt, segmented2, vec, block_size, size, src, &M, SP, X, Y, O, T);
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

	//Minutiae Data 값 입력
	output.write((char*)&W, sizeof(int));
	output.write((char*)&H, sizeof(int));
	output.write((char*)&M, sizeof(int));
	output.write((char*)&SP, sizeof(int));
	for (int i = 0; i < 50; i++) {
		output.write((char*)&X[i], sizeof(int));
		output.write((char*)&Y[i], sizeof(int));
		output.write((char*)&O[i], sizeof(char));
		output.write((char*)&T[i], sizeof(char));
	}


	waitKey(0);
	return 0;
}