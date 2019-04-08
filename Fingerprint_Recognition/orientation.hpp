#ifndef ORENTATION_HPP
#define ORENTATION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


// gray 변환 후 CV_32F로 변환 필요

pair<Mat,Mat> orientation(Mat src, int size = 8)
{
	Mat inputImage = src;

	cvtColor(src, inputImage, COLOR_RGB2GRAY);
	inputImage.convertTo(inputImage, CV_32F, 1.0 / 255, 0);

	medianBlur(inputImage, inputImage, 3);
	imshow("medianBlur", inputImage);

	int blockSize = size;// SPECIFY THE BLOCKSIZE;

	Mat orientationMap;

	Mat fprintWithDirectionsSmoo = inputImage.clone();
	Mat tmp(inputImage.size(), inputImage.type());
	Mat coherence(inputImage.size(), inputImage.type());
	orientationMap = tmp.clone();

	//Gradiants x and y
	Mat grad_x, grad_y;

	// TO DO#2:

	//CASE:1- USE SOBEL OPERATOR OPENCV SYNTAX (INPUT IMAGE, SOBEL_OUTPUT, OTHER PARAMETERS) --> APPLY BOTH X-DIRECTION & Y-DIRECTION
	Sobel(inputImage, grad_x, inputImage.depth(), 1, 0, 3);
	Sobel(inputImage, grad_y, inputImage.depth(), 0, 1, 3);
	//CASE:2- USE SCHARR OPERATOR OPENCV SYNTAX (INPUT IMAGE, SCHARR_OUTPUT, OTHER PARAMETERS) --> APPLY BOTH X-DIRECTION & Y-DIRECTION  
	//NOTE: WHEN YOU EXECUTE THE PROGRAM USE CASE:1 OR CASE:2 NOT BOTH AT THE SAME TIME

	//Vector vield
	Mat Fx(inputImage.size(), inputImage.type()),
		Fy(inputImage.size(), inputImage.type()),
		Fx_gauss,
		Fy_gauss;
	Mat smoothed(inputImage.size(), inputImage.type());

	// Local orientation for each block
	int width = inputImage.cols;
	int height = inputImage.rows;
	int blockH;
	int blockW;

	//select block
	for (int i = 0; i < height; i += blockSize)
	{
		for (int j = 0; j < width; j += blockSize)
		{
			float Gsx = 0.0;
			float Gsy = 0.0;
			float Gxx = 0.0;
			float Gyy = 0.0;

			//for check bounds of img
			blockH = ((height - i) < blockSize) ? (height - i) : blockSize;
			blockW = ((width - j) < blockSize) ? (width - j) : blockSize;

			//average at block WхW
			for (int u = i; u < i + blockH; u++)
			{
				for (int v = j; v < j + blockW; v++)
				{
					Gsx += (grad_x.at<float>(u, v)*grad_x.at<float>(u, v)) - (grad_y.at<float>(u, v)*grad_y.at<float>(u, v));
					Gsy += 2 * grad_x.at<float>(u, v) * grad_y.at<float>(u, v);
					Gxx += grad_x.at<float>(u, v)*grad_x.at<float>(u, v);
					Gyy += grad_y.at<float>(u, v)*grad_y.at<float>(u, v);
				}
			}

			float coh = sqrt(pow(Gsx, 2) + pow(Gsy, 2)) / (Gxx + Gyy);
			//smoothed
			float fi = 0.5*fastAtan2(Gsy, Gsx)*CV_PI / 180;

			Fx.at<float>(i, j) = cos(2 * fi);
			Fy.at<float>(i, j) = sin(2 * fi);

			//fill blocks
			for (int u = i; u < i + blockH; u++)
			{
				for (int v = j; v < j + blockW; v++)
				{
					orientationMap.at<float>(u, v) = fi;
					Fx.at<float>(u, v) = Fx.at<float>(i, j);
					Fy.at<float>(u, v) = Fy.at<float>(i, j);
					coherence.at<float>(u, v) = (coh < 0.85) ? 1 : 0;
				}
			}

		}
	} ///for

	  // TO DO#3:

	  // DO GAUSSIAN BLUR SMOOTHING (BOTH IN X & Y DIRECTIONS) --> TAKE Fx & Fy AS INPUTS AND GET Fx_gauss & Fy_gauss AS OUTPUTS WITH KERNEL SIZE 5X5
	GaussianBlur(Fx, Fx_gauss, Size(5, 5), 1, 0);
	GaussianBlur(Fy, Fy_gauss, Size(5, 5), 0, 1);
	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			smoothed.at<float>(m, n) = 0.5*fastAtan2(Fy_gauss.at<float>(m, n), Fx_gauss.at<float>(m, n))*CV_PI / 180;
			if ((m%blockSize) == 0 && (n%blockSize) == 0) {
				int x = n;
				int y = m;
				int ln = sqrt(2 * pow(blockSize, 2)) / 2;
				float dx = ln * cos(smoothed.at<float>(m, n) - CV_PI / 2);
				float dy = ln * sin(smoothed.at<float>(m, n) - CV_PI / 2);
				arrowedLine(fprintWithDirectionsSmoo, Point(x, y + blockH), Point(x + dx, y + blockW + dy), Scalar::all(255), 1, LINE_AA, 0, 0.06*blockSize);
			}
		}
	}///for2
	normalize(orientationMap, orientationMap, 0, 1, NORM_MINMAX);
//	imshow("Orientation field", orientationMap);

//	orientationMap = smoothed.clone();

	normalize(smoothed, smoothed, 0, 1, NORM_MINMAX);

	//imshow("Smoothed orientation field", smoothed);
	//imshow("Coherence", coherence);
	// imshow("Orientation", fprintWithDirectionsSmoo);
	pair<Mat,Mat> returning = { fprintWithDirectionsSmoo, orientationMap };
	return returning;
}

#endif