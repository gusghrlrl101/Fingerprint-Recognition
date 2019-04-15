#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("image/Team1/2019_1_2_L_I_1.bmp");
	imshow("src", src);

	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); 

	Mat imgLaplacian;
	filter2D(src, imgLaplacian, CV_32F, kernel);

	Mat sharp;
	src.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	imshow("imgResult", imgResult);

	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

	//! [bin]
	// Create binary image from source image
	Mat bw;
	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
	threshold(bw, bw, 50, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("Binary Image", bw);

	//! [dist]
	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, DIST_L2, 3);

	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY_INV);

	Mat kernel1 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	imshow("Peaks", dist);

	src.copyTo(dist);
	imshow("final",src);

	waitKey();
	return 0;
}
