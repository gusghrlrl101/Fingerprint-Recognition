#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main() {
	Mat img;
	img = imread("E:\\opencv\\Fingerprint\\Fingerprint-Recognition\\Fingerprint_Recognition\\image\\Team1\\2019_1_1_L_I_1.bmp");

	imshow("first", img);
	waitKey(0);

	return 0;
}