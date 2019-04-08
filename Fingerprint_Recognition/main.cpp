#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main() {
	Mat src = imread("image/etc/1.bmp");
	imshow("SRC", src);
	waitKey(0);

	return 0;
}