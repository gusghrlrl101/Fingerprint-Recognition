#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

//TODO : Real World로의 거리 표현 필요
//TODO : Pt1과 Pt2의 위치에 따라서 분기 필요

int distance(Mat& src, Point& pt1, Point& pt2) {
	//cv::cvtColor(src, src, COLOR_BGR2GRAY); // Gray Version
	//threshold(src, src, 127, 255, THRESH_BINARY); // 임시 threshold

	//Point pt1(120, 100), pt2(300, 500); // 최초 두 점 정의

	int distanceX = 0, distanceY = 0; // X좌표 간의 거리, Y 좌표 간의 거리
	double distance = 0, indiDistance = 0; // 두 점 간의 거리, 융선 간의 거리
	vector<Point> line1; // 점 사이의 좌표들을 저장하는 벡터
	line1.push_back(pt1);

	//거리계산
	distanceX = abs(pt1.x - pt2.x);
	distanceY = abs(pt1.y - pt2.y);
	distance = sqrt(distanceX*distanceX + distanceY * distanceY);

	bool chk = false; // 홀짝 계산용 flag
	Point temp;
	if (distanceX >= distanceY) { // X간의 거리가 더 넓은지, Y 간의 거리가 더 넓은지에 따라 분기
		double rate = (double)(distanceX / distanceY); // 기울기
		while (line1[line1.size() - 1].x != pt2.x&&line1[line1.size() - 1].y != pt2.y) {
			//번갈아가면서 더해줘야 맞는 직선이 나옴
			if (chk) {
				temp = { (int)(line1[line1.size() - 1].x + rate + 1), line1[line1.size() - 1].y + 1 };
				chk = false;
			}
			else {
				temp = { (int)(line1[line1.size() - 1].x + rate), line1[line1.size() - 1].y + 1 };
				chk = true;
			}

			line1.push_back(temp);
		}
	}
	else {
		double rate = (double)(distanceY / distanceX); // 기울기
		while (line1[line1.size() - 1].x != pt2.x&&line1[line1.size() - 1].y != pt2.y) {
			//번갈아가면서 더해줘야 맞는 직선이 나옴
			if (chk) {
				temp = { line1[line1.size() - 1].x + 1,  (int)(line1[line1.size() - 1].y + rate + 1) };
				chk = false;
			}
			else {
				temp = { line1[line1.size() - 1].x + 1,  (int)(line1[line1.size() - 1].y + rate) };
				chk = true;
			}
			line1.push_back(temp);
		}
	}
	int count = 0; // 융선 수
	for (int i = 0; i < line1.size() - 1; i++) {
		if (src.at<uchar>(line1[i].y, line1[i].x) >= 127 && src.at<uchar>(line1[i + 1].y, line1[i + 1].x) < 127) { // 흰색에서 검은색 될 때
			count++;
		}
	}
	cout << "융선 수 :  " << count << endl;
	cout << "총 거리 : " << distance << endl;
	indiDistance = distance / (double)count;
	cout << "융선 간 거리 :  " << indiDistance << endl;

	return indiDistance;
}