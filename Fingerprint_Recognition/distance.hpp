#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

int distance(Mat& src, Point& pt1, Point& pt2) {

	int distanceX = 0, distanceY = 0; // X좌표 간의 거리, Y 좌표 간의 거리
	double distance = 0, indiDistance = 0; // 두 점 간의 거리, 융선 간의 거리
	vector<Point> line1; // 점 사이의 좌표들을 저장하는 벡터
	line1.push_back(pt1); // 최초 출발점 push

	//거리계산
	distanceX = abs(pt1.x - pt2.x); // x 좌표 간 직선거리
	distanceY = abs(pt1.y - pt2.y); // y 좌표 간 직선거리
	distance = sqrt(distanceX*distanceX + distanceY * distanceY); // 유클리드 거리

	bool chk = false; // 홀짝 계산용 flag
	Point temp;
	if (distanceX >= distanceY) { // X간의 거리가 더 넓은지, Y 간의 거리가 더 넓은지에 따라 분기
		double rate = (double)(distanceX / (distanceY + DBL_EPSILON)); // 기울기
		while (line1[line1.size() - 1].x != pt2.x && line1[line1.size() - 1].y != pt2.y) { // 도달할 때까지
			//번갈아가면서 더해줘야 맞는 직선이 나옴
			if (pt1.y <= pt2.y) {
				if (line1[line1.size() - 1].y >= pt2.y) // y가 만약 초과해버리면 종료
					break;
				if (chk) {
					temp = { (int)(line1[line1.size() - 1].x + rate + 1), line1[line1.size() - 1].y + 1 };
					chk = false;
				}
				else {
					temp = { (int)(line1[line1.size() - 1].x + rate), line1[line1.size() - 1].y + 1 };
					chk = true;
				}
			}
			else {
				if (line1[line1.size() - 1].y <= pt2.y) // y가 만약 초과해버리면 종료
					break;
				if (chk) {
					temp = { (int)(line1[line1.size() - 1].x + rate + 1), line1[line1.size() - 1].y - 1 };
					chk = false;
				}
				else {
					temp = { (int)(line1[line1.size() - 1].x + rate), line1[line1.size() - 1].y - 1 };
					chk = true;
				}
			}
			line1.push_back(temp);
		}
	}
	else {
		double rate = (double)(distanceY / (distanceX + DBL_EPSILON)); // 기울기
		while (line1[line1.size() - 1].x != pt2.x&&line1[line1.size() - 1].y != pt2.y) {
			//번갈아가면서 더해줘야 맞는 직선이 나옴
			if (pt1.y <= pt2.y) {
				if (line1[line1.size() - 1].y >= pt2.y)
					break;
				if (chk) {
					temp = { line1[line1.size() - 1].x + 1,  (int)(line1[line1.size() - 1].y + rate + 1) };
					chk = false;
				}
				else {
					temp = { line1[line1.size() - 1].x + 1,  (int)(line1[line1.size() - 1].y + rate) };
					chk = true;
				}
			}
			else {
				if (line1[line1.size() - 1].y <= pt2.y)
					break;
				if (chk) {
					temp = { line1[line1.size() - 1].x + 1,  (int)(line1[line1.size() - 1].y - rate - 1) };
					chk = false;
				}
				else {
					temp = { line1[line1.size() - 1].x + 1,  (int)(line1[line1.size() - 1].y - rate) };
					chk = true;
				}
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
	// 융선 수가 3보다 작다면 무의미한 결과라고 판단
	if (count < 3)
		return 0;

	indiDistance = distance / (double)count;

	return indiDistance;
}


void calculate(Mat imgt, Mat src) {
	imgt /= 255;
	vector<Minutiae> minutiaes = findMinutiae(imgt, src);
	imgt *= 255;

	vector<Point> ending;
	vector<Point> bif;
	int endingN = 0, bifN = 0, coreN = 0, deltaN = 0;

	// ending point와 bifurcation 간의 거리를 측정
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
	}

	int count = 1;
	int distanceMean = 0, distanceMax = 0, distanceMin = 987654321;
	int distanceN = 0;
	for (int i = 0; i < ending.size(); i++) {
		for (int j = 0; j < bif.size(); j++) {
			Point temp1 = { ending[i].x, ending[i].y };
			Point temp2 = { bif[j].x, bif[j].y };
			if (temp1.x <= temp2.x) // x가 왼쪽인 것을 두 번째 인자에 넣어 준다.
				distanceN = distance(imgt, temp1, temp2);
			else
				distanceN = distance(imgt, temp2, temp1);

			// 융선이 3보다 작을 때 0이 반환되므로 그렇지 않을 때만 count 증가
			if (distanceN > 0)
				count++;
			else
				continue;
			// Max
			if (distanceN > distanceMax)
				distanceMax = distanceN;
			// Min
			if (distanceN < distanceMin && distanceN > 0)
				distanceMin = distanceN;
			// Sum for calculate Mean
			distanceMean += distanceN;
		}
	}

	distanceMean /= count;
	cout << "Max : " << distanceMax << endl;
	cout << "Min : " << distanceMin << endl;
	cout << "Mean : " << distanceMean << endl;
}