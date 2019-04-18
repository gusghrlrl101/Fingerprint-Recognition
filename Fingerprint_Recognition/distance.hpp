#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <math.h>


using namespace cv;
using namespace std;

//TODO : Real World���� �Ÿ� ǥ�� �ʿ�
//TODO : Pt1�� Pt2�� ��ġ�� ���� �б� �ʿ�

int distance(Mat& src, Point& pt1, Point& pt2) {
	//cv::cvtColor(src, src, COLOR_BGR2GRAY); // Gray Version
	//threshold(src, src, 127, 255, THRESH_BINARY); // �ӽ� threshold

	//Point pt1(120, 100), pt2(300, 500); // ���� �� �� ����

	int distanceX = 0, distanceY = 0; // X��ǥ ���� �Ÿ�, Y ��ǥ ���� �Ÿ�
	double distance = 0, indiDistance = 0; // �� �� ���� �Ÿ�, ���� ���� �Ÿ�
	vector<Point> line1; // �� ������ ��ǥ���� �����ϴ� ����
	line1.push_back(pt1);

	//�Ÿ����
	distanceX = abs(pt1.x - pt2.x);
	distanceY = abs(pt1.y - pt2.y);
	distance = sqrt(distanceX*distanceX + distanceY * distanceY);

	bool chk = false; // Ȧ¦ ���� flag
	Point temp;
	if (distanceX >= distanceY) { // X���� �Ÿ��� �� ������, Y ���� �Ÿ��� �� �������� ���� �б�
		double rate = (double)(distanceX / distanceY); // ����
		while (line1[line1.size() - 1].x != pt2.x&&line1[line1.size() - 1].y != pt2.y) {
			//�����ư��鼭 ������� �´� ������ ����
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
		double rate = (double)(distanceY / distanceX); // ����
		while (line1[line1.size() - 1].x != pt2.x&&line1[line1.size() - 1].y != pt2.y) {
			//�����ư��鼭 ������� �´� ������ ����
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
	int count = 0; // ���� ��
	for (int i = 0; i < line1.size() - 1; i++) {
		if (src.at<uchar>(line1[i].y, line1[i].x) >= 127 && src.at<uchar>(line1[i + 1].y, line1[i + 1].x) < 127) { // ������� ������ �� ��
			count++;
		}
	}
	cout << "���� �� :  " << count << endl;
	cout << "�� �Ÿ� : " << distance << endl;
	indiDistance = distance / (double)count;
	cout << "���� �� �Ÿ� :  " << indiDistance << endl;

	return indiDistance;
}


void calculate(Mat imgt, Mat src) {
	imgt /= 255;
	vector<Minutiae> minutiaes = findMinutiae(imgt, src);
	imgt *= 255;

	vector<Point> ending;
	vector<Point> bif;
	vector<Point> core;
	vector<Point> delta;
	int endingN = 0, bifN = 0, coreN = 0, deltaN = 0;
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
			cout << "#" << count << "��°" << endl;
			Point temp1 = { ending[i].x, ending[i].y };
			Point temp2 = { core[j].x, core[j].y };
			if (temp1.x <= temp2.x)
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
}