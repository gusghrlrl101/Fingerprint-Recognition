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

	int distanceX = 0, distanceY = 0; // X��ǥ ���� �Ÿ�, Y ��ǥ ���� �Ÿ�
	double distance = 0, indiDistance = 0; // �� �� ���� �Ÿ�, ���� ���� �Ÿ�
	vector<Point> line1; // �� ������ ��ǥ���� �����ϴ� ����
	line1.push_back(pt1); // ���� ����� push

	//�Ÿ����
	distanceX = abs(pt1.x - pt2.x); // x ��ǥ �� �����Ÿ�
	distanceY = abs(pt1.y - pt2.y); // y ��ǥ �� �����Ÿ�
	distance = sqrt(distanceX*distanceX + distanceY * distanceY); // ��Ŭ���� �Ÿ�

	bool chk = false; // Ȧ¦ ���� flag
	Point temp;
	if (distanceX >= distanceY) { // X���� �Ÿ��� �� ������, Y ���� �Ÿ��� �� �������� ���� �б�
		double rate = (double)(distanceX / (distanceY + DBL_EPSILON)); // ����
		while (line1[line1.size() - 1].x != pt2.x && line1[line1.size() - 1].y != pt2.y) { // ������ ������
			//�����ư��鼭 ������� �´� ������ ����
			if (pt1.y <= pt2.y) {
				if (line1[line1.size() - 1].y >= pt2.y) // y�� ���� �ʰ��ع����� ����
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
				if (line1[line1.size() - 1].y <= pt2.y) // y�� ���� �ʰ��ع����� ����
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
		double rate = (double)(distanceY / (distanceX + DBL_EPSILON)); // ����
		while (line1[line1.size() - 1].x != pt2.x&&line1[line1.size() - 1].y != pt2.y) {
			//�����ư��鼭 ������� �´� ������ ����
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
	int count = 0; // ���� ��
	for (int i = 0; i < line1.size() - 1; i++) {
		if (src.at<uchar>(line1[i].y, line1[i].x) >= 127 && src.at<uchar>(line1[i + 1].y, line1[i + 1].x) < 127) { // ������� ������ �� ��
			count++;
		}
	}
	// ���� ���� 3���� �۴ٸ� ���ǹ��� ������ �Ǵ�
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

	// ending point�� bifurcation ���� �Ÿ��� ����
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
			if (temp1.x <= temp2.x) // x�� ������ ���� �� ��° ���ڿ� �־� �ش�.
				distanceN = distance(imgt, temp1, temp2);
			else
				distanceN = distance(imgt, temp2, temp1);

			// ������ 3���� ���� �� 0�� ��ȯ�ǹǷ� �׷��� ���� ���� count ����
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