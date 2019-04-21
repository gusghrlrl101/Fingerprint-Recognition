#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>


using namespace std;
using namespace cv;


struct Minutiae {
	int x;
	int y;
	int angle;
	int type; //ending:1  bifurcation:2
};

vector<Minutiae> findMinutiae(Mat& img, Mat& seg) {
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);


	Mat area;
	seg.convertTo(area, CV_8UC1);
	Mat mask = getStructuringElement(1, Size(3, 3), Point(1, 1));
	dilate(area, area, mask, Point(-1, -1), 7);

	int ending = 0;
	int bifurcation = 0;
	vector<Minutiae> mVector;
	Minutiae minutiae;

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *a, *b, *c;    // north (pAbove)
	uchar *d, *e, *f;
	uchar *g, *h, *i;    // south (pBelow)

	uchar *pDst;

	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		b = &(pAbove[0]);
		c = &(pAbove[1]);
		e = &(pCurr[0]);
		f = &(pCurr[1]);
		h = &(pBelow[0]);
		i = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			a = b;   b = c;   c = &(pAbove[x + 1]);
			d = e;   e = f;   f = &(pCurr[x + 1]);
			g = h;   h = i;   i = &(pBelow[x + 1]);

			int sum = *a + *b + *c + *d + *f + *g + *h + *i;
			int xor_ = (*a ^ *b) + (*b ^ *c) + (*c ^ *f) + (*f ^ *i) + (*d ^ *g) + (*g ^ *h) + (*h ^ *i) + (*d ^ *a);
			int and_ = (*a & *b) + (*b & *c) + (*c & *f) + (*f & *i) + (*d & *g) + (*g & *h) + (*h & *i) + (*d & *a);

			int thr = 5;
			if (*e == 1 && (sum == 1)) {
				uchar* segVal = &(area.ptr<uchar>(y))[x];
				if (*segVal == 0) {
					bool isAlready = false;
					for (auto mnt : mVector) {
						int distt = abs(mnt.x - x) + abs(mnt.y - y);
						if (distt <= thr) {
							isAlready = true;
							break;
						}
					}

					if (!isAlready) {
						//					if (0 <= x - 3 && x + 3 < img.cols && 0 <= y - 3 && y + 3 < img.rows) {
						ending++;
						minutiae.x = x; minutiae.y = y;
						minutiae.type = 1;
						mVector.push_back(minutiae);
						//					}
					}
				}
			}

			if (*e == 1 && (xor_ == 6 || (xor_ == 6 && and_ == 2))) {
				uchar* segVal = &(area.ptr<uchar>(y))[x];
				if (*segVal == 0) {
					bool isAlready = false;
					for (auto mnt : mVector) {
						int distt = abs(mnt.x - x) + abs(mnt.y - y);
						if (distt <= thr) {
							isAlready = true;
							break;
						}
					}

					if (!isAlready) {
						//					if (0 <= x - 3 && x + 3 < img.cols && 0 <= y - 3 && y + 3 < img.rows) {
						bifurcation++;
						minutiae.x = x; minutiae.y = y;
						minutiae.type = 2;
						mVector.push_back(minutiae);
						//					}
					}
				}
			}

		}
	}
	cout << "ending: " << ending << ", bifurcation: " << bifurcation << endl;

	return mVector;
}


float angle(Mat& dst, vector<pair<float, float>>& vec, int& u, int& v, int& block_size, Size size, int& type) {
	float fi = 0.0;

	int val = size.width / block_size;
	int width = u / block_size;
	int height = v / block_size;
	cout << "angle: " << height << ", " << width << endl;

	fi = - atan2f(vec[height * val + width].second, vec[height * val + width].first) * 180.0f / CV_PI;

	// end�� ���
	if (type == 1) {
		// ���� ������ ������� ���
		if (fi > 0) {
			// ���� �� ���� ������ ���⿡ ���� ���� �ִ� ��� 180�� �ٲ���
			if (dst.at<uchar>(v, u - 1) == 1 || dst.at<uchar>(v + 1, u - 1 ) == 1 || dst.at<uchar>(v + 1, u) == 1)
				fi -= 180;
		}
		// ���� ������ �������� ���
		else if (fi < 0) {
			// ���� �� ���� �� ���� ���⿡ ���� ���� �ִ� ��� 180�� �ٲ���
			if (dst.at<uchar>(v, u - 1) == 1 || dst.at<uchar>(v - 1, u - 1) == 1 || dst.at<uchar>(v - 1, u) == 1)
				fi += 180;
		}
	}
	// bifar�� ���
	else if (type == 2) {
		// �̹��� ũ�⸸ŭ �湮 ����� 2���� ���� ����
		vector<vector<bool>> visit(size.width, vector<bool>(size.height, false));
		visit[u][v] = true;
		int dir = 0;

		// ���� �� ���� 3���� ���� ����
		pair<int, int> index[3] = { { -1,-1 }, {-1,-1}, {-1,-1} };

		// t = 0 �϶� �����¿� ���� ã��, 3�� �� ��ã���� �밢�� ���� ã��
		for (int t = 0; t < 2; t++) {
			for (int i = -1; i <= +1; i++) {
				for (int j = -1; j <= +1; j++) {
					if (t == 1 || t == 0 && i*j == 0) {
						if (0 <= u + i && u + i < size.width && 0 <= v + j && v + j < size.height &&
							!visit[u + i][v + j] && dst.at<uchar>(v + j, u + i) == 1) {
							if (dir < 3) {
								index[dir++] = { u + i, v + j };
								visit[u + i][v + j] = true;
							}
						}
					}
				}
			}
		}

		// �̵��� Ƚ��
		int count = 10;
		for (int tt = 0; tt < count; tt++) {
			for (int dir = 0; dir < 3; dir++) {
				bool isFinish = false;
				// t = 0 �϶� �����¿� ���� Ž��
				for (int t = 0; t < 2; t++) {
					// 8���� Ž��
					for (int i = -1; i <= +1; i++) {
						for (int j = -1; j <= +1; j++) {
							if (t == 1 || t == 0 && i * j == 0) {
								int next_i = index[dir].first + i;
								int next_j = index[dir].second + j;

								// ���� �̵��� ������ �ְ� ���� �ִ� ��� �̵�
								if (0 <= next_i && next_i < size.width &&
									0 <= next_j && next_j < size.height &&
									!visit[next_i][next_j] &&
									dst.at<uchar>(next_j, next_i) == 1) {
									index[dir] = { next_i, next_j };
									visit[next_i][next_j] = true;

									isFinish = true;
									break;
								}
							}
						}
						if (isFinish)
							break;
					}
					if (isFinish)
						break;
				}
			}
		}

		// 3 �� �� 2���� ��� ������ ���� ���� ���� ����
		int min_theta = 361;
		int min_one = -1, min_two = -1;
		for (int i = 0; i < 3; i++) {
			int one = -1, two = -1;
			// 2���� ����
			for (int j = 0; j < 3; j++) {
				if (i == j)
					continue;
				if (one == -1)
					one = j;
				else
					two = j;
			}
			// 2���� ���͸� ����
			int v1x = index[one].first - u;
			int v1y = index[one].second - v;
			int v2x = index[two].first - u;
			int v2y = index[two].second - v;

			// �� ������ ������ ����
			int inner = v1x * v2x + v1y * v2y;
			// �� ������ ũ�⸦ ����
			float v1_size = sqrt(pow(v1x, 2) + pow(v1y, 2));
			float v2_size = sqrt(pow(v2x, 2) + pow(v2y, 2));

			// �� ������ ���� ���� ����
			int theta = acosf(inner / (v1_size*v2_size)) * 180.0f / CV_PI;

			// �ּҰ����� ������ ����
			if (min_theta > theta) {
				min_theta = theta;
				min_one = one;
				min_two = two;
			}
		}

		// ���� ���� ������ ���� �κ����� �߰� ���� ����
		float mid_x = (index[min_one].first + index[min_two].first) / 2.0f;
		float mid_y = (index[min_one].second + index[min_two].second) / 2.0f;

		// ���� ������ �߰� �������� ���͸� ����
		float vx = mid_x - u;
		float vy = mid_y - v;

		// ������ x���� ������ ���, ������ 180�� �ٲ������
		if (vx < 0) {
			if (fi > 0)
				fi -= 180;
			else if (fi < 0)
				fi += 180;
		}
	}

	return fi;
}


Mat printMinutiae(Mat src, Mat& srcc, vector<pair<float, float>>& vec, int& block_size, Size size, Mat& original) {
	Mat temp;
	Mat dst = src.clone();
	dst /= 255;         // convert to binary image

	Mat dst2 = original.clone();
	cvtColor(dst2, dst2, COLOR_GRAY2BGR);

	vector<Minutiae> mVector = findMinutiae(dst, srcc);


	for (int i = 0; i < mVector.size(); i++)
		mVector[i].angle = angle(dst, vec, mVector[i].x, mVector[i].y, block_size, size, mVector[i].type);

	dst *= 255;
	cvtColor(dst, dst, COLOR_GRAY2RGB);
	threshold(dst, dst, 127, 255, THRESH_BINARY_INV);

	Scalar end = Scalar(255, 120, 000);
	Scalar bif = Scalar(000, 200, 200);

	cvtColor(original, original, COLOR_GRAY2BGR);
	for (int i = 0; i < mVector.size(); i++) {
		if (mVector[i].type == 1) {
			circle(dst2, Point(mVector[i].x, mVector[i].y), 5, end, 1, 8);
			line(dst2, { mVector[i].x, mVector[i].y },
				{ mVector[i].x + (int)(10.0f * cos(-mVector[i].angle * CV_PI / 180.0f)), mVector[i].y + int(10.0f * sin(-mVector[i].angle * CV_PI / 180.0f)) }
			, end);
		}
		else if (mVector[i].type == 2) {
			rectangle(dst2, Point(mVector[i].x - 4, mVector[i].y - 4), Point(mVector[i].x + 4, mVector[i].y + 4), bif, 1);
			line(dst2, { mVector[i].x, mVector[i].y },
				{ mVector[i].x + (int)(10.0f * cos(-mVector[i].angle * CV_PI / 180.0f)), mVector[i].y + int(10.0f * sin(-mVector[i].angle* CV_PI / 180.0f)) }
			, bif);
		}

//		Mat temp_dst_up;
//		pyrUp(dst, temp_dst_up);
//		imshow("drawing", temp_dst_up);
//		waitKey(0);
	}

	return dst2;
}