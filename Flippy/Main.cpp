#include <iostream>
using namespace std;

#include <opencv2\opencv.hpp>
#include <opencv2\features2d.hpp>
using namespace cv;

int main()
{
	/*Mat identity = Mat::eye(3, 3, CV_32FC1);

	cout << "Identity: " << identity << endl;
	cin.ignore();*/

	/*
	Pixel locations of hand-selected feature points in face_001.png (all points specified from face's perspective) (all written as 'x, y')
	right eye outer corner - 243, 590
	right eye inner corner - 306, 584
	left eye inner corner - 391, 584
	left eye outer corner - 462, 582
	right eyebrow outer end - 204, 569
	right eyebrow inner end - 300, 545
	left eyebrow inner end - 400, 545
	left eyebrow outer end - 490, 549
	right mouth corner - 296, 772
	left mouth corner - 421, 768
	right nostril centre - 322, 704
	left nostril centre - 376, 697

	*/

	const Point selectedPoints [] = { Point(243, 590), Point(306, 584), Point(391, 584), Point(462, 582), Point(204, 569), Point(300, 545),
		Point(400, 545), Point(490, 549), Point(296, 772), Point(421, 768), Point(322, 704), Point(376, 697)
	};

	vector<Point2f> firstImagePointsVector, secondImagePointsVector;

	for (int i = 0; i < 12; i++) {
		firstImagePointsVector.push_back(Point2f(selectedPoints[i]));
	}

	string firstImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\face_001.png";
	Mat firstImage = imread(firstImageLocation);
	//Mat firstImageGrey;
	//cvtColor(firstImage, firstImageGrey, CV_BGR2GRAY);

	for (int picNum = 2; picNum < 3; picNum++) {

		string secondImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\face_003.png";
		Mat secondImage = imread(secondImageLocation);
		//Mat secondImageGrey;
		//cvtColor(secondImage, secondImageGrey, CV_BGR2GRAY);
		vector<uchar> status;
		vector<float> err;
		Mat secondImageCopy;
		secondImageCopy = secondImage.clone();

		calcOpticalFlowPyrLK(firstImage, secondImage, firstImagePointsVector, secondImagePointsVector, status, err);

		/*for (auto & i : secondImagePointsVector) {
			circle(secondImageCopy, i, 3, Scalar(0.0, 255.0, 0.0, 255.0));
		}*/

		for (size_t i = 0; i != secondImagePointsVector.size(); i++) {
			if (status[i] == 1) {
				circle(secondImageCopy, secondImagePointsVector[i], 3, Scalar(0.0, 255.0, 0.0, 255.0));
			}
		}

		imwrite("C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\Matched\\face_003.png", secondImageCopy);

		secondImagePointsVector.swap(firstImagePointsVector);
		secondImage.copyTo(firstImage);
	}

	//imshow("Optical flow test", secondImageCopy);
	//waitKey(0);

	return EXIT_SUCCESS;
}