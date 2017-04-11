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

	vector<Point2f> selectedPointsVector;

	for (int i = 0; i < 12; i++) {
		selectedPointsVector.push_back(Point2f(selectedPoints[i]));
	}

	vector<KeyPoint> selectedKeypointsVector;
	KeyPoint::convert(selectedPointsVector, selectedKeypointsVector);

	string firstImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\face_001.png";
	Mat firstImage = imread(firstImageLocation);
	Mat firstImageGrey;
	cvtColor(firstImage, firstImageGrey, CV_BGR2GRAY);

	Mat firstDescriptorsArray, secondDescriptorsArray;	
	Ptr<FeatureDetector> featureDescriptionComputer = ORB::create();
	//featureDescriptionComputer->detect(firstImage, selectedKeypointsVector);
	//KeyPointsFilter::retainBest(selectedKeypointsVector, 72);
	featureDescriptionComputer->compute(firstImageGrey, selectedKeypointsVector, firstDescriptorsArray);

	string secondImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\face_002.png";
	Mat secondImage = imread(secondImageLocation);
	Mat secondImageGrey;
	cvtColor(secondImage, secondImageGrey, CV_BGR2GRAY);
	vector<KeyPoint> secondImageKeyPointsVector;
	
	featureDescriptionComputer->detect(secondImageGrey, secondImageKeyPointsVector);
	//KeyPointsFilter::retainBest(secondImageKeyPointsVector, 72);
	featureDescriptionComputer->compute(secondImageGrey, secondImageKeyPointsVector, secondDescriptorsArray);


	/*drawKeypoints(secondImage, secondImageKeyPointsVector, secondImage);
	imshow("second image with detected keypoints", secondImage);
	waitKey(0);*/

	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> matches;
	descriptorMatcher->match(firstDescriptorsArray, secondDescriptorsArray, matches);

	Mat outputImage;

	drawMatches(firstImageGrey, selectedKeypointsVector, secondImageGrey, secondImageKeyPointsVector, matches, outputImage);
	imwrite("C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\Matched\\face1_face2_GREY_ORB_matched.png", outputImage);
	imshow("output image", outputImage);
	waitKey(0);

	/*cout << "Descriptors array: " << descriptorsArray << endl;

	cin.ignore();*/

	return EXIT_SUCCESS;
}