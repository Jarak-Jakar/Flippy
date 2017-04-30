#include <iostream>
#include <iomanip>
using namespace std;

#include <opencv2\opencv.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\calib3d.hpp>
#include "Main.h"
#include "stasm_lib.h"
using namespace cv;

#define NUM_OF_IMAGES 455

void detectFace(cv::Mat &image, std::stringstream &imagePath, cv::Mat &imageGray, cv::CascadeClassifier &face_cascade, std::vector<cv::Rect> &faceRects, cv::CascadeClassifier &eye_cascade, std::vector<cv::Rect> &eyeRects, std::stringstream &saveLocation)
{
	image = imread(imagePath.str());
	cvtColor(image, imageGray, CV_BGR2GRAY);

	face_cascade.detectMultiScale(imageGray, faceRects, 1.3, 5);

	for (auto & face : faceRects) {
		Mat faceMat = image(face);
		eye_cascade.detectMultiScale(faceMat, eyeRects, 1.3, 5);
		rectangle(image, face, Scalar(255.0, 0.0, 0.0, 255.0), 3);
		for (auto & eye : eyeRects) {
			eye.x += face.x;
			eye.y += face.y;
			rectangle(image, eye, Scalar(0.0, 255.0, 0.0, 255.0), 3);
		}
	}

	//imwrite(saveLocation.str(), image);
}

void detectFaces() {
	Mat image, imageGray;
	stringstream imagePath, saveLocation;
	auto face_cascade = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	auto eye_cascade = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml");
	vector<Rect> faceRects, eyeRects;

	for (int picNum = 11; picNum <= 11; picNum++) {
		imagePath << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\BaseImages\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";
		saveLocation << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\FaceDetection1\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";

		detectFace(image, imagePath, imageGray, face_cascade, faceRects, eye_cascade, eyeRects, saveLocation);

		imagePath = stringstream();
		saveLocation = stringstream();

		cv::imshow("Detected face?", image);
		cv::waitKey(0);
	}

}

//bool checkForHueWithinThreshold(Mat& imageToSearch, Point& locale, int searchThreshold, uchar hueToFind) {
//	Vec3b pointColor = imageToSearch.at<Vec3b>(locale);
//	if ((pointColor[0] >= (hueToFind - searchThreshold)) && (pointColor[0] <= (hueToFind + searchThreshold))) {
//		return true;
//	}
//	return false;
//}
//
//Point searchForMatchingColorPoint(Mat& imageToSearch, Point& centreLocale, int colorSearchThreshold, uchar hueToFind, int windowRadius) {
//	int X = centreLocale.x;
//	int Y = centreLocale.y;
//
//
//}

int main()
{
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
	vector<Vec3b> pointHues;

	string firstImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\BaseImages\\face_001.png";
	//string firstImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\stasm4.1.0\\data\\testface.jpg";
	Mat firstImage = imread(firstImageLocation);
	//Mat firstImageHSV;
	//Vec3b pointColor;
	//cvtColor(firstImage, firstImageHSV, CV_BGR2HSV);

	for (int i = 0; i < 12; i++) {
		firstImagePointsVector.push_back(Point2f(selectedPoints[i]));
		//pointColor = firstImageHSV.at<Vec3b>(firstImagePointsVector[i]);
		//pointHues.push_back(pointColor);
	}


	
	Mat secondImage, secondImageGray;
	Mat secondImageCopy;
	vector<uchar> status;
	vector<float> err;
	auto face_cascade = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	auto eye_cascade = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml");
	vector<Rect> faceRects, eyeRects;

	// Section for testing out stasm

	Mat firstImageGray;
	cvtColor(firstImage, firstImageGray, CV_BGR2GRAY);
	auto stereomatcher = StereoSGBM::create(0, 32, 15);

	/*int foundFace;
	float landmarks[2 * stasm_NLANDMARKS];

	int result = stasm_search_single(&foundFace, landmarks, (char*)firstImageGray.data,
		firstImageGray.cols, firstImageGray.rows, "C:\\Misc\\James\\UoA Contract\\Flippy\\stasm4.1.0\\data\\testface.jpg",
		"C:\\Misc\\James\\UoA Contract\\Flippy\\stasm4.1.0\\data");

	stasm_force_points_into_image(landmarks, firstImageGray.cols, firstImageGray.rows);

	for (int i = 0; i < stasm_NLANDMARKS; i++) {
		firstImageGray.at<float>(Point(cvRound(landmarks[i * 2 + 1]), cvRound(landmarks[i * 2]))) = 255;
	}

	cv::imshow("stasm", firstImageGray);
	cv::waitKey(0);
	return 0;*/

	for (int picNum = 2; picNum <= NUM_OF_IMAGES; picNum++) {

		stringstream secondImageLocation;
		secondImageLocation << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\BaseImages\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";

		secondImage = imread(secondImageLocation.str());
		
		secondImageCopy = secondImage.clone();
		cvtColor(secondImage, secondImageGray, CV_BGR2GRAY);
		calcOpticalFlowPyrLK(firstImage, secondImage, firstImagePointsVector, secondImagePointsVector, status, err);

		Mat fundamentalMat = findFundamentalMat(firstImagePointsVector, secondImagePointsVector);

		Mat homographyMat1, homographyMat2;
		bool result = stereoRectifyUncalibrated(firstImagePointsVector, secondImagePointsVector, fundamentalMat, 
			secondImage.size(), homographyMat1, homographyMat2);

		Mat firstImageRectified = firstImage.clone();
		Mat secondImageRectified = secondImage.clone();
		Mat cameraMatrix = Mat::zeros(3, 3, CV_64F);
		cameraMatrix.at<double>(0, 0) = 0.028 / 0.00000112;
		cameraMatrix.at<double>(1, 1) = 0.028 / 0.00000112;
		cameraMatrix.at<double>(0, 2) = firstImage.rows / 2.0;
		cameraMatrix.at<double>(1, 2) = firstImage.cols / 2.0;
		cameraMatrix.at<double>(2, 2) = 1.0;

		Mat m1c1, m2c1, m1c2, m2c2;
		Mat camInv = cameraMatrix.inv();
		//Mat R1 = camInv.mul(homographyMat1).mul(cameraMatrix);
		Mat R1 = camInv * homographyMat1 * cameraMatrix;
		//Mat R2 = camInv.mul(homographyMat2).mul(cameraMatrix);
		Mat R2 = camInv * homographyMat2 * cameraMatrix;

		Mat distortCoeffs = Mat::zeros(0, 0, CV_64F);

		initUndistortRectifyMap(cameraMatrix, distortCoeffs, R1, cameraMatrix, Size(firstImage.rows, firstImage.cols), 
			CV_32FC1, m1c1, m2c1);
		initUndistortRectifyMap(cameraMatrix, distortCoeffs, R2, cameraMatrix, Size(secondImage.rows, secondImage.cols),
			CV_32FC1, m1c2, m2c2);

		remap(firstImageGray, firstImageRectified, m1c1, m2c1, INTER_LINEAR);
		remap(secondImageGray, secondImageRectified, m1c2, m2c2, INTER_LINEAR);

		Mat disparitymap;

		stereomatcher->compute(firstImageRectified, secondImageRectified, disparitymap);


		//face_cascade.detectMultiScale(secondImageGray, faceRects);

		//Rect faceRect = faceRects.front();

		//size_t pointsVectorSize = secondImagePointsVector.size();

		//for (size_t i = 0; i < pointsVectorSize; i++) {
		//	Point2f pointy = secondImagePointsVector[i];
		//	if (status[i] == 1) {
		//		/*if (pointy.x < faceRect.x) {
		//			pointy.x = faceRect.x;
		//		}
		//		if (pointy.y < faceRect.y) {
		//			pointy.y = faceRect.y;
		//		}
		//		if (pointy.x > (faceRect.x + faceRect.width)) {
		//			pointy.x = faceRect.x + faceRect.width;
		//		}
		//		if (pointy.y > (faceRect.y + faceRect.height)) {
		//			pointy.y = faceRect.y + faceRect.height;
		//		}*/


		//		circle(secondImageCopy, pointy, 3, Scalar(0.0, 255.0, 0.0, 255.0));
		//		//secondImagePointsVector[i] = pointy;
		//	}
		//	else {
		//		//cout << "PicNum: " << picNum << ".  Status of " << i << " is " << status[i] << endl;
		//		firstImagePointsVector.erase(firstImagePointsVector.begin() + i);
		//		secondImagePointsVector.erase(secondImagePointsVector.begin() + i);
		//		pointsVectorSize--;
		//	}
		//}

		stringstream saveLocation;
		saveLocation << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\StereoMatching1\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";

		imwrite(saveLocation.str(), disparitymap);

		secondImagePointsVector.swap(firstImagePointsVector);
		secondImage.copyTo(firstImage);
	}

	cv::imshow("Optical flow test", secondImageCopy);
	cv::waitKey(0);

	//cin.ignore();

	//detectFaces();

	return EXIT_SUCCESS;
}