#include <iostream>
#include <iomanip>
using namespace std;

#include <opencv2\opencv.hpp>
#include <opencv2\features2d.hpp>
using namespace cv;

#define NUM_OF_IMAGES 455

void detectFaces() {
	Mat image, imageGray;
	stringstream imagePath, saveLocation;
	auto face_cascade = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	auto eye_cascade = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml");
	vector<Rect> faceRects, eyeRects;

	for (int picNum = 1; picNum <= NUM_OF_IMAGES; picNum++) {
		imagePath << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\BaseImages\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";
		image = imread(imagePath.str());
		imagePath = stringstream();
		cvtColor(image, imageGray, CV_BGR2GRAY);

		face_cascade.detectMultiScale(imageGray, faceRects);

		for (auto & face : faceRects) {
			Mat faceMat = image(face);
			eye_cascade.detectMultiScale(faceMat, eyeRects);
			rectangle(image, face, Scalar(255.0, 0.0, 0.0, 255.0), 3);
			for (auto & eye : eyeRects) {
				eye.x += face.x;
				eye.y += face.y;
				rectangle(image, eye, Scalar(0.0, 255.0, 0.0, 255.0), 3);
			}
		}



		saveLocation << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\FaceDetection1\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";
		imwrite(saveLocation.str(), image);
		saveLocation = stringstream();

		//imshow("Detected face?", image);
		//waitKey(0);
	}

}

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

	//const Point selectedPoints [] = { Point(243, 590), Point(306, 584), Point(391, 584), Point(462, 582), Point(204, 569), Point(300, 545),
	//	Point(400, 545), Point(490, 549), Point(296, 772), Point(421, 768), Point(322, 704), Point(376, 697)
	//};

	//vector<Point2f> firstImagePointsVector, secondImagePointsVector;

	//for (int i = 0; i < 12; i++) {
	//	firstImagePointsVector.push_back(Point2f(selectedPoints[i]));
	//}

	//string firstImageLocation = "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\BaseImages\\face_001.png";
	//Mat firstImage = imread(firstImageLocation);
	//Mat secondImage;
	//Mat secondImageCopy;
	//vector<uchar> status;
	//vector<float> err;

	//for (int picNum = 2; picNum < 456; picNum++) {

	//	stringstream secondImageLocation;
	//	secondImageLocation << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\BaseImages\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";

	//	secondImage = imread(secondImageLocation.str());
	//	
	//	secondImageCopy = secondImage.clone();
	//	calcOpticalFlowPyrLK(firstImage, secondImage, firstImagePointsVector, secondImagePointsVector, status, err);

	//	size_t pointsVectorSize = secondImagePointsVector.size();

	//	for (size_t i = 0; i < pointsVectorSize; i++) {
	//		if (status[i] == 1) {
	//			circle(secondImageCopy, secondImagePointsVector[i], 3, Scalar(0.0, 255.0, 0.0, 255.0));
	//		}
	//		else {
	//			//cout << "PicNum: " << picNum << ".  Status of " << i << " is " << status[i] << endl;
	//			firstImagePointsVector.erase(firstImagePointsVector.begin() + i);
	//			secondImagePointsVector.erase(secondImagePointsVector.begin() + i);
	//			pointsVectorSize--;
	//		}
	//	}

	//	stringstream saveLocation;
	//	saveLocation << "C:\\Misc\\James\\UoA Contract\\Flippy\\Images\\Tracking2\\" << "face_" << setw(3) << setfill('0') << picNum << ".png";

	//	imwrite(saveLocation.str(), secondImageCopy);

	//	secondImagePointsVector.swap(firstImagePointsVector);
	//	secondImage.copyTo(firstImage);
	//}

	//imshow("Optical flow test", secondImageCopy);
	//waitKey(0);

	//cin.ignore();

	detectFaces();

	return EXIT_SUCCESS;
}