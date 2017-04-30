#pragma once

void detectFace(cv::Mat &image, std::stringstream &imagePath, cv::Mat &imageGray, cv::CascadeClassifier &face_cascade, std::vector<cv::Rect> &faceRects, cv::CascadeClassifier &eye_cascade, std::vector<cv::Rect> &eyeRects, std::stringstream &saveLocation);
