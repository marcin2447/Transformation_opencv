
#include<iostream>
#include <sstream>
#include<opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;

Mat obrazDopasowania;
void transformuj(Mat &obraz1, Mat &obraz2, Mat &obrazWynikowy, Mat &macierzTransformacji)
{
	Mat obraz1_Gray, obraz2_Gray;
	cvtColor(obraz1, obraz1_Gray, CV_BGR2GRAY);
	cvtColor(obraz2, obraz2_Gray, CV_BGR2GRAY);

	std::vector<KeyPoint> punktyKluczowe1, punktyKluczowe2;
	Mat deskryptory1, deskryptory2;

	int maksymalnaLiczbaDopasowan = 2000;
	Ptr<Feature2D> orb = ORB::create(maksymalnaLiczbaDopasowan);
	orb->detectAndCompute(obraz1_Gray, Mat(), punktyKluczowe1, deskryptory1);
	orb->detectAndCompute(obraz2_Gray, Mat(), punktyKluczowe2, deskryptory2);

	vector<DMatch> dopasowania;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
	matcher->match(deskryptory1, deskryptory2, dopasowania, Mat());

	sort(dopasowania.begin(), dopasowania.end());

	float procentDobrychDopasowan = 0.17f;
	int liczbaDobrychDopasowan = dopasowania.size() * procentDobrychDopasowan;
	dopasowania.erase(dopasowania.begin() + liczbaDobrychDopasowan, dopasowania.end());

	drawMatches(obraz1, punktyKluczowe1, obraz2, punktyKluczowe2, dopasowania, obrazDopasowania);
	imwrite("dopasowania.jpg", obrazDopasowania);

	vector<Point2f> punkty1, punkty2;

	for (int i = 0; i < dopasowania.size(); i++)
	{
		punkty1.push_back(punktyKluczowe1[dopasowania[i].queryIdx].pt);
		punkty2.push_back(punktyKluczowe2[dopasowania[i].trainIdx].pt);
	}

	macierzTransformacji = findHomography(punkty1, punkty2, RANSAC);
	warpPerspective(obraz1, obrazWynikowy, macierzTransformacji, obraz2.size());
}




int main(int argc, char **argv)
{
	string nazwa1 = "img1.jpg";
	cout << "Obraz1: " << nazwa1<<endl;
	Mat obraz1 = imread(nazwa1);

	string nazwa2("img2.jpg");
	cout << "Obraz2: " << nazwa2 << endl;
	Mat obraz2 = imread(nazwa2);

	Mat obrazWynikowy, macierzTransformacji;

	transformuj(obraz2, obraz1, obrazWynikowy, macierzTransformacji);


	imwrite("result.jpg", obrazWynikowy);
	cout << "Macierz transformacji:\n" << macierzTransformacji << endl;

	imshow("Obraz1", obraz1);
	imshow("Obraz2", obraz2);
	imshow("Wynik", obrazWynikowy);
	imshow("Dopasowania", obrazDopasowania);
	waitKey(0);
	system("pause");
	return 0;
}