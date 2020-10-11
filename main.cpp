// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;
void show(Mat& mat) {
	auto windowname = "Test";
	namedWindow(windowname, 0);
	resizeWindow(windowname, 500, 500);
	imshow(windowname, mat);
	waitKey();
}

static void help(const char* programName) {
	cout <<
	     "\nA program using pyramid scaling, Canny, contours and contour simplification\n"
	     "to find squares in a list of images (pic1-6.png)\n"
	     "Returns sequence of squares detected on the image.\n"
	     "Call:\n"
	     "./" << programName << " [file_name (optional)]\n"
	                            "Using OpenCV version " << CV_VERSION << "\n" << endl;
}
int thresh = 20, N = 11;
const char* wndname = "Square Detection Demo";
// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( const Point& pt1, const Point& pt2, const Point& pt0 ) {
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

static vector<Point2f> orderPoints(const vector<Point2f>& points) {
	vector<Point2f> orderedPoints(4);
	auto sumPoint = [](const Point& pointA, const Point& pointB) {
		return (pointA.x + pointA.y) <= (pointB.x + pointB.y);
	};

	const auto [minSum, maxSum] = minmax_element(points.begin(), points.end(), sumPoint);

	orderedPoints[0] = *minSum;
	orderedPoints[2] = *maxSum;

	auto diffPoint = [](const Point& pointA, const Point& pointB) {
		return (pointA.x - pointA.y) <= (pointB.x - pointB.y);
	};

	const auto [minDiff, maxDiff] = minmax_element(points.begin(), points.end(), diffPoint);

	orderedPoints[3] = *minDiff;
	orderedPoints[1] = *maxDiff;

	return orderedPoints;
}

static void fourPointTransform(Mat& image, const vector<Point2f>& points) {
	auto orderedPoints = orderPoints(points);

	auto widthA = norm(orderedPoints[2] - orderedPoints[3]);
	auto widthB = norm(orderedPoints[1] - orderedPoints[0]);
	auto maxWidth = max(int(widthA), int(widthB));

	auto heightA = norm(orderedPoints[1] - orderedPoints[2]);
	auto heightB = norm(orderedPoints[0] - orderedPoints[3]);
	auto maxHeight = max(int(heightA), int(heightB));

	vector<Point2f> newPositions({
		Point2f(0, 0),
		Point2f(maxWidth - 1, 0),
		Point2f(maxWidth - 1, maxHeight - 1),
		Point2f(0, maxHeight - 1)
	});

	auto M = getPerspectiveTransform(orderedPoints, newPositions);
	warpPerspective(image, image, M, Size(maxWidth, maxHeight));
}

// returns sequence of squares detected on the image.
static void findSquares( const Mat& image, vector<vector<Point> >& squares ) {
	squares.clear();
	Mat blurred(image);
	medianBlur(image, blurred, 9);

	Mat gray0(blurred.size(), CV_8U), gray;

	vector<vector<Point> > contours;
	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++) {
		int ch[] = {c, 0};
		mixChannels(&blurred, 1, &gray0, 1, ch, 1);

		while(squares.empty() && N < 100) {
			// try several threshold levels
			for (int l = 0; l < N; l++) {
				// hack: use Canny instead of zero threshold level.
				// Canny helps to catch squares with gradient shading
				if (l == 0) {
					// apply Canny. Take the upper threshold from slider
					// and set the lower to 0 (which forces edges merging)
					Canny(gray0, gray, 10, thresh, 3);
					// dilate canny output to remove potential
					// holes between edge segments
					dilate(gray, gray, Mat(), Point(-1, -1));
				} else {
					// apply threshold if l!=0:
					//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
					gray = gray0 >= (l + 1) * 255 / N;
				}

				// find contours and store them all as a list
				findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

				show(gray);

				vector<Point> approx;
				// test each contour
				for (auto &contour : contours) {
					// approximate contour with accuracy proportional
					// to the contour perimeter
					approxPolyDP(contour, approx, arcLength(contour, true) * 0.02, true);
					// square contours should have 4 vertices after approximation
					// relatively large area (to filter out noisy contours)
					// and be convex.
					// Note: absolute value of an area is used because
					// area may be positive or negative - in accordance with the
					// contour orientation
					if (approx.size() == 4 &&
					    fabs(contourArea(approx)) > 1000 &&
					    isContourConvex(approx)) {
						double maxCosine = 0;
						for (int j = 2; j < 5; j++) {
							// find the maximum cosine of the angle between joint edges
							double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
							maxCosine = MAX(maxCosine, cosine);
						}
						// if cosines of all angles are small
						// (all angles are ~90 degree) then write quandrange
						// vertices to resultant sequence
						if (maxCosine < 0.2)
							squares.push_back(approx);
					}
				}
			}

			N += 10;
		}
	}
}
int main(int argc, char** argv) {
	vector<string> names;
	//names.emplace_back("assets/0.jpg");
	//names.emplace_back("assets/1.jpg");
	//names.emplace_back("assets/2.jpg");
	//names.emplace_back("assets/3.jpg");
	names.emplace_back("assets/4.jpg");
	//names.emplace_back("assets/5.jpg");
	//names.emplace_back("assets/6.jpg");
	//names.emplace_back("assets/7.jpg");
	//names.emplace_back("assets/8.jpg");
	//names.emplace_back("assets/9.jpg");
	//names.emplace_back("assets/10.jpg");
	//names.emplace_back("assets/11.jpg");
	//names.emplace_back("assets/12.jpg");
	//names.emplace_back("assets/13.jpg");
	//names.emplace_back("assets/14.jpg");
	//names.emplace_back("assets/15.jpg");
	//names.emplace_back("assets/16.jpg");
	//names.emplace_back("assets/17.jpg");
	//names.emplace_back("assets/18.jpg");
	//names.emplace_back("assets/19.jpg");
	//names.emplace_back("assets/20.jpg");
	//names.emplace_back("assets/21.jpg");

	help(argv[0]);
	if (argc > 1) {
		names.emplace_back(argv[1]);
	}
	for (auto& name : names) {
		string filename = samples::findFile(name);
		Mat image = imread(filename, IMREAD_COLOR);
		if (image.empty()) {
			cout << "Couldn't load " << filename << endl;
			continue;
		}
		vector<vector<Point> > squares;
		findSquares(image.clone(), squares);

		sort(squares.begin(), squares.end(), [](const vector<Point>& a, const vector<Point>& b){
			return contourArea(a) > contourArea(b);
		});

		Mat drawn = image.clone();
		polylines(drawn, squares, true, Scalar(0, 255, 0), 3, LINE_AA);
		namedWindow(wndname, 0);
		resizeWindow(wndname, 250, 250);
		imshow(wndname, drawn);

		if(false) {
			int number = 0;
			for (auto square : squares) {
				if (number > 1) break;

				auto wndname1 = "Square Cut:" + to_string(number++);

				Mat cut = image.clone();
				fourPointTransform(cut, vector<Point2f>(square.begin(), square.end()));
				namedWindow(wndname1, 0);
				resizeWindow(wndname1, 250, 250);
				imshow(wndname1, cut);
			}
		}

		int c = waitKey();
		if (c == 27)
			break;
	}

	return 0;
}