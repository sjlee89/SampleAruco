#include <vector>
#include <iostream>
#include <ctime>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2\viz.hpp>
#include <opencv2/viz/viz3d.hpp>

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

using namespace std;
using namespace cv;

int CreateMarker(String name) 
{
	int dictionaryId = 15;
	int markerId = 0;
	int borderBits = 1;
	int markerSize = 200;

	bool showImage = 1;	
	
	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	Mat markerImg;
	aruco::drawMarker(dictionary, markerId, markerSize, markerImg, borderBits);

	if (showImage) {
		imshow("marker", markerImg);
		waitKey(0);
	}

	imwrite(name, markerImg);

	return 0;
}

void CreateCharuco(String name)
{
	int squaresX = 5;
	int squaresY = 7;
	int squareLength = 80;
	int markerLength = 40;
	int dictionaryId = 10;
	int margins = 10;
	int borderBits = 1;

	bool showImage = 1;		

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	Size imageSize;
	imageSize.width = squaresX * squareLength + 2 * margins;
	imageSize.height = squaresY * squareLength + 2 * margins;

	Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(squaresX, squaresY, (float)squareLength,
		(float)markerLength, dictionary);

	// show created board
	Mat boardImage;
	board->draw(imageSize, boardImage, margins, borderBits);

	if (showImage) {
		imshow("board", boardImage);
		waitKey(0);
	}

	imwrite(name, boardImage);
}

static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

static bool saveCameraParams(const string &filename, Size imageSize, float aspectRatio, int flags,
	const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	time_t tt;
	time(&tt);
	struct tm *t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;

	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;

	if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

	if (flags != 0) {
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;

	return true;
}

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

int CamCalib(string outputFile)
{
	int squaresX = 5;
	int squaresY = 7;
	float squareLength = 80;
	float markerLength = 40;
	int dictionaryId = 10;

	bool showChessboardCorners = 1;

	int calibrationFlags = 0;
	float aspectRatio = 1;

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

	bool refindStrategy = 1;
	int camId = 1;
	String video;

	VideoCapture inputVideo;
	int waitTime;
	if (!video.empty()) {
		inputVideo.open(video);
		waitTime = 0;
	}
	else {
		inputVideo.open(camId);
		waitTime = 10;
	}

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	// create charuco board object
	Ptr<aruco::CharucoBoard> charucoboard =
		aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
	Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

	// collect data from each frame
	vector< vector< vector< Point2f > > > allCorners;
	vector< vector< int > > allIds;
	vector< Mat > allImgs;
	Size imgSize;

	while (inputVideo.grab()) {
		Mat image, imageCopy;
		inputVideo.retrieve(image);

		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;

		// detect markers
		aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

		// refind strategy to detect more markers
		if (refindStrategy) aruco::refineDetectedMarkers(image, board, corners, ids, rejected);

		// interpolate charuco corners
		Mat currentCharucoCorners, currentCharucoIds;
		if (ids.size() > 0)
			aruco::interpolateCornersCharuco(corners, ids, image, charucoboard, currentCharucoCorners,
				currentCharucoIds);

		// draw results
		image.copyTo(imageCopy);
		if (ids.size() > 0) aruco::drawDetectedMarkers(imageCopy, corners);

		if (currentCharucoCorners.total() > 0)
			aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);

		putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
			Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

		imshow("out", imageCopy);
		char key = (char)waitKey(waitTime);
		if (key == 27) break;
		if (key == 'c' && ids.size() > 0) {
			cout << "Frame captured" << endl;
			allCorners.push_back(corners);
			allIds.push_back(ids);
			allImgs.push_back(image);
			imgSize = image.size();
		}
	}

	if (allIds.size() < 1) {
		cerr << "Not enough captures for calibration" << endl;
		return 0;
	}

	Mat cameraMatrix, distCoeffs;
	vector< Mat > rvecs, tvecs;
	double repError;

	if (calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrix = Mat::eye(3, 3, CV_64F);
		cameraMatrix.at< double >(0, 0) = aspectRatio;
	}

	// prepare data for calibration
	vector< vector< Point2f > > allCornersConcatenated;
	vector< int > allIdsConcatenated;
	vector< int > markerCounterPerFrame;
	markerCounterPerFrame.reserve(allCorners.size());
	for (unsigned int i = 0; i < allCorners.size(); i++) {
		markerCounterPerFrame.push_back((int)allCorners[i].size());
		for (unsigned int j = 0; j < allCorners[i].size(); j++) {
			allCornersConcatenated.push_back(allCorners[i][j]);
			allIdsConcatenated.push_back(allIds[i][j]);
		}
	}

	// calibrate camera using aruco markers
	double arucoRepErr;
	arucoRepErr = aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
		markerCounterPerFrame, board, imgSize, cameraMatrix,
		distCoeffs, noArray(), noArray(), calibrationFlags);

	// prepare data for charuco calibration
	int nFrames = (int)allCorners.size();
	vector< Mat > allCharucoCorners;
	vector< Mat > allCharucoIds;
	vector< Mat > filteredImages;
	allCharucoCorners.reserve(nFrames);
	allCharucoIds.reserve(nFrames);

	for (int i = 0; i < nFrames; i++) {
		// interpolate using camera parameters
		Mat currentCharucoCorners, currentCharucoIds;
		aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoboard,
			currentCharucoCorners, currentCharucoIds, cameraMatrix,
			distCoeffs);

		allCharucoCorners.push_back(currentCharucoCorners);
		allCharucoIds.push_back(currentCharucoIds);
		filteredImages.push_back(allImgs[i]);
	}

	if (allCharucoCorners.size() < 4) {
		cerr << "Not enough corners for calibration" << endl;
		return 0;
	}

	// calibrate camera using charuco
	repError =
		aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize,
			cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

	bool saveOk = saveCameraParams(outputFile, imgSize, aspectRatio, calibrationFlags,
		cameraMatrix, distCoeffs, repError);
	if (!saveOk) {
		cerr << "Cannot save output file" << endl;
		return 0;
	}

	cout << "Rep Error: " << repError << endl;
	cout << "Rep Error Aruco: " << arucoRepErr << endl;
	cout << "Calibration saved to " << outputFile << endl;

	// show interpolated charuco corners for debugging
	if (showChessboardCorners) {
		for (unsigned int frame = 0; frame < filteredImages.size(); frame++) {
			Mat imageCopy = filteredImages[frame].clone();
			if (allIds[frame].size() > 0) {

				if (allCharucoCorners[frame].total() > 0) {
					aruco::drawDetectedCornersCharuco(imageCopy, allCharucoCorners[frame],
						allCharucoIds[frame]);
				}
			}

			imshow("out", imageCopy);
			char key = (char)waitKey(0);
			if (key == 27) break;
		}
	}

	return 0;
}

int DetectMarker(string CamCalbFile)
{
	int dictionaryId = 15;
	bool showRejected = 1;
	bool estimatePose = 1;
	float markerLength = 40;

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

	//override cornerRefinementMethod read from config file
	detectorParams->cornerRefinementMethod = 2;
	
	std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;

	int camId = 1;

	String video;

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
	
	Mat camMatrix, distCoeffs;
	if (estimatePose) {
		bool readOk = readCameraParameters(CamCalbFile, camMatrix, distCoeffs);
		if (!readOk) {
			cerr << "Invalid camera file" << endl;
			return 0;
		}
	}

	VideoCapture inputVideo;
	int waitTime;
	if (!video.empty()) {
		inputVideo.open(video);
		waitTime = 0;
	}
	else {
		inputVideo.open(camId);
		waitTime = 10;
	}

	double totalTime = 0;
	int totalIterations = 0;

	while (inputVideo.grab()) {
		Mat image, imageCopy;
		inputVideo.retrieve(image);

		double tick = (double)getTickCount();

		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;
		vector< Vec3d > rvecs, tvecs;

		// detect markers and estimate pose
		aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
		if (estimatePose && ids.size() > 0)
			aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
				tvecs);

		double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
		totalTime += currentTime;
		totalIterations++;
		if (totalIterations % 30 == 0) {
			cout << "Detection Time = " << currentTime * 1000 << " ms "
				<< "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
		}

		// draw results
		image.copyTo(imageCopy);
		if (ids.size() > 0) {
			aruco::drawDetectedMarkers(imageCopy, corners, ids);
			if (estimatePose) {
				for (unsigned int i = 0; i < ids.size(); i++)
					aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
						markerLength * 0.5f);
			}
		}	


		if (showRejected && rejected.size() > 0)
			aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

		imshow("out", imageCopy);
		char key = (char)waitKey(waitTime);
		if (key == 27) break;
	}

	return 0;
}

bool exists(const char *fname)
{
	FILE *file;
	if ((file = fopen(fname, "r")))
	{
		fclose(file);
		return true;
	}

	return false;
}

int TestViz(string CamCalbFile)
{
	int dictionaryId = 15;
	bool showRejected = 1;
	bool estimatePose = 1;
	float markerLength = 40;
	int camId = 1;

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

	//override cornerRefinementMethod read from config file
	detectorParams->cornerRefinementMethod = 2;

	std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;
	
	String video;

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	Mat camMatrix, distCoeffs;
	if (estimatePose) {
		bool readOk = readCameraParameters(CamCalbFile, camMatrix, distCoeffs);
		if (!readOk) {
			cerr << "Invalid camera file" << endl;
			return 0;
		}
	}
	
	viz::Viz3d ARWindow("SampleAR");

	cv::Mat image;
	
	// Webcam frame pose, without this frame is upside-down
	Affine3f imagePose(Vec3f(3.14159, 0, 0), Vec3f(0, 0, 0));

	// Viz viewer pose to see whole webcam frame
	Vec3f cam_pos(0.0f, 0.0f, 900.0f), cam_focal_point(0.0f, 0.0f, 0.0f), cam_y_dir(0.0f, 0.0f, 0.0f);
	Affine3f viewerPose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

	// Video capture from source
	VideoCapture cap(camId);
	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	cap >> image;

	// Initialization
	viz::WMesh mesh(viz::Mesh::load("./Data/MX.ply"));
	viz::WImage3D img(image, Size2d(frame_width, frame_height));
	ARWindow.setWindowSize(Size(640, 480));
	ARWindow.setBackgroundColor(viz::Color::black());
	ARWindow.setViewerPose(viewerPose);
	ARWindow.showWidget("Image", img);
	ARWindow.showWidget("Mx", mesh);
	Affine3f initmeshpose(Vec3f(0, 0, 0), Vec3f(0, 0, 0));
	mesh.setPose(initmeshpose);

	// Rotation vector of 3D model
	Mat rot_vec = Mat::zeros(1, 3, CV_32F);
	vector<Vec3d> rvecs, tvecs;
	Vec3d rvec, tvec;

	ARWindow.spinOnce(1, true);
	while (!ARWindow.wasStopped()) {
		if (cap.read(image)) {
			cv::Mat image, imageCopy;
			cap.retrieve(image);
			image.copyTo(imageCopy);

			// Marker detection
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(image, dictionary, corners, ids);

			if (ids.size() > 0) {

				// Draw a green line around markers
				cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
				

				// Get rotation and translation vectors of each markers
				cv::aruco::estimatePoseSingleMarkers(corners, 0.05, camMatrix, distCoeffs, rvecs, tvecs);

				for (int i = 0; i < ids.size(); i++) {
					cv::aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
					
					// Augment 3D object on the first marker
					rvec = rvecs[0];
					rot_vec.at<float>(0, 0) = rvec[0];
					rot_vec.at<float>(0, 1) = rvec[1];
					rot_vec.at<float>(0, 2) = rvec[2];

					tvec = tvecs[0];				
				}								
			}

			// Show camera frame in Viz window
			img.setImage(imageCopy);
			img.setPose(imagePose);	

			// Mesh pose calculation
			Mat rmat;
			Rodrigues(rot_vec, rmat);

			// CV to GL Coord.
			Mat viewMatrix = Mat::zeros(4, 4, CV_32F);
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					viewMatrix.at<float>(row, col) = rmat.at<float>(row, col);
				}
				viewMatrix.at<float>(row, 3) = tvec[row];
			}
			viewMatrix.at<float>(3, 3) = 1.0f;

			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
			cvToGl.at<float>(0, 0) = 1.0f;
			cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis
			cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis
			cvToGl.at<float>(3, 3) = 1.0f;
			viewMatrix = cvToGl * viewMatrix;	

			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					rmat.at<float>(row, col) = viewMatrix.at<float>(row, col);
				}
			}

			//Affine3f MeshPose(rmat, Vec3f(tvec[0], tvec[1], tvec[2]));
			Affine3f MeshPose(rmat, Vec3f(viewMatrix.at<float>(0, 3), viewMatrix.at<float>(1, 3), viewMatrix.at<float>(2, 3)));

			// Set the pose of 3D model
			mesh.setPose(MeshPose);
			
			ARWindow.spinOnce(1, true);
		}
	}	

	return 0;
}

int main()
{
	// Create marker
	String m_sMName = "./Markers/SampleMarker3.png";	
	if(!exists(m_sMName.c_str()))
		CreateMarker(m_sMName);

	// Create checkerboard for camera calibration
	String m_sBName = "./Calib/SampleBoard.png";
	if (!exists(m_sMName.c_str()))
		CreateMarker(m_sBName);

	// Camera Calibration
	string m_sCamCal = "./Calib/Camcal.txt";
	if (!exists(m_sCamCal.c_str()))
		CamCalib(m_sCamCal);

	// Marker detection
	//DetectMarker(m_sCamCal);

	// Marker detection & Mesh augmentation
	TestViz(m_sCamCal);

	return 0;

}