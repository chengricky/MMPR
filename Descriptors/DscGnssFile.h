#ifndef PICGNSSFILE
#define PICGNSSFILE

#include "../Header.h"
#include <fstream>
#include <string>

// Data for the single frame
struct DataFrame
{
	cv::Mat netVLAD;
	std::vector<cv::Point2f> mKPts; // TODO: combine 2d and 3D points?
	cv::Mat mDecs;
	cv::Mat ptNorm;
	double latitudeValue;
	double longitudeValue;
};

/// This class restore the data of the current frame, and paths for all of the frames
class PicGNSSFile
{
public:
	PicGNSSFile(){};
	PicGNSSFile(std::string filepath, bool ifGNSS, float cx, float cy, float fx, float fy, int interval=1)
	{collectFileVec(filepath, ifGNSS, cx, cy, fx, fy, interval);};
	// Initialization (collect file paths of images, features and GNSS data)
	void collectFileVec(std::string filepath, bool ifGNSS, float cx, float cy, float fx, float fy, int interval=1);
	// Read the next frame data
	bool getNextFrame();

	DataFrame frame;

	// 返回文件夹内的信息
	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };
	double getFocal(){return (fx+fy)/2;};
	cv::Point2d getPrincipalPoint(){return cv::Point2d(cx, cy);};

private:
	// 存储文件夹内所有图像、特征、GNSS数据的vectors
	std::vector<double> latitude;
	std::vector<double> longitude;
	unsigned long filePointer;
	unsigned long fileVolume;
	std::vector<std::string> featFilesRGB;
	std::vector<std::string> featFilesIR;
	std::vector<std::string> imgFilesDepth;
	float cx, cy, fx, fy;

	// Generate 3D key points from a depth image
	bool generate3DPts(int gridSize=4);

	// Read GNSS.txt File
	bool readTxtGnssLabel(bool ifGNSS, std::string filepath);
	std::vector<std::string> splitWithStl(const std::string &str, const std::string &pattern);
};

#endif
