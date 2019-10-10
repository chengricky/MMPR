#ifndef PICGNSSFILE
#define PICGNSSFILE

#include "../Header.h"
#include <fstream>
#include <string>

class PicGNSSFile
{
public:
	// Initialization (collect file paths of images, features and GNSS data)
	void collectFileVec(std::string filepath, bool ifGNSS, float cx, float cy, float fx, float fy, int interval=1);
	// Read the next frame data
	bool getNextFrame();
	// 存储单帧数据
	cv::Mat netVLAD;
	std::vector<cv::Point2f> mKPts;
	cv::Mat mDecs;
	cv::Mat ptNorm;
	double latitudeValue;
	double longitudeValue;
	// 返回文件夹内的信息
	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };

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

	// Read GNSS.txt File
	bool readTxtGnssLabel(bool ifGNSS, std::string filepath);
	std::vector<std::string> splitWithStl(const std::string &str, const std::string &pattern);
};

#endif
