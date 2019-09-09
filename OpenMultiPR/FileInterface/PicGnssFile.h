#pragma once
#include "FileInterface.h"
#include <fstream>
#include <string>

class PicGNSSFile : public FileInterface
{
public:
	PicGNSSFile() {};
	PicGNSSFile(std::string filepath, int mode, bool ifGNSS, std::string fileKeyWord);
	void init(std::string filepath, int mode, bool ifGNSS, std::string fileKeyWord, int interval=1);
	void init(std::string filepath, int mode, bool ifGNSS) {};
	bool doMain();
	bool doMainFeatureFile();
	cv::Size getImgSize();

	cv::Mat colorImg;
	cv::Mat depthImg;
	cv::Mat IRImg;
	cv::Mat netVLAD;
	double latitudeValue;
	double longitudeValue;
	bool readVideo() { return false; }
	std::string getColorImgPath(int imgIdx){ return colorFiles[imgIdx]; };
	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };

	const static int RGB = 1;
	const static int RGBD = 2;
	const static int RGBDIR = 3;//带IR图像

private:
	int mode;
	std::vector<std::string> depthFiles;
	std::vector<std::string> colorFiles;
	std::vector<std::string> IRFiles;
	std::vector<double> latitude; //纬度
	std::vector<double> longitude; //经度
	unsigned long filePointer;
	unsigned long fileVolume;
	
	void findFilesfromColor(std::string path, std::string prefix, std::string suffix, bool& depth, bool&ground);

	// Read GNSS.txt File
	bool readTxtGnssLabel(bool ifGNSS, std::string filepath);

};