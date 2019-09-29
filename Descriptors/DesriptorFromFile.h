#ifndef DESCRIPTORFROMFILE
#define DESCRIPTORFROMFILE

#include <vector>
#include "../Descriptors/PicGnssFile.h"
#include "../Descriptors/GlobalConfig.h"

class DescriptorFromFile
{
public:
	DescriptorFromFile(GlobalConfig& config, bool isRefImage);
	virtual ~DescriptorFromFile() {};

	PicGNSSFile picFiles;
	std::string picPath;
	cv::Mat GPS;
	cv::Mat netVLADs;
	std::vector<cv::Mat> descs;
	std::vector<cv::vector<cv::KeyPoint>> kpts;

	int getVolume() { return picFiles.getFileVolume(); };

};
#endif