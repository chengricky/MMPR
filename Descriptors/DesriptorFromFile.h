#ifndef DESCRIPTORFROMFILE
#define DESCRIPTORFROMFILE

#include <vector>
#include "../Descriptors/DscGnssFile.h"
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
	std::vector<std::vector<cv::Point2f>> kpts;

	int getVolume() { return picFiles.getFileVolume(); };

};
#endif