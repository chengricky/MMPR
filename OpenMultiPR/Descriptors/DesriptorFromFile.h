#pragma once
#include <vector>
#include "..\FileInterface\PicGnssFile.h"
#include "..\FileInterface\GlobalConfig.h"

class DescriptorFromFile
{
public:
	DescriptorFromFile(GlobalConfig& config, bool isRefImage);
	virtual ~DescriptorFromFile() {};

	PicGNSSFile picFiles;
	std::string picPath;
	cv::Mat GPS;
	cv::Mat netVLADs;

	int getVolume() { return picFiles.getFileVolume(); };

};
