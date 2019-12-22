#ifndef DESCRIPTORFROMFILE
#define DESCRIPTORFROMFILE

#include <vector>
#include "FeatReader.h"
#include "GlobalConfig.h"

class DescriptorFromFile
{
public:
	DescriptorFromFile(GlobalConfig& config, bool isRefImage);
	virtual ~DescriptorFromFile() {};

	std::shared_ptr<FeatReader> pFeatReader;
	std::string picPath;
	cv::Mat netVLADs;
	std::vector<std::vector<int>> clss;

	int getVolume() { return pFeatReader->getFileVolume(); };

};
#endif