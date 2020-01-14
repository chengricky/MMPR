#ifndef DESCRIPTORFROMFILE
#define DESCRIPTORFROMFILE

#include <vector>
#include "../Descriptors/DscGnssFile.h"
#include "../Tools/GlobalConfig.h"

class DescriptorFromFile
{
public:
	DescriptorFromFile(const GlobalConfig& config, bool isRefImage);
	virtual ~DescriptorFromFile() {};

	PicGNSSFile picFiles;
	std::string picPath;
	cv::Mat GPS;
	cv::Mat netVLADs;
	std::vector<cv::Mat> descs;	
	std::vector<std::vector<cv::Point2f>> kpts;
	std::vector<cv::Mat> pt3dNorms; // 注意与kpt是不匹配的 --> 改为不去掉无效深度点后是否已经匹配？

	int getVolume() { return picFiles.getFileVolume(); };

};

// Datareader for query sequence
class PartialDescriptorsFromFile
{
public:
	PartialDescriptorsFromFile(const GlobalConfig& config);
	virtual ~PartialDescriptorsFromFile() {};

	std::shared_ptr<PicGNSSFile> picFiles;
	int queryVolume;
	std::string picPath;
	std::vector<cv::Mat> GPS;
	std::vector<cv::Mat> netVLADs;
	std::vector<cv::Mat> descs;	
	std::vector<std::vector<cv::Point2f>> kpts;
	std::vector<cv::Mat> pt3dNorms; // 注意与kpt是不匹配的

	// get a new frame from the query sequence (return true, if has)
	bool getFrame();
	int queryIdx;

};

#endif