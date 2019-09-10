#pragma once
#include <vector>
#include "..\FileInterface\PicGnssFile.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"
#include "LDB\ldb.h"
#include "..\FileInterface\GlobalConfig.h"
#include "DescriptorExtraction.h"

class Descriptors
{
public:
	Descriptors(GlobalConfig& config, bool isRefImage);
	virtual ~Descriptors() {};

	PicGNSSFile picFiles;
	std::string picPath;
	cv::Mat CS;
	cv::Mat GG;
	// GIST_RGB, GIST_D, GIST_IR;
	cv::Mat GIST[3];
	// LDB_RGB, LDB_D, LDB_IR;
	cv::Mat LDB[3];
	std::vector<cv::Mat> ORB_RGB, /*ORB_D,*/ ORB_IR;
	cv::Mat GPS;

	int getVolume() { return picFiles.getFileVolume(); };

private:
	std::vector<cv::Mat> getAllImage(const PicGNSSFile& picsRec, const cv::Size& imgSize);

	bool isColor;
	Extraction extraction;
};