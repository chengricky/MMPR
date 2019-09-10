#pragma once

#include "../Header.h"
#include "LDB\ldb.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"

// command pattern - eliminating decoupling
class ImgDescriptorExtractor
{
protected:
	int imgIdx; //the index of input RGBD-(IR)-(D) image group	
	cv::Mat result;
public:		
	ImgDescriptorExtractor(int imgIdx) : imgIdx(imgIdx), result() {};
	virtual bool extract(std::vector<cv::Mat>  img) = 0;
	cv::Mat getResult() { return result; };
	int getImgIdx() { return imgIdx; };
};

class OCVExtractor : public ImgDescriptorExtractor
{
protected:
	cv::Ptr<cv::FeatureDetector> detector; // Feature2D=FeatureDetector
	std::vector<cv::KeyPoint> keypoints;
public:
	OCVExtractor(int imgIdx) : ImgDescriptorExtractor(imgIdx) {};
	bool extract(std::vector<cv::Mat> img);
	std::vector<cv::KeyPoint> getKeypoints() { return keypoints; };
};

#ifdef USE_CONTRIB
class SURFExtractor : public OCVExtractor
{	
public:
	SURFExtractor(int imgIdx);
};
#endif

class ORBExtractor : public OCVExtractor
{
public:
	ORBExtractor(int imgIdx);
	auto getORB() { return detector; };
};

class GISTExtractor : public ImgDescriptorExtractor
{
	bool isNormalize;
	// GISTŒ¨∂»º∆À„=sum(orients)*blocks*blocks*nPics
	cls::GISTParams GIST_PARAMS;
public:
	GISTExtractor(int imgIdx, bool useColor, bool isNormalize, cv::Size imgSize ) ;
	bool extract(std::vector<cv::Mat> todoImages);
};

class CSExtractor : public ImgDescriptorExtractor
{
	cv::Size imgSize;
public:
	CSExtractor(int imgIdx, cv::Size imgSize);
	bool extract(std::vector<cv::Mat> todoImages);
};

class LDBExtractor : public ImgDescriptorExtractor
{
	bool useColor;
	cv::Mat illumination_conversion(cv::Mat image);
public:
	LDBExtractor(int imgIdx, bool useColor);
	bool extract(std::vector<cv::Mat> todoImages);
};

class GoogLeNetExtractor : public ImgDescriptorExtractor
{
	cv::dnn::Net net;
	std::vector<int> idxes;
public:
	GoogLeNetExtractor(int imgIdx);
	bool extract(std::vector<cv::Mat> todoImages);
};

class Extraction
{
	std::vector<ImgDescriptorExtractor*> extractions;
public:
	virtual ~Extraction();
	void add(ImgDescriptorExtractor* c) { extractions.push_back(c); }
	void release() { for (auto p : extractions) delete p; }
	void run(std::vector<cv::Mat> const&  todoImages);
	ImgDescriptorExtractor* getResult(int idx);
	int getSize() { return extractions.size(); };
};