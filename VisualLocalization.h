#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "Descriptors/DesriptorFromFile.h"
#include "Descriptors/GroundTruth.h"
#include "Frame.h"

class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);
	virtual ~VisualLocalization();

	// 使用flann检索knn结果
	std::vector<std::vector<int>> getTopKRetrieval(const int& k);
	// 使用
	int MatchFrameToFrameFlann(const Frame &frame1, const Frame &frame2);

	void getBestMatch_FeatureFile();
	bool getDistanceMatrix(float GNSS=30);
	bool getEnhancedDistanceMatrix(int winSize);
	bool getGlobalSearch();


	
private:
	// 训练集数据(保存记录的路径)
	DescriptorFromFile* featurebase;
	std::vector<std::vector<std::pair<double, int>>> gps;

	// 测试集数据
	DescriptorFromFile* featurequery;
	std::vector<bool> keyGT, keyPredict, keyGPS;
	bool withGPS;
	std::string descriptor;

	/// get a distance matrix, which is as follows:	
	cv::Mat GPSDistance;
	cv::Mat GPSMask_uchar;

	cv::Mat netVLAD_Distance;
	
	
	cv::Ptr<cv::flann::Index> searchDB; 


	//   ----> database
	//  |
	//  |
	//  V
	//  query images
	cv::Mat enhanceMatrix(const cv::Mat& distanceMat, int winSize);
	int matRow, matCol;


	// 最佳匹配结果
	std::vector<int> GPSGlobalBest;
	std::vector<int> netVLADglobalResult;

	GroundTruth ground;
};

// 计算二进制描述符的汉明距离
int hamming_matching(cv::Mat desc1, cv::Mat desc2);

#endif