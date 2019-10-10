#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "Descriptors/DesriptorFromFile.h"
#include "Descriptors/GroundTruth.h"

class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);

	// 使用flann检索knn结果
	void getTopKRetrieval(const int& k);

	// 使用几何验证获得最佳匹配结果
	void getBestGeomValid();

	// 使用序列匹配获得最终定位结果
	void getSeqMatch();

	bool getGlobalSearch();

	// 获得视觉定位各阶段结果
	std::vector<std::vector<int>> getRetrievalRet(){return retrievalRet;};
	std::vector<cv::Vec2i> getGeomValRet(){return geomValRet;};

	
private:
	// 训练集数据(保存记录的路径)
	std::shared_ptr<DescriptorFromFile> featurebase;
	std::vector<std::vector<std::pair<double, int>>> gps;

	// 测试集数据
	std::shared_ptr<DescriptorFromFile> featurequery;
	bool withGPS;
	std::string descriptor;

	// 使用H矩阵筛选topk
	int MatchFrameToFrameFlann(const cv::Mat &mDspts1, const std::vector<cv::Point2f>& mKpts1,
								const cv::Mat &mDspts2, const std::vector<cv::Point2f>& mKpts2);
	bool generateVideo(std::vector<int> matchingResults, std::string path="");
	bool getDistanceMatrix(const float& gnssTh);

	// 检索、几何验证的结果
	std::vector<std::vector<int>> retrievalRet;
	std::vector<cv::Vec2i> geomValRet;

	/// get a distance matrix, which is as follows:	
	//   ----> database
	//  |
	//  |
	//  V
	//  query images	
	int matRow, matCol;
	cv::Mat GPSDistance;
	cv::Mat GPSMask_uchar;
	
	std::shared_ptr<cv::flann::Index> searchDB; 

	// 最佳匹配结果
	std::vector<int> GPSGlobalBest;
	std::vector<int> netVLADglobalResult;

	GroundTruth ground;
};

#endif