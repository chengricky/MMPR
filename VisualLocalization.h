#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "Descriptors/DesriptorFromFile.h"
#include "Descriptors/GroundTruth.h"

class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);

	// ʹ��flann����knn���
	std::vector<std::vector<int>> getTopKRetrieval(const int& k);
	void getBestMatch(const std::vector<std::vector<int>>& topk, std::vector<int>& ret);

	bool getGlobalSearch();
	
private:
	// ѵ��������(�����¼��·��)
	std::shared_ptr<DescriptorFromFile> featurebase;
	std::vector<std::vector<std::pair<double, int>>> gps;

	// ���Լ�����
	std::shared_ptr<DescriptorFromFile> featurequery;
	std::vector<bool> keyGT, keyPredict, keyGPS;
	bool withGPS;
	std::string descriptor;

	// ʹ��H����ɸѡtopk
	int MatchFrameToFrameFlann(const cv::Mat &mDspts1, const std::vector<cv::Point2f>& mKpts1,
								const cv::Mat &mDspts2, const std::vector<cv::Point2f>& mKpts2);


	/// get a distance matrix, which is as follows:	
	cv::Mat GPSDistance;
	cv::Mat GPSMask_uchar;

	cv::Mat netVLAD_Distance;
	
	
	std::shared_ptr<cv::flann::Index> searchDB; 


	//   ----> database
	//  |
	//  |
	//  V
	//  query images
	// cv::Mat enhanceMatrix(const cv::Mat& distanceMat, int winSize);
	int matRow, matCol;


	// ���ƥ����
	std::vector<int> GPSGlobalBest;
	std::vector<int> netVLADglobalResult;

	GroundTruth ground;
};

// ����������������ĺ�������
int hamming_matching(cv::Mat desc1, cv::Mat desc2);

#endif