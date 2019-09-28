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

	// ʹ��flann����knn���
	std::vector<std::vector<int>> getTopKRetrieval(const int& k);
	// ʹ��
	int MatchFrameToFrameFlann(const Frame &frame1, const Frame &frame2);

	void getBestMatch_FeatureFile();
	bool getDistanceMatrix(float GNSS=30);
	bool getEnhancedDistanceMatrix(int winSize);
	bool getGlobalSearch();


	
private:
	// ѵ��������(�����¼��·��)
	DescriptorFromFile* featurebase;
	std::vector<std::vector<std::pair<double, int>>> gps;

	// ���Լ�����
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


	// ���ƥ����
	std::vector<int> GPSGlobalBest;
	std::vector<int> netVLADglobalResult;

	GroundTruth ground;
};

// ����������������ĺ�������
int hamming_matching(cv::Mat desc1, cv::Mat desc2);

#endif