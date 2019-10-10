#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "Descriptors/DesriptorFromFile.h"
#include "Descriptors/GroundTruth.h"

class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);

	// ʹ��flann����knn���
	void getTopKRetrieval(const int& k);

	// ʹ�ü�����֤������ƥ����
	void getBestGeomValid();

	// ʹ������ƥ�������ն�λ���
	void getSeqMatch();

	bool getGlobalSearch();

	// ����Ӿ���λ���׶ν��
	std::vector<std::vector<int>> getRetrievalRet(){return retrievalRet;};
	std::vector<cv::Vec2i> getGeomValRet(){return geomValRet;};

	
private:
	// ѵ��������(�����¼��·��)
	std::shared_ptr<DescriptorFromFile> featurebase;
	std::vector<std::vector<std::pair<double, int>>> gps;

	// ���Լ�����
	std::shared_ptr<DescriptorFromFile> featurequery;
	bool withGPS;
	std::string descriptor;

	// ʹ��H����ɸѡtopk
	int MatchFrameToFrameFlann(const cv::Mat &mDspts1, const std::vector<cv::Point2f>& mKpts1,
								const cv::Mat &mDspts2, const std::vector<cv::Point2f>& mKpts2);
	bool generateVideo(std::vector<int> matchingResults, std::string path="");
	bool getDistanceMatrix(const float& gnssTh);

	// ������������֤�Ľ��
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

	// ���ƥ����
	std::vector<int> GPSGlobalBest;
	std::vector<int> netVLADglobalResult;

	GroundTruth ground;
};

#endif