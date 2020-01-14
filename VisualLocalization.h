#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "ParameterTuning.h"
#include "Descriptors/DesriptorFromFile.h"
#include "Descriptors/GroundTruth.h"
#include <memory>

class VisualLocalization
{
public:
	VisualLocalization(const GlobalConfig& config);

	bool readDatabase();

	bool readQuery();

	bool readGroundTruth();

	bool localizeQuery();

	bool getGlobalSearch();

	void getPerformance();

	// ����Ӿ���λ���׶ν��
	std::vector<int> getRetrievalRet(){return retrievalRet;};
	cv::Vec2i getGeomValRet(){return geomValRet;};

	
private:
	std::shared_ptr<GlobalConfig> configPtr;

	// ѵ��������(�����¼��·��)
	std::shared_ptr<DescriptorFromFile> featurebase;
	std::vector<std::vector<std::pair<double, int>>> gps;

	// ���Լ�����
	std::shared_ptr<PartialDescriptorsFromFile> featurequery;
	bool withGPS;
	std::string descriptor;
	
	// Use FLANN to retrieve KNNs
	bool getTopKRetrieval(const int& k, const float& GPSthresh = 15);

	// ʹ�ü�����֤������ƥ����
	void getBestGeomValid();

	// ʹ������ƥ�������ն�λ���
	void getSeqMatch();



	std::shared_ptr<Parameter2F1> seqSLAMPtr;

	// Return the matched Key points after ratio test
	std::vector<cv::DMatch> matchFrameToFrameFlann(const cv::Mat &mDspts1, const cv::Mat &mDspts2);
	// Use H-matrix to filter top-k candidate
	int verifyHmatrix(const std::vector<cv::DMatch>&, const std::vector<cv::Point2f>& mKpts1, 
		const std::vector<cv::Point2f>& mKpts2);
	int verifyEmatrix(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point2f>& mKpts1, 
		const std::vector<cv::Point2f>& mKpts2, cv::Matx33d& R, cv::Vec3d& t);

	bool getDistanceMatrix(const float& gnssTh);

	// The results of image retrieval
	std::vector<int> retrievalRet;
	// The results of geometric validation
	cv::Vec2i geomValRet;

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

	double focal;
	cv::Point2d principalPoint;
};

#endif