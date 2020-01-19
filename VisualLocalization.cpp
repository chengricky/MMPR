#include "VisualLocalization.h"
#include "Tools/Timer.h"
#include <dirent.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
#include "opencv2/surface_matching/icp.hpp"
#include "Descriptors/GroundTruth.h"
#include <iomanip>
#include <numeric>

using namespace std;

#define TEST


VisualLocalization::VisualLocalization(const GlobalConfig& config) : geomValRet(cv::Vec2i()), 
	retrievalRet(std::vector<int>())
{ 	
	if (!config.getValid())
		throw invalid_argument("Configuration is invalid!");
	
	configPtr = std::make_shared<GlobalConfig>(config);

};

bool VisualLocalization::readDatabase()
{
	std::cout << "====>Reading Database Features." << std::endl;
	featurebase = std::make_shared<DescriptorFromFile>(*configPtr, true);	
	matCol = featurebase->getVolume();

	// construct the FLANN database
	std::cout << "====>Constructing the FLANN database." << std::endl;
	cv::flann::KDTreeIndexParams kdIndex(5); //同时建立多个随机kd树，确定tree的数量
	searchDB = make_shared<cv::flann::Index>(featurebase->netVLADs, kdIndex); //默认为L2 distance

	focal = featurebase->picFiles.getFocal();
	principalPoint = featurebase->picFiles.getPrincipalPoint();
}

bool VisualLocalization::readGroundTruth()
{
	std::cout << "====>Reading Ground Truth." << std::endl;
	ground.init(configPtr->pathTest + "/of.txt", configPtr->pathRec + "/of.txt");
	ground.generateGroundTruth(5);

}

bool VisualLocalization::readQuery()
{
	std::cout << "====>Preparing to Load Query Sequence." << std::endl;
	featurequery = std::make_shared<PartialDescriptorsFromFile>(*configPtr);

	seqSLAMPtr = std::make_shared<Parameter2F1>(ground.gt, matCol);
}

bool VisualLocalization::getGlobalSearch()//GPS global best
{
	/// only GPS localization
	cv::Mat GPSdistanceMat = GPSDistance;
	GPSdistanceMat.setTo(FLT_MAX, GPSMask_uchar);
	for (size_t i = 0; i < matRow; i++)//query
	{
		//When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is a single-row or single-column matrix.
		//In OpenCV (following MATLAB) each array has at least 2 dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be (i1,0)/(i2,0)) 
		//and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be (0,j1)/(0,j2)).
		int* minPos = new int[2];
		cv::minMaxIdx(GPSdistanceMat.row(i), nullptr, nullptr, minPos, nullptr);	//可否改成top-k		?
		GPSGlobalBest.push_back(minPos[1]);
		delete minPos;
	}
	return true;
}

bool VisualLocalization::getDistanceMatrix(const float& gnssTh)
{
	// GPS
	auto GPSQuery = featurequery->GPS.back();
	auto GPSRef = featurebase->GPS;
	if (GPSQuery.empty() || GPSRef.empty())
	{
		GPSDistance = cv::Mat();
		GPSMask_uchar = cv::Mat();
	}
	else
	{
		int mat_row = 1;
		GPSDistance = cv::Mat(mat_row, matCol, CV_32FC1);
		for (size_t i = 0; i < mat_row; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<float>(i, j) = GNSSdistance(GPSQuery.at<float>(i,1), GPSQuery.at<float>(i, 0), GPSRef.at<float>(j, 1), GPSRef.at<float>(j, 0));
		cv::Mat GPSuchar;
		GPSDistance.convertTo(GPSuchar, CV_8U);
		cv::threshold(GPSuchar, GPSMask_uchar, gnssTh, 255, cv::THRESH_BINARY_INV);
	}
}

bool VisualLocalization::localizeQuery()
{
	auto has_query = featurequery->getFrame();

	if(!has_query)
		return false;

	getTopKRetrieval(configPtr->topK);
	std::cout<<"==>Top-"<<configPtr->topK<<" Retrieval is Ready."<<std::endl;	
	
	getBestGeomValid();
	std::cout<<"==>Best Match is Obatained from Geometric Validation."<<std::endl;

	seqSLAMPtr->placeRecognition(geomValRet, true);
	std::cout<<"==>Sequential Matching is Ready."<<std::endl;

	return true;
}

bool VisualLocalization::getTopKRetrieval(const int& k, const float& GPSthresh)
{
	auto query = featurequery->netVLADs.back();
	auto ref = featurebase->netVLADs;
	if (!featurequery->GPS.empty() && !featurebase->GPS.empty())
		getDistanceMatrix(GPSthresh);
	
	if(ref.empty()||query.empty())
	{
		retrievalRet = vector<int>();
		return false;
	}		
	else
	{
		cv::Mat idx, dist;
		if(GPSMask_uchar.empty())
		{
			searchDB->knnSearch(query, idx, dist, k, cv::flann::SearchParams());
			retrievalRet = (vector<int>)idx;
		}
		else
		{
			cv::Mat ref_tmp;
			std::vector<int> idx_;
			for(int kk = 0; kk < GPSMask_uchar.cols; kk++)
			{
				if(GPSMask_uchar.at<uchar>(0, kk))
				{
					ref_tmp.push_back(ref.row(kk));
					idx_.push_back(kk);
				}
					
			}
			cv::flann::KDTreeIndexParams kdIndex(5); //同时建立多个随机kd树，确定tree的数量
			searchDB = make_shared<cv::flann::Index>(ref_tmp, kdIndex); //默认为L2 distance
			searchDB->knnSearch(query, idx, dist, k, cv::flann::SearchParams());
			std::vector<int> vec;
			for (size_t j = 0; j < k; j++)
			{
				auto rest = idx.at<int>(0,j);
				vec.push_back(idx_[rest]);
			}
			retrievalRet = vec;
		}			
	}
	// if(retrievalRet.size() > featurequery->queryVolume)
	// 	retrievalRet.erase(retrievalRet.begin());
	return true;
}

#include "PointCloud.h"
void VisualLocalization::getBestGeomValid()
{

	// std::cout<<"Getting Best Match of "<<i<<"-th query."<<std::endl;
	auto qDesc = featurequery->descs.back();
	auto qKpt = featurequery->kpts.back();
	auto qPtNorm = featurequery->pt3dNorms.back();
	vector<pair<int,int>> numMatches;
	vector<pair<cv::Vec2d,int>> errors;
	for (size_t j = 0; j < retrievalRet.size(); j++)
	{
		vector<int> indexTuple, matchTuple;
		vector<cv::Vec2d> errorTuple;
		for(int ii = std::max(0, retrievalRet[j] - 0); ii <= std::min(matCol-1, retrievalRet[j] + 0); ii++)
		{
			indexTuple.push_back(ii);
		}
		for (auto ii : indexTuple)
		{
			auto dbDesc = featurebase->descs[ii];
			auto dbKpt = featurebase->kpts[ii];
			
			// Get the matched key points
			auto matches = matchFrameToFrameFlann(qDesc, dbDesc);
			cv::Matx33d R = cv::Matx33d::eye();
			cv::Vec3d t = cv::Vec3d(0, 0, 0);
			int numMatch = verifyHmatrix(matches, qKpt, dbKpt);
			// int numMatch = verifyEmatrix(matches, qKpt, dbKpt, R, t);
			matchTuple.push_back(numMatch);
			// numMatches.push_back(std::make_pair(numMatch,retrievalRet[j]));	

			if(configPtr->withDepth)
			{
				auto dbPtNorm = featurebase->pt3dNorms[ii];
				double residual;
				cv::Matx44d poseMatrix = cv::Matx44d::eye();
				// cv::ppf_match_3d::ICP icp(300, 5000.0f, 1.5f, 1);
				myICP icp(500, 0.005f, 3.0f, 1);
				// if (numMatch==0)
				// {
				// 	icp.registerModelToScene(qPtNorm, dbPtNorm, residual, poseMatrix);
				// }
				// cv::ppf_match_3d::Pose3DPtr pose_3D = cv::makePtr<cv::ppf_match_3d::Pose3D>();
				// pose_3D->updatePose(R, t);
				// pose_3D->residual = 1e10;
				// // if(numMatch<configPtr->minInlierNum)
				// // {
					// R = cv::Matx33d::eye();
					// t = cv::Vec3d(0, 0, 0);
				// // }					
				// std::vector<cv::ppf_match_3d::Pose3DPtr> vecPose;//可以设定多个初始值
				// vecPose.push_back(pose_3D);

				double dist_sqr=1e3;
				
				cv::Mat qPtNorm_, dbPtNorm_;
				for (auto match : matches)
				{
					qPtNorm_.push_back(qPtNorm.row(match.queryIdx));
					dbPtNorm_.push_back(dbPtNorm.row(match.trainIdx));
				}
				// int failed = icp.registerModelToScene(qPtNorm, dbPtNorm, residual, poseMatrix);
				int failed = icp.registerModelToScene(qPtNorm_, dbPtNorm_, matches, residual, poseMatrix);
				// int failed = icp.registerModelToScene(qPtNorm, dbPtNorm, vecPose);
				
				// if (!failed)
				// {					
				// 	double x = vecPose[0]->t[0];
				// 	double z = vecPose[0]->t[2];
				// 	dist_sqr = std::pow(x/1000, 2) + std::pow(z/1000, 2);
				// }

				// cv::Vec2d ret(vecPose[0]->residual, dist_sqr);

				if (!failed)
				{					
					double x = poseMatrix(0, 3);
					double z = poseMatrix(2, 3);
					dist_sqr = std::pow(x/1000, 2) + std::pow(z/1000, 2);
				}

				cv::Vec2d ret(residual, dist_sqr);
				errorTuple.push_back(ret);
				// errors.push_back(std::make_pair(ret, retrievalRet[j]));	
			}
		}
		int sum = std::accumulate(matchTuple.begin(), matchTuple.end(), 0);
		numMatches.push_back(std::make_pair(sum/matchTuple.size(),retrievalRet[j]));	
		
		cv::Vec2d sumVec(0, 0);
		for (auto e : errorTuple)
			sumVec += e;
		sumVec = cv::Vec2d(sumVec[0]/errorTuple.size(), sumVec[1]/errorTuple.size());
		errors.push_back(std::make_pair(sumVec, retrievalRet[j]));

	}
	cv::Vec2i tmp;
	// Rank by inlier number of 2D geometric verification
	if(numMatches.empty())
		tmp[1] = -1;
	else
	{
		// Increasing Order
		std::sort(numMatches.begin(), numMatches.end(), 
					[](const pair<int,int>& t1, const pair<int,int>& t2){
						return t1.first<t2.first;
					} ); 
		for(auto iter = numMatches.begin(); iter < numMatches.end(); )
			if(iter->first < configPtr->minInlierNum)
				iter = numMatches.erase(iter);
			else
				break;
		if(numMatches.empty())
			tmp[1] = -1;
		else 
			tmp[1] = numMatches.back().second;	
	}	
	
	// Rank by inlier number of 3D geometric verification
	if(errors.empty())
		tmp[0] = -1;
	else
	{
		// Decreasing Order
		std::sort(errors.begin(), errors.end(), 
					[](const pair<cv::Vec2d,int>& t1, const pair<cv::Vec2d,int>& t2){
						return t1.first[1]>t2.first[1];
					} );
		for(auto iter = errors.begin(); iter < errors.end(); )
			if(iter->first[1] > 15*15)
				iter = errors.erase(iter);
			else
				break;
		if(errors.empty())
			tmp[0] = -1;
		else
		{
			std::cout<<errors.back().first[1]<<std::endl;
			tmp[0] = errors.back().second;		
		}
			
		
	}			
	// if(std::abs(tmp[0]-tmp[1])>20&&(tmp[0]>=0&&tmp[1]>=0))
	// {
	// 	tmp[0] = tmp[1] = -1;
	// }
	// if (tmp[0]!=-1&&tmp[1]!=-1)
	// {
	// 	if(std::abs(tmp[0]-tmp[1])>20)
	// 		tmp[0]=-1;
	// 	else
	// 		tmp[0]=(tmp[0]+tmp[1])/2;
	// 	tmp[1]=-1;
	// }
	geomValRet = tmp;
	// if(geomValRet.size() > featurequery->queryVolume)
	// 	geomValRet.erase(geomValRet.begin());
	
}


void VisualLocalization::getPerformance()
{
	float p, r ;
	std::cout << "F1" << "\t" << "p" << "\t" << "r" << std::endl;
	std::cout << seqSLAMPtr->calculateF1score(&p, &r) << "\t" << p << "\t" << r << std::endl;
	std::cout << "Error" << std::endl;
	std::cout << seqSLAMPtr->calculateErr() << std::endl;
}
std::vector<cv::DMatch> VisualLocalization::matchFrameToFrameFlann(const cv::Mat &mDspts1, const cv::Mat &mDspts2)
{
	std::vector<cv::DMatch> matches;
    if(mDspts1.empty() || mDspts2.empty())
    {
        cout<<"Frame descriptor is empty!\n";
        return matches;
    }
	cv::FlannBasedMatcher flannMatcher;
    std::vector<std::vector<cv::DMatch> > kMatches;

	assert(mDspts2.rows>2);
    flannMatcher.knnMatch(mDspts1, mDspts2, kMatches, 2);
    
	for (int i = 0; i < kMatches.size(); i++)
    {
        if (kMatches[i].size() >= 2 && kMatches[i][0].distance*1.0 / (kMatches[i][1].distance+FLT_MIN) < 0.7)
        {
			matches.push_back(kMatches[i][0]);
        }
    }
	return matches;
}

int VisualLocalization::verifyHmatrix(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point2f>& mKpts1, 
	const std::vector<cv::Point2f>& mKpts2)
{

	vector<cv::Point2f> qPts, dbPts;
	for (int i = 0; i < matches.size(); i++)
    {
		qPts.push_back(mKpts1[matches[i].queryIdx]);
		dbPts.push_back(mKpts2[matches[i].trainIdx]);
    }

	cv::Mat mask;
	if(qPts.size()==0)
	{
		return 0;
	}
	auto hMatrix = cv::findHomography(qPts, dbPts, cv::RANSAC, 4, mask);
	int good = 0;
	for(int i = 0;i < mask.rows; i++)
	{
		auto pRow = mask.ptr<uchar>(i);
		if(*pRow)
			good++;
	}

    return good;
}

int VisualLocalization::verifyEmatrix(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point2f>& mKpts1, 
	const std::vector<cv::Point2f>& mKpts2, cv::Matx33d& R, cv::Vec3d& t)
{

	vector<cv::Point2f> qPts, dbPts;
	for (int i = 0; i < matches.size(); i++)
    {
		qPts.push_back(mKpts1[matches[i].queryIdx]);
		dbPts.push_back(mKpts2[matches[i].trainIdx]);
    }

	cv::Mat mask;
	if(qPts.size()==0)
	{
		return 0;
	}

	cv::Mat eMatrix = cv::findEssentialMat(qPts, dbPts, focal, principalPoint, cv::RANSAC, 0.999, 4.0, mask);
	if(eMatrix.empty())
		return 0;

	int good = cv::recoverPose(eMatrix, qPts, dbPts, R, t, focal, principalPoint, mask);

    return good;
}

// 返回时间戳格式为 yyyy-mm-dd_hh-mm-ss
std::string getTimeStamp()
{
	std::time_t timep = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());	
	struct tm *p = std::localtime(&timep); /*转换为struct tm结构的当地时间*/

	stringstream timeStampStream;
	timeStampStream << 1900 + p->tm_year << setw(2) << setfill('0') << 1 + p->tm_mon << setw(2) << setfill('0') << p->tm_mday << "_";
	timeStampStream << p->tm_hour << "-" << p->tm_min << "-" << p->tm_sec;
	return timeStampStream.str();
}
