#include "VisualLocalization.h"
#include "Tools/Timer.h"
#include <dirent.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
// #include "SequenceSearch.h"
#include "ParameterTuning.h"
#include "Descriptors/GroundTruth.h"
#include <iomanip>

using namespace std;

//#define GAsearch
//#define Sweepingsearch
#define TEST
//#define GPS_TEST

VisualLocalization::VisualLocalization(GlobalConfig& config) : geomValRet(std::vector<int>()), 
												retrievalRet(std::vector<std::vector<int>>())
{ 	
	if (!config.getValid())
	{
		throw invalid_argument("Configuration is invalid!");
	}
	std::cout << "====>Reading Database Features." << std::endl;
	featurebase = make_shared<DescriptorFromFile>(config, true);
	
	std::cout << "====>Reading Query Features." << std::endl;
	featurequery = make_shared<DescriptorFromFile>(config, false);	

	matRow = featurequery->getVolume();
	matCol = featurebase->getVolume();

	std::cout << "====>Reading Ground Truth." << std::endl;
	ground.init(config.pathTest + "/of.txt", config.pathRec + "/of.txt");
	ground.generateGroundTruth(5);

	// construct the FLANN database
	std::cout << "====>Constructing the FLANN database." << std::endl;
	cv::flann::KDTreeIndexParams kdIndex(5); //同时建立多个随机kd树，确定tree的数量
	searchDB = make_shared<cv::flann::Index>(featurebase->netVLADs, kdIndex); //默认为L2 distance

};

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
	auto GPSQuery = featurequery->GPS;
	auto GPSRef = featurebase->GPS;
	if (GPSQuery.empty() || GPSRef.empty())
	{

		GPSDistance = cv::Mat();
		GPSMask_uchar = cv::Mat();
	}
	else
	{
		GPSDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<float>(i, j) = GNSSdistance(GPSQuery.at<float>(i,1), GPSQuery.at<float>(i, 0), GPSRef.at<float>(j, 1), GPSRef.at<float>(j, 0));
		cv::Mat GPSuchar;
		GPSDistance.convertTo(GPSuchar, CV_8U);
		cv::threshold(GPSuchar, GPSMask_uchar, gnssTh, 255, cv::THRESH_BINARY_INV);
		//cv::threshold(GPSDistance, GPSMask, gnssTh, FLT_MAX, cv::THRESH_BINARY_INV);
		//GPSMask.convertTo(GPSMask_uchar, CV_8U);
	}
}

void VisualLocalization::getTopKRetrieval(const int& k)
{
	auto query = featurequery->netVLADs;
	auto ref = featurebase->netVLADs;
	if (!featurequery->GPS.empty() && !featurebase->GPS.empty())
		getDistanceMatrix(15);
	
	if(ref.empty()||query.empty())
		retrievalRet = vector<vector<int>>();
	else
	{
		for(int i = 0; i < query.rows; i++)
		{
			cv::Mat idx, dist;
			if(GPSMask_uchar.empty())
			{
				searchDB->knnSearch(query.row(i), idx, dist, k, cv::flann::SearchParams());
				retrievalRet.push_back((vector<int>)idx);
			}
			else
			{
				vector<int> vec;
				searchDB->knnSearch(query.row(i), idx, dist, k*5, cv::flann::SearchParams());
				for (size_t j = 0; j < k*5; j++)
				{
					auto rest = idx.at<int>(0,j);
					if(GPSMask_uchar.at<uchar>(i,rest))
						vec.push_back(rest);
					if(vec.size()>=k)
						break;
				}
				retrievalRet.push_back(vec);
			}		

		}		
	}
}

void VisualLocalization::getBestGeomValid()
{
	for(int i=0; i<retrievalRet.size(); i++)
	{
		std::cout<<"Getting Best Match of "<<i<<"-th query."<<std::endl;
		auto qDesc = featurequery->descs[i];
		auto qKpt = featurequery->kpts[i];
		vector<pair<int,int>> numMatches;
		for (size_t j = 0; j < retrievalRet[i].size(); j++)
		{
			auto dbDesc = featurebase->descs[retrievalRet[i][j]];
			auto dbKpt = featurebase->kpts[retrievalRet[i][j]];
			int numMatch = MatchFrameToFrameFlann(qDesc, qKpt, dbDesc, dbKpt);
			numMatches.push_back(std::make_pair(numMatch,retrievalRet[i][j]));
		}
		if(numMatches.empty())
			geomValRet.push_back(-1);
		else
		{
			auto tmp = std::max_element(numMatches.begin(), numMatches.end(), 
										[](const pair<int,int>& t1, const pair<int,int>& t2){
											return t1.first>t2.first;
										} );
			geomValRet.push_back(tmp->second);		
		}		
	}	
}

void VisualLocalization::getSeqMatch()
{
	Parameter2F1 pt(ground.gt, geomValRet, cv::Size(matCol, matRow));
	float p, r ;
	pt.placeRecognition();
	generateVideo(pt.getMatchingResults());
	std::cout << "F1" << "\t" << "p" << "\t" << "r" << std::endl;
	std::cout << pt.calculateF1score(&p, &r) << "\t" << p << "\t" << r << std::endl;
	std::cout << "Error" << std::endl;
	std::cout << pt.calculateErr() << std::endl;
}

int VisualLocalization::MatchFrameToFrameFlann(const cv::Mat &mDspts1, const std::vector<cv::Point2f>& mKpts1,
												const cv::Mat &mDspts2, const std::vector<cv::Point2f>& mKpts2)
{
    if(mDspts1.empty() || mDspts2.empty())
    {
        cout<<"Frame descriptor is empty!\n";
        return 0;
    }
	cv::FlannBasedMatcher flannMatcher;
    std::vector<std::vector<cv::DMatch> > kMatches;

	assert(mDspts2.rows>2);
    flannMatcher.knnMatch(mDspts1, mDspts2, kMatches,3);

    int good = 0;
	vector<cv::Point2f> qPts, dbPts;
	for (int i = 0; i < kMatches.size(); i++)
    {
        if (kMatches[i].size() >= 2 && kMatches[i][0].distance*1.0 / kMatches[i][1].distance < 0.7)
        {
			qPts.push_back(mKpts1[kMatches[i][0].queryIdx]);
			dbPts.push_back(mKpts2[kMatches[i][0].trainIdx]);
        }
    }
	cv::Mat mask;
	cv::findHomography(qPts, dbPts, cv::RANSAC, 2, mask);	//TODO:调整阈值？
	for(int i = 0;i < mask.cols; i++)
	{
		auto pRow = mask.ptr<uchar>(i);
		if(*pRow)
			good++;
	}
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

#include <sys/stat.h>
#include <sys/types.h>
bool VisualLocalization::generateVideo(std::vector<int> matchingResults, std::string path)
{
	// if (path.empty())
	// {
	// 	DIR *dir;
	// 	if ((dir = opendir("results")) == NULL)
	// 		mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	// 	path = "results\\";
	// 	path += getTimeStamp()+".avi";
	// }
	// cv::VideoWriter writer(path, cv::VideoWriter::fourcc('F', 'L', 'V', '1'), 1.0, cv::Size(320, 240 * 2));
	// if (!writer.isOpened())
	// {
	// 	writer.release();
	// 	return false;
	// }
	// for (size_t i = 0; i < matchingResults.size(); i++)
	// {
	// 	cv::Mat query_database = cv::imread(descriptorquery->picFiles.getColorImgPath(i));
	// 	if (matchingResults[i]!=-1)
	// 	{
	// 		query_database.push_back(cv::imread(descriptorbase->picFiles.getColorImgPath(matchingResults[i])));
	// 	}
	// 	else
	// 	{
	// 		query_database.push_back(cv::Mat(cv::Size(320, 240), CV_8UC3, cv::Scalar(0, 0, 0)));
	// 	}
	// 	writer << query_database;
	// }
	// writer.release();
	return true;
}
