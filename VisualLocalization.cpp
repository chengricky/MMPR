#include "VisualLocalization.h"
#include "Tools/Timer.h"
//#include <direct.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
// #include "SequenceSearch.h"
// #include "ParameterTuning.h"
#include "Descriptors/GroundTruth.h"
#include <iomanip>

using namespace std;

//#define GAsearch
//#define Sweepingsearch
#define TEST
//#define GPS_TEST

VisualLocalization::VisualLocalization(GlobalConfig& config)
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
#ifdef GPS_TEST
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
#endif // GPS_TEST

	return true;
}

vector<vector<int>> VisualLocalization::getTopKRetrieval(const int& k)
{
	auto query = featurequery->netVLADs;
	auto ref = featurebase->netVLADs;
	if(ref.empty()||query.empty())
		return vector<vector<int>>();
	else
	{
		vector<vector<int>> vecs;
		for(int i = 0; i < query.rows; i++)
		{
			// cout<<query.row(i).size<<endl;
			cv::Mat idx, dist;
			searchDB->knnSearch(query.row(i), idx, dist, k, cv::flann::SearchParams());
			auto vec = (vector<int>)idx;
			vecs.push_back(vec);
		}		
		return vecs;
	}
}

void VisualLocalization::getBestMatch(const vector<vector<int>>& topk, vector<int>& ret)
{
	for(int i=0; i<topk.size(); i++)
	{
		std::cout<<"Getting Best Match of "<<i<<"-th query."<<std::endl;
		auto qDesc = featurequery->descs[i];
		auto qKpt = featurequery->kpts[i];
		vector<pair<int,int>> numMatches;
		for (size_t j = 0; j < topk[i].size(); j++)
		{
			auto dbDesc = featurebase->descs[topk[i][j]];
			auto dbKpt = featurebase->kpts[topk[i][j]];
			int numMatch = MatchFrameToFrameFlann(qDesc, qKpt, dbDesc, dbKpt);
			numMatches.push_back(std::make_pair(numMatch,topk[i][j]));
		}
		if(numMatches.empty())
			ret.push_back(-1);
		else
		{
			auto tmp = std::max_element(numMatches.begin(), numMatches.end(), 
										[](const pair<int,int>& t1, const pair<int,int>& t2){
											return t1.first>t2.first;
										} );
			ret.push_back(tmp->second);
		
		}

		
	}
	
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
	cv::findHomography(qPts, dbPts, cv::RANSAC, 5, mask);	//TODO:调整阈值？
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



/**
* @brief This method computes the Hamming distance between two binary
* descriptors
* @param desc1 First descriptor
* @param desc2 Second descriptor
* @return Hamming distance between the two descriptors
*/
int hamming_matching(cv::Mat desc1, cv::Mat desc2) {

	int distance = 0;

	if (desc1.rows != desc2.rows || desc1.cols != desc2.cols || desc1.rows != 1 || desc2.rows != 1) {

		std::cout << "The dimension of the descriptors is different." << std::endl;
		return -1;

	}

	for (int i = 0; i < desc1.cols; i++) {
		distance += (*(desc1.ptr<unsigned char>(0) + i)) ^ (*(desc2.ptr<unsigned char>(0) + i));
	}

	return distance;

}

