#include "VisualLocalization.h"
#include "Tools/Timer.h"
//#include <direct.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
#include "SequenceSearch.h"
#include "ParameterTuning.h"
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

	featurebase = new DescriptorFromFile(config, true);
	std::cout << "Database Features were read." << std::endl;
	featurequery = new DescriptorFromFile(config, false);
	std::cout << "Query Features were read." << std::endl;
	matRow = featurequery->getVolume();
	matCol = featurebase->getVolume();

	cv::destroyAllWindows();
	ground.init(config.pathTest + "\\of.txt", config.pathRec + "\\of.txt");
	ground.generateGroundTruth(5);

	// construct the FLANN database
	cv::flann::KDTreeIndexParams kdIndex(5); //同时建立多个随机kd树，确定tree的数量
	searchDB = new cv::flann::Index(featurebase->netVLADs, kdIndex); //默认为L2 distance

};

VisualLocalization::~VisualLocalization()
{
	if (featurebase != nullptr)
	{
		delete featurebase;
	}
	if (featurequery != nullptr)
	{
		delete featurequery;
	}
};


bool VisualLocalization::getDistanceMatrix(float gnssTh)
{
	int matRow = featurequery->getVolume();
	int matCol = featurebase->getVolume();


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
		cv::threshold(GPSuchar, GPSMask_uchar, gnssTh, 255, cv::THRESH_BINARY);
		//cv::threshold(GPSDistance, GPSMask, gnssTh, FLT_MAX, cv::THRESH_BINARY_INV);
		//GPSMask.convertTo(GPSMask_uchar, CV_8U);
	}
	
	return true;
}


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
		cv::Mat idx;
		vector<vector<int>> vecs;
		for(int i = 0; i < query.rows; i++)
		{
			searchDB->knnSearch(query.row(i), idx, cv::Mat(), k, cv::flann::SearchParams());
			auto vec = (vector<int>)(idx.reshape(1,k));
			vecs.push_back(vec);
		}		
		return vecs;
	}
}

void VisualLocalization::getBestMatch(const vector<vector<int>>& topk, vector<int>& ret)
{
	for(int i=0; i<topk.size(); i++)
	{
		auto qDesc = featurequery->descs[i];
		auto qKpt = featurequery->kpts[i];
		vector<pair<int,int>> numMatches;
		for (size_t j = 0; j < topk[i].size(); j++)
		{
			auto dbDesc = featuredatabase->descs[topk[i][j]];
			auto dbKpt = featurdatabase->kpts[topk[i][j]];
			int numMatch = MatchFrameToFrameFlann(qDesc, qKpt, dbDesc, dbKpt);
			numMatches.push_back(make_pair<int,int>(numMatch,topk[i][j]));
		}
		if(numMatches.empty())
			ret.push_back(-1);
		else
		{
			auto tmp = std::max_element(numMatches.begin(), numMatches.end(), 
			[](const pair<int,int>& t1, const pair<int,int>& t2){return t1.first>t2.first} );
			ret.push_back(tmp->second);
		
		}
		

		
	}
	
}

int VisualLocalization::MatchFrameToFrameFlann(const cv::Mat &mDspts1, const std::vector<cv::KeyPoint>& mKpts1,
												const cv::Mat &mDspts2, const std::vector<cv::KeyPoint>& mKpts2)
{
    if(mDspts1.empty() || mDspts2.empty())
    {
        cout<<"Frame descriptor is empty!\n";
        return 0;
    }
    cv::FlannBasedMatcher flannMatcher(new cv::flann::LshIndexParams(6, 9, 1));
    std::vector<std::vector<cv::DMatch> > kMatches;

    flannMatcher.knnMatch(mDspts1, mDspts2, kMatches, 2);

    int good = 0;
	vector<cv::KeyPoint> qPts, dbPts;
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



void VisualLocalization::getBestMatch_FeatureFile()
{
	// generate distance matric for netVLAD
	// netVLAD
	auto Query = featurequery->netVLADs;
	auto Ref = featurebase->netVLADs;
	if (Query.empty() || Ref.empty())
		netVLAD_Distance = cv::Mat();
	else
	{
		netVLAD_Distance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
			{
				//netVLAD_Distance.at<float>(i, j) = cv::norm(Query.row(i), Ref.row(j), cv::NORM_L2);
				double ab = Query.row(i).dot(Ref.row(j));
				double aa = Query.row(i).dot(Query.row(i));
				double bb = Ref.row(j).dot(Ref.row(j));
				double cosine = ab / std::sqrt(aa*bb);
				netVLAD_Distance.at<float>(i, j) = 1-cosine;
			}
				
	}
	// GPS
	auto GPSQuery = featurequery->GPS;
	auto GPSRef = featurebase->GPS;
	if (GPSQuery.empty() || GPSRef.empty())

		GPSDistance = cv::Mat();
		GPSMask_uchar = cv::Mat();
	}
	else
	{
		GPSDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<float>(i, j) = GNSSdistance(GPSQuery.at<float>(i, 1), GPSQuery.at<float>(i, 0), GPSRef.at<float>(j, 1), GPSRef.at<float>(j, 0));
		cv::Mat GPSuchar;
		GPSDistance.convertTo(GPSuchar, CV_8U);
		cv::threshold(GPSuchar, GPSMask_uchar, (float)15, 255, cv::THRESH_BINARY);
	}
	// get global best idx
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
	/// netVLAD 
	cv::Mat netVLADdistanceMat = netVLAD_Distance;
	if (!GPSMask_uchar.empty())
	{
		netVLADdistanceMat.setTo(FLT_MAX, GPSMask_uchar);
	}	
	for (size_t i = 0; i < matRow; i++)//query
	{
		//When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is a single-row or single-column matrix.
		//In OpenCV (following MATLAB) each array has at least 2 dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be (i1,0)/(i2,0)) 
		//and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be (0,j1)/(0,j2)).
		int* minPos = new int[2];
		cv::minMaxIdx(netVLADdistanceMat.row(i), nullptr, nullptr, minPos, nullptr);	//可否改成top-k		?
		std::cout << minPos[1] << std::endl;
		netVLADglobalResult.push_back(minPos[1]);
		delete minPos;
	}

	cv::Size matSize(matCol, matRow);

#ifdef GPS_TEST
	Parameter2F1 pt(ground.gt, GGglobalResult, GPSGlobalBest, BoWGlobalBest_D, BoWGlobalBest_IR, GISTGlobalBest_RGB, GISTGlobalBest_D, GISTGlobalBest_IR,
		LDBGlobalBest_RGB, LDBGlobalBest_D, LDBGlobalBest_IR, matSize);
	float *p = new float;
	float *r = new float;
	std::cout << "GPS_TEST" << std::endl;
	std::vector<double> coeff4 = { 1,0,0, 0,0,0, 0,0,0, 0 };
	pt.updateParams(coeff4);
	pt.updateParams(0);
	pt.updateParams(1, 1, 1);
	pt.placeRecognition();
	//pt.printMatchingResults();
	std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	std::cout << pt.calculateErr() << std::endl;
	std::cout << "-----" << std::endl;
#else
	Parameter2F1 pt(ground.gt, netVLADglobalResult, std::vector<int>(), std::vector<int>(), std::vector<int>(), std::vector<int>(), std::vector<int>(),
		std::vector<int>(), std::vector<int>(), std::vector<int>(), matSize);
#endif // GPS_TEST


#ifdef GAsearch
	//// use OpenGA to optimize coefficents
	std::vector<double> coeff;
	pt.prepare4MultimodalCoefficients();
	optimizeMultimodalCoefficients(&pt, coeff);
	pt.updateParams(coeff);
	////// calculate score matrix for single descriptor
	pt.placeRecognition();
	//pt.printMatchingResults();
	std::cout << pt.calculateF1score() << std::endl;
#endif // GAsearch

#ifdef Sweepingsearch
	// sweep the parameter
	for (float i = 0.1; i <= 0.8; i += 0.05)
	{
		pt.updateParams(1 / i, i, 16);
		pt.placeRecognition();
		std::cout << i << "\t" << pt.calculateF1score() << std::endl;
	}
	std::cout << "\n";
	// 应该实际的值是一半
	for (float i = 3; i <= 79; i += 4)
	{
		pt.updateParams(1 / 0.4, 0.4, i);
		pt.placeRecognition();
		std::cout << i << "\t" << pt.calculateF1score() << std::endl;
	}
	std::cout << "\n";
	pt.updateParams(1 / 0.4, 0.4, 10);
	for (float i = 0; i <= 0.6; i += 0.02)
	{
		pt.updateParams(i);
		pt.placeRecognition();
		float *p = new float;
		float *r = new float;
		std::cout << i << "\t" << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
		delete p;
		delete r;
	}
#endif // Sweepingsearch

#ifdef TEST
	float *p = new float;
	float *r = new float;
	std::cout << "default" << std::endl;
	std::vector<double> coeff3(9, 0);
	pt.updateParams(coeff3, 8, 1);
	pt.updateParams(0.07);
	pt.placeRecognition();
	pt.saveMatchingResults();
	std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	std::cout << pt.calculateErr() << std::endl;
	std::cout << "-----" << std::endl;

	//std::cout << "lambda=1" << std::endl;
	//std::vector<double> coeff3(8, 1);
	//pt.updateParams(coeff3);
	//pt.updateParams(0.16 * 9 / 10);
	//pt.placeRecognition();
	////pt.printMatchingResults();
	//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	//std::cout << pt.calculateErr() << std::endl;
	//std::cout << "-----" << std::endl;

	//std::cout << "without GoogLeNet" << std::endl;
	//std::vector<double> coeff = 
	//{ 1.24517777,	1.685089537,
	//	1.578978596,	1.091489515,	0.987401397,
	//	0.526293315,	0.623399388,	0.840100646,
	//	1.422069836 };
	//pt.updateParams(coeff);
	//pt.updateParams(std::vector<double>(), 8, 0);
	//pt.updateParams(0.16 * 8.577930164	/ 10);
	//pt.placeRecognition();
	////pt.printMatchingResults();
	//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	//std::cout << pt.calculateErr() << std::endl;
	//std::cout <<"-----" << std::endl;
	////
	//std::cout << "without GIST" << std::endl;
	//pt.updateParams(coeff);
	//pt.updateParams(std::vector<double>(), 3, 0);
	//pt.updateParams(std::vector<double>(), 4, 0);
	//pt.updateParams(std::vector<double>(), 2, 0);
	//pt.updateParams(0.16 * 6.342130492	/ 10);
	//pt.placeRecognition();
	////pt.printMatchingResults();
	//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	//std::cout << pt.calculateErr() << std::endl;
	//std::cout << "-----" << std::endl;

	//std::cout << "without LDB" << std::endl;
	//pt.updateParams(coeff);
	//pt.updateParams(std::vector<double>(), 6, 0);
	//pt.updateParams(std::vector<double>(), 7, 0);
	//pt.updateParams(std::vector<double>(), 5, 0);
	//pt.updateParams(0.16 * 8.010206651	/ 10);
	//pt.placeRecognition();
	////pt.printMatchingResults();
	//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	//std::cout << pt.calculateErr() << std::endl;
	//std::cout << "-----" << std::endl;

	//std::cout << "without BoW" << std::endl;
	//pt.updateParams(coeff);
	//pt.updateParams(std::vector<double>(), 0, 0);
	//pt.updateParams(std::vector<double>(), 1, 0);
	//pt.updateParams(0.16 * 7.069732693	/ 10);
	//pt.placeRecognition();
	////pt.printMatchingResults();
	//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
	//std::cout << pt.calculateErr() << std::endl;
	//std::cout << "-----" << std::endl;

	delete p;
	delete r;
#endif


}

static bool getDistanceMap(cv::Mat DistanceMat, std::string DescriptorTtype)
{
	cv::Mat CS_norm;
	cv::normalize(DistanceMat, CS_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::resize(CS_norm, CS_norm, cv::Size(CS_norm.cols * 5, CS_norm.rows * 5));
	cv::imshow("Distance Matrix - "+ DescriptorTtype, CS_norm);
	return true;
}


cv::Mat VisualLocalization::enhanceMatrix(const cv::Mat& distanceMat, int winSize) //winSize必须为奇数
{
	assert(winSize % 2);
	cv::Mat enhancedMat(distanceMat.size(), CV_32FC1);
	if (!distanceMat.empty())
	{
		for (size_t i = 0; i < distanceMat.rows; i++)
		{
			const float* pDis = distanceMat.ptr<float>(i);
			float* pEnh = enhancedMat.ptr<float>(i);
			for (size_t j = 0; j < (winSize-1)/2; j++)
			{
				float sum = std::accumulate(pDis, pDis + winSize, 0.0f);//是否可行？
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis, pDis + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //标准差
				pEnh[j] = (pDis[j]-mean) / stdev;
			}
			for (size_t j = 0; j < distanceMat.cols-(winSize-1); j++)
			{
				float sum = std::accumulate(pDis+j, pDis+j + winSize, 0.0f);//是否可行？
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis + j, pDis + j + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //标准差
				pEnh[j+ (winSize - 1) / 2] = (pDis[j + (winSize - 1) / 2] - mean) / stdev;
			}
			for (size_t j = distanceMat.cols - (winSize - 1) / 2; j < distanceMat.cols; j++)
			{
				float sum = std::accumulate(pDis + distanceMat.cols - winSize, pDis + distanceMat.cols, 0.0f);//是否可行？
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis, pDis + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //标准差
				pEnh[j] = (pDis[j] - mean) / stdev;
			}
		}
	}
	else
	{
		enhancedMat = distanceMat;
	}
	return enhancedMat;
}

bool VisualLocalization::getEnhancedDistanceMatrix(int winSize)
{
	//CSDistance = enhanceMatrix(CSDistance, winSize);
	//GISTDistance = enhanceMatrix(GISTDistance, winSize);
	//GPSDistance = enhanceMatrix(GPSDistance, winSize);
	//LDBDistance = enhanceMatrix(LDBDistance, winSize);
	return true;
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

