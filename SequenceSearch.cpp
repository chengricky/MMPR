#include "SequenceSearch.h"


void SequenceSearch::coneSearch(std::vector<std::vector<std::pair<int, float>>> globalResult)
{
	if (globalResult.empty())
	{
		scoreMat = cv::Mat(matSize.height,matSize.width, CV_32FC1, 0.0f);
		return;
	}
	scoreMat = cv::Mat(matSize.height, matSize.width, CV_32FC1, 0.0f);
	for (int i = 0; i < matSize.height; i++)//query
	{
		float* pS = scoreMat.ptr<float>(i);
		for (int j = 0; j < matSize.width; j++)//database
		{
			int min_y, max_y;		
			float sum_dist = 0;		
			if (isOnlineMode)	// on-line mode
			{
				min_y = std::max(0, (int)i - (numSearch - 1) );
				max_y = i;
			}			
			else	// off-line mode
			{
				min_y = std::max(0, (int)i - (numSearch - 1) / 2);
				max_y = std::min((int)i + (numSearch - 1) / 2, matSize.height - 1);
			}
			for (int k = min_y; k <= max_y; k++)//query within coneopen
			{
				int min_x=-1, max_x=-1;
				if (k<i)
				{
					min_x = std::max(int((k - i)*vmax + j), 0);
					max_x =  (k - i)*vmin + j;
				}
				else
				{
					max_x = std::min(int((k - i)*vmax + j), matSize.width -1);
					min_x = (k - i)*vmin + j;
				}
				for (auto e : globalResult[k])
				{
					if(e.first>=min_x &&e.first<=max_x)
						sum_dist += 1/e.second;
					else if (biDirection && e.first >= std::min(2 * j - max_x, matSize.width - 1)
						&& e.first <= std::min(2 * j - min_x, matSize.width - 1))
						sum_dist += 1/e.second;
				}								
			}

			pS[j] = float(sum_dist) / float(max_y- min_y+1);

		}
		// double maxV;
		// int maxL[2];
		// cv::minMaxIdx(scoreMat.row(i), 0, &maxV, 0, maxL);
		// ret.push_back(maxL[1]);

	}
}

cv::Mat getScoreMap(const cv::Mat& DistanceMat)
{
	cv::Mat ret;
	DistanceMat.copyTo(ret);
	for (size_t i = 0; i < DistanceMat.rows; i++)
	{
		cv::normalize(DistanceMat.row(i), ret.row(i), 0, 255, cv::NORM_MINMAX);
	}
	return ret;
}

void SequenceSearch::windowedUniquenessThresholding()
{
	auto scoreNormed = getScoreMap(scoreMat);
	std::cout<<"====>Getting Final Results by Thresholding."<<std::endl;
	for (size_t i = 0; i < scoreNormed.rows; i++) //query
	{
		double maxVal;
		int* maxPos = new int[2];
		cv::minMaxIdx(scoreNormed.row(i), nullptr, &maxVal, nullptr, maxPos);		
		
		// int window = 10;
		// double ratioTh = 1.1;
		int a = std::max(0, maxPos[1] - window / 2);
		int b = std::min(scoreNormed.cols-1, maxPos[1] + window / 2);
		cv::Mat synScoreMask = cv::Mat::ones(1, scoreNormed.cols, CV_8UC1);

		for (size_t i = a; i <= b; i++)
			synScoreMask.at<uchar>(0, i) = 0;

		double maxVal_;
		int* maxPos_ = new int[2];
		cv::minMaxIdx(scoreNormed.row(i), nullptr, &maxVal_, nullptr, maxPos_, synScoreMask);
		double ratio = maxVal / maxVal_;

		if (ratio > ratioTh)
			ret.push_back(maxPos[1] + 1);
		else
			ret.push_back(-1);
		delete maxPos;
		delete maxPos_;	
	}
}

#include <sstream>
std::vector<int> SequenceSearch::getRet(){

	std::stringstream pth;
	pth<<retPath<<"/"<<this->ratioTh<<".txt";
	std::ofstream file(pth.str());
	for(int i=0; i<ret.size(); i++)
	{
		file<<i<<"\t"<<ret[i]<<std::endl;
	}
	file.close();


	
	return ret;
	
}