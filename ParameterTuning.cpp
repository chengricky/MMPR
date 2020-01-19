#include "ParameterTuning.h"
#include "SequenceSearch.h"
#include <cmath>
#include <algorithm>


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

void Parameter2F1::placeRecognition(cv::Vec2i GlobalBest, const bool& isWindowUniqueness)
{
	// if (!matchingResults.empty())
	// 	std::vector<int>().swap(matchingResults);
	
	std::cout<<"====>Getting Score Matrix."<<std::endl;
	pSS.init(GlobalBest, numRef, parameters.numsequence, parameters.vmin, parameters.vmax);
	pSS.coneSearch();
	auto scoreNormed = getScoreMap(pSS.scoreMat);

	std::cout<<"====>Getting Final Results by Thresholding."<<std::endl;
	for (size_t i = 0; i < scoreNormed.rows; i++) //query
	{
		double maxVal;
		int* maxPos = new int[2];
		cv::minMaxIdx(scoreNormed.row(i), nullptr, &maxVal, nullptr, maxPos);		
		
		if (isWindowUniqueness)	// Windowed Uniqueness Thresholding
		{
			int window = 15;
			double ratioTh = 1.1;
			int a = std::max(0, maxPos[1] - window / 2);
			int b = std::min(scoreNormed.cols-1, maxPos[1] + window / 2);
			cv::Mat synScoreMask = cv::Mat::ones(1, scoreNormed.cols, CV_8UC1);

			for (size_t j = a; j <= b; j++)
				synScoreMask.at<uchar>(0, j) = 0;

			double maxVal_;
			int* maxPos_ = new int[2];
			cv::minMaxIdx(scoreNormed.row(i), nullptr, &maxVal_, nullptr, maxPos_, synScoreMask);
			double ratio = maxVal / maxVal_;

			if (ratio > ratioTh)
				matchingResults.push_back(maxPos[1] + 1);
			else
				matchingResults.push_back(-1);
			delete maxPos;
			delete maxPos_;
		}
		else	// Score Thresholding
		{
			if (maxVal > parameters.scoreTh)
				matchingResults.push_back(maxPos[1] + 1);
			else
				matchingResults.push_back(-1);
			delete[] maxPos;
		}
	}
}

void Parameter2F1::prepare4MultimodalCoefficients(cv::Vec2i GlobalBest)
{
	pSS.init(GlobalBest, numRef, parameters.numsequence, parameters.vmin, parameters.vmax);
	pSS.coneSearch();
}

float Parameter2F1::calculateF1score(float* precision, float* recall)
{
	cv::Mat plotPR(matchingResults.size(), numRef, CV_8UC3, cv::Scalar(255, 255, 255));
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (!gt[i].empty())
		{
			for (auto var : gt[i])
			{
				plotPR.at<cv::Vec3b>(i, var-1) = cv::Vec3b(0,255,0);
			}			
		}
	}
	for (size_t i = 0; i < matchingResults.size(); i++)
	{
		if (matchingResults[i] != -1)
			plotPR.at<cv::Vec3b>(i, matchingResults[i]-1) = cv::Vec3b(0, 0, 255);
	}
	// std::cout<<pSS.globalResult.size();
	for(int i=0;i<pSS.globalResult.size();i++)
	{
		if (pSS.globalResult[i](0) != -1) //3D
		{
			plotPR.at<cv::Vec3b>(i, pSS.globalResult[i](0)) = cv::Vec3b(255, 0, 0);	
		}				
		if (pSS.globalResult[i](1) != -1) //2D
		{
			// std::cout<<i<<" "<<pSS.globalResult[i](1)<<std::endl;
			plotPR.at<cv::Vec3b>(i, pSS.globalResult[i](1)) = cv::Vec3b(0, 0, 0);	
		}				
	}
	cv::resize(plotPR, plotPR, cv::Size(), 3, 3, cv::INTER_NEAREST);
	cv::imshow("plotPR", plotPR);
	cv::waitKey(1);
	int fp = 0, tp = 0, tn = 0, fn = 0;
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (gt[i].empty()) // ground truth is empty 
		{
			if (matchingResults[i] == -1)
			{
				tn++;
			}
			else
			{
				fp++;
			}
		}
		else  // ground truth is not empty
		{
			if (matchingResults[i]==-1)
			{
				fn++;
			}
			else
			{
				std::vector<int>::iterator ret = std::find(gt[i].begin(), gt[i].end(), matchingResults[i]);
				if (ret == gt[i].end()) // no findings
				{
					fp++;
				}
				else
				{
					tp++;
				}
			}

		}
	}
	float p = (float)tp / (tp + fp);
	float r = (float)tp / (tp + fn);
	float f1 = 2 * (p*r) / (p + r);
	if (precision!=nullptr)
	{
		*precision = p;
	}
	if (recall!=nullptr)
	{
		*recall = r;
	}
	return f1;
}


float Parameter2F1::calculateErr()
{
	// cv::Mat plotPR(matchingResults.size(), numRef, CV_8UC3, cv::Scalar(255, 255, 255));
	// for (size_t i = 0; i < gt.size(); i++)
	// {
	// 	if (!gt[i].empty())
	// 	{
	// 		for (auto var : gt[i])
	// 		{
	// 			plotPR.at<cv::Vec3b>(i, var - 1) = cv::Vec3b(0, 255, 0);
	// 		}
	// 	}
	// }
	// for (size_t i = 0; i < matchingResults.size(); i++)
	// {
	// 	if (matchingResults[i] != -1)
	// 		plotPR.at<cv::Vec3b>(i, matchingResults[i] - 1) = cv::Vec3b(0, 0, 255);
	// }
	// cv::resize(plotPR, plotPR, cv::Size(), 2, 2);
	// cv::imshow("plotPR", plotPR);
	// cv::waitKey(1);
	int err_sum = 0, num_sum = 0;
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (!gt[i].empty()&& matchingResults[i] != -1) // ground truth is not empty 
		{
			std::vector<int>::iterator ret = std::find(gt[i].begin(), gt[i].end(), matchingResults[i]);
			if (ret == gt[i].end()) // no findings
			{
				//err_sum += std::min(std::abs(matchingResults[i]- *(gt[i].begin())), std::abs(matchingResults[i] - *(gt[i].end()-1)));
				err_sum += std::abs(matchingResults[i] - (*(gt[i].begin()) + *(gt[i].end() - 1))/2.0);
			}
			num_sum += 1;
		}
	}
	return float(err_sum)/ num_sum;
}