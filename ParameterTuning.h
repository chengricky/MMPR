#ifndef _PARAMETERTUNING_H
#define _PARAMETERTUNING_H

#include "Header.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "SequenceSearch.h"

struct Params
{
	float vmax = 1 / 0.4, vmin = 0.4, numsequence = 10;
	float scoreTh = 0.16;
};


class Parameter2F1
{
	std::vector<std::vector<int>> gt;
	std::vector<cv::Vec2i> pGlobal;
	Params parameters;
	std::vector<int> matchingResults; // -1 means empty, 1-based results
	cv::Size matSize;
	SequenceSearch pSS;

public:
	Parameter2F1(std::vector<std::vector<int>> gt, std::vector<cv::Vec2i> GlobalBest, cv::Size matSize) : 
				gt(gt), pGlobal(GlobalBest), matSize(matSize) { };
	~Parameter2F1() { };
	//update different parameters
	void updateParams(float vmax, float vmin, float numseq)
	{
		parameters.vmax = vmax;
		parameters.vmin = vmin;
		parameters.numsequence = numseq;
	}
	void updateParams(float scoreTh)
	{
		parameters.scoreTh = scoreTh;
	}


	void prepare4MultimodalCoefficients();	
	// get matching results
	void placeRecognition(const bool& isWindowUniqueness=true);

	void printMatchingResults() {
		for (size_t i = 0; i < matchingResults.size(); i++)
		{
			std::cout << i << "..." << matchingResults[i] << std::endl;
		}
	}
	void saveMatchingResults() {
		std::ofstream of("result.txt");
		for (size_t i = 0; i < matchingResults.size(); i++)
		{
			of << matchingResults[i] << std::endl;
		}
	}
	std::vector<int> getMatchingResults() { return matchingResults; };

	//calculate F1 score according to groundtruth(gt) and PR results (matchingResults).
	float calculateF1score(float *p = nullptr, float* r = nullptr);
	float calculateErr();

};



#endif