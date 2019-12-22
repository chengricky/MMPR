
#ifndef SEQUENCESEARCH_H
#define SEQUENCESEARCH_H

#include "Header.h"
#include "Descriptors/GlobalConfig.h"

class SequenceSearch
{
	cv::Size matSize;
	int numSearch;
	float vmin, vmax;
	bool biDirection, isOnlineMode;

	//window thresholding
	int window;
	float ratioTh;

	std::vector<int> ret;
	cv::Mat scoreMat;	
	std::string retPath;

public:
	SequenceSearch(cv::Size matSize, const GlobalConfig& config, const bool& biDirection=false):
					matSize(matSize), numSearch(config.n_q), vmax(config.v_max), 
					vmin(config.v_min), window(config.window), ratioTh(config.ratioTh),
					retPath(config.retPath), biDirection(biDirection), isOnlineMode(config.online){ };
	~SequenceSearch() {};

	void coneSearch(std::vector<std::vector<std::pair<int, float>>> globalResult);

	void windowedUniquenessThresholding();

	std::vector<int> getRet();
	
};

#endif