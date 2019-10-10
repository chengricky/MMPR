
#ifndef SEQUENCESEARCH_H
#define SEQUENCESEARCH_H

#include "Header.h"

class SequenceSearch
{
	std::vector<cv::Vec2i> globalResult;//0-based
	cv::Size matSize;
	int numSearch;
	float vmin, vmax;

public:
	SequenceSearch() {};
	void init(std::vector<cv::Vec2i> globalResult, cv::Size matSize, int numSearch, float vmin, float vmax)
	{
		this->globalResult = (globalResult);
		this->matSize = matSize;
		this->numSearch = numSearch;
		this->vmax = vmax;
		this->vmin = vmin;
	};
	~SequenceSearch() {};

	void trajectorySearch();
	void coneSearch(const bool& biDirection=false, const bool& isOnlineMode = true);
	cv::Mat scoreMat;	
};

#endif