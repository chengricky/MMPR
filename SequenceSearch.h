
#ifndef SEQUENCESEARCH_H
#define SEQUENCESEARCH_H

#include "Header.h"

class SequenceSearch
{
	
	int numDatabase;
	int numSearch;
	float vmin, vmax;

public:
	SequenceSearch() {};
	void init(cv::Vec2i globalResult, int numDatabase, int numSearch, float vmin, float vmax)
	{
		this->globalResult.push_back(globalResult);
		this->numDatabase = numDatabase;
		this->numSearch = numSearch;
		this->vmax = vmax;
		this->vmin = vmin;
	};
	~SequenceSearch() {};

	void trajectorySearch();
	void coneSearch(const bool& biDirection=false, const bool& isOnlineMode = true);
	cv::Mat scoreMat;	

	std::vector<cv::Vec2i> globalResult;//0-based
	
};

#endif