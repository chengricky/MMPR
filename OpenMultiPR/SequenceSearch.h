#pragma once
#include "Header.h"

class SequenceSearch
{
	std::vector<int> globalResult;//0-based
	cv::Size matSize;
	int numSearch;
	float vmin, vmax;

public:
	SequenceSearch() {};
	void init(std::vector<int> globalResult, cv::Size matSize, int numSearch, float vmin, float vmax)
	{
		this->globalResult = (globalResult);
		this->matSize = matSize;
		this->numSearch = numSearch;
		this->vmax = vmax;
		this->vmin = vmin;
	};
	~SequenceSearch() {};

	void trajectorySearch();
	void coneSearch();
	cv::Mat scoreMat;	
};