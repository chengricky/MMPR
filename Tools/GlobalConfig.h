#ifndef _GLOBALCONFIG_H
#define _GLOBALCONFIG_H

#include <string>
#include "../Header.h"


class GlobalConfig 
{
	// read configuration file
	bool readConfig();
	cv::FileStorage fs;

public:
	GlobalConfig(std::string ymlPath); 
	bool getValid() const { return valid; };

	// the configs
	std::string pathRec;
	std::string pathTest;
	bool fileType; // true=feature, false=image
	int interval;

	
	bool withGPS;
	float GPSthresh;
	//彩色图保持为彩色图(1)或转换为灰度图(0)
	bool withDepth;
	
	float cx, cy, fx, fy;

	int topK;

	// config valid
	bool valid;

	int num_queue;
	int minInlierNum;
};

#endif