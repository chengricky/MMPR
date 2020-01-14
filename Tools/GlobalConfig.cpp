#include "GlobalConfig.h"


GlobalConfig::GlobalConfig(std::string ymlPath) 
{
	fs.open(ymlPath, cv::FileStorage::READ);
	valid = true;
	if (!readConfig())
	{
		valid = false;
	}
}

bool GlobalConfig::readConfig()
{
	if (!fs.isOpened())
	{
		return false;
	}
	fs["PathRec"] >> pathRec;
	fs["PathTest"] >> pathTest;
	fs["FileType"] >> fileType;
	fs["Interval"] >> interval;
	
	fs["GNSS"] >> withGPS;
	fs["GNSSthres"] >> GPSthresh;
	
	fs["fx"]>>fx;
	fs["fy"]>>fy;
	fs["cx"]>>cx;
	fs["cy"]>>cy;
	fs["topK"]>>topK;
	fs["Depth"]>>withDepth;
	fs["num_queue"]>>num_queue;
	fs["minInlierNum"]>>minInlierNum;

	return true;
}
