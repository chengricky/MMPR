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
	fs["GPS"] >> withGPS;
	fs["fx"]>>fx;
	fs["fy"]>>fy;
	fs["cx"]>>cx;
	fs["cy"]>>cy;

	return true;
}
