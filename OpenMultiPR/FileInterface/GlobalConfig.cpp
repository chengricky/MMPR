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
	fs["BoW_CodeBook"] >> codeBook;
	fs["FileType"] >> fileType;
	fs["ColorImg"] >> useColor;
	fs["DepthImg"] >> useDepth;
	fs["InfraredImg"] >> useIR;
	fs["queryImgSize"] >> qImgSize;
	fs["databaseImgSize"] >> dImgSize;
	fs["GIST"] >> useGIST;
	fs["Interval"] >> interval;
	fs["CS"] >> useCS;
	fs["ORB-BoW"] >> useBoW;
	fs["LDB"] >> useLDB;
	fs["Color"] >> isColor;
	fs["GPS"] >> withGPS;

	return true;
}
