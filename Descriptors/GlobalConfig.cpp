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
	fs["Interval"] >> interval;
	fs["RetPath"] >> retPath;
	fs["n_q"] >> n_q;
	fs["v_max"] >> v_max;
	fs["v_min"] >> v_min;
	fs["online"] >> online;
	fs["ratioTh"] >> ratioTh;
	fs["window"] >> window;
	fs["UseSceneNum"]>>sceneTopK;

	return true;
}
