#ifndef _GLOBALCONFIG_H
#define _GLOBALCONFIG_H

#include <string>
#include "../Header.h"


// 设计模式 - 单件 singleton
class GlobalConfig 
{
	// 私有的类的静态成员，不依赖于对象而存在
	static GlobalConfig config;

	// 私有构造函数、拷贝构造函数、赋值操作符
	GlobalConfig(std::string yaml);
	GlobalConfig(const GlobalConfig&) ;
	GlobalConfig& operator=(GlobalConfig) {};

	// read configuration file
	bool readConfig();

	cv::FileStorage fs;

public:
	static GlobalConfig& instance() { return config; }
	virtual ~GlobalConfig() {};
	bool getValid() const { return valid; };

	// the configs
	std::string pathRec, pathTest, retPath;
	int interval, n_q, window, sceneTopK;
	bool online;
	float v_min, v_max, ratioTh;




	// config valid
	bool valid;
};

#endif