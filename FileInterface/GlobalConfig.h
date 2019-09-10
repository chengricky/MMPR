#pragma once
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
	cv::Size dImgSize;
	cv::Size qImgSize;
	std::string pathRec;
	std::string pathTest;
	std::string codeBook;
	bool fileType; // true=feature, false=image
	int interval;

	//彩色图保持为彩色图(1)或转换为灰度图(0)
	bool isColor; 
	bool useColor;
	bool useIR;
	bool useDepth;
	bool withGPS;
	
	// Descriptor Type
	bool useGIST;
	bool useCS;
	bool useBoW;
	bool useLDB;

	// config valid
	bool valid;
};

