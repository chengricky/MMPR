#ifndef _GLOBALCONFIG_H
#define _GLOBALCONFIG_H

#include <string>
#include "../Header.h"


// ���ģʽ - ���� singleton
class GlobalConfig 
{
	// ˽�е���ľ�̬��Ա���������ڶ��������
	static GlobalConfig config;

	// ˽�й��캯�����������캯������ֵ������
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
	std::string pathRec;
	std::string pathTest;
	bool fileType; // true=feature, false=image
	int interval;

	//��ɫͼ����Ϊ��ɫͼ(1)��ת��Ϊ�Ҷ�ͼ(0)
	bool withGPS;
	
	float cx, cy, fx, fy;

	// config valid
	bool valid;
};

#endif