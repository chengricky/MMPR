#pragma once
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
	cv::Size dImgSize;
	cv::Size qImgSize;
	std::string pathRec;
	std::string pathTest;
	std::string codeBook;
	bool fileType; // true=feature, false=image
	int interval;

	//��ɫͼ����Ϊ��ɫͼ(1)��ת��Ϊ�Ҷ�ͼ(0)
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

