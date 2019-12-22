#ifndef PICGNSSFILE
#define PICGNSSFILE

#include "../Header.h"
#include <fstream>
#include <string>

class FeatReader
{
public:
	// Initialization (collect file paths of images, features and GNSS data)
	FeatReader(std::string filepath, int interval=1);
	// Read the next frame data
	bool getNextFrame();
	// 存储单帧数据
	cv::Mat netVLAD;
	std::vector<int> cls;
	// 返回文件夹内的信息
	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };

private:
	// 存储文件夹内所有特征的vectors
	unsigned long filePointer;
	unsigned long fileVolume;
	std::vector<std::string> featFilesRGB, sceneFilesRGB;

};

#endif
