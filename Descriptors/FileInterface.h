#ifndef _FILEINTERFACE_H
#define _FILEINTERFACE_H

#include <vector>
#include <string>
#include "../Header.h"

class FileInterface
{
public:
	FileInterface() {}
	~FileInterface() {}
	virtual void init(std::string folderPath, bool ifLabeled, int) = 0;

	//virtual bool doMain() = 0;
	virtual bool loadNextVideo() { return true; }

	//virtual bool readVideo() = 0;



	std::vector<cv::Point2d> vertexes;
	std::vector<int> frames;
};

#endif