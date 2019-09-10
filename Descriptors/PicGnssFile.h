#ifndef PICGNSSFILE
#define PICGNSSFILE

#include "FileInterface.h"
#include <fstream>
#include <string>

class PicGNSSFile : public FileInterface
{
public:
	PicGNSSFile() {};
	PicGNSSFile(std::string filepath, bool ifGNSS);
	void init(std::string filepath, bool ifGNSS, int interval=1);
	//bool doMain();
	bool doMainFeatureFile();

	cv::Mat netVLAD;
	double latitudeValue;
	double longitudeValue;

	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };


private:
	std::vector<double> latitude; //Î³¶È
	std::vector<double> longitude; //¾­¶È
	unsigned long filePointer;
	unsigned long fileVolume;
	std::vector<std::string> featFiles;

	// Read GNSS.txt File
	bool readTxtGnssLabel(bool ifGNSS, std::string filepath);

};

#endif
