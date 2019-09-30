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

	bool doMainFeatureFile();

	cv::Mat netVLAD;
	std::vector<cv::Point2f> mKPts;
	cv::Mat mDecs;
	double latitudeValue;
	double longitudeValue;

	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };


private:
	std::vector<double> latitude;
	std::vector<double> longitude;
	unsigned long filePointer;
	unsigned long fileVolume;
	std::vector<std::string> featFilesRGB;
	std::vector<std::string> featFilesIR;

	// Read GNSS.txt File
	bool readTxtGnssLabel(bool ifGNSS, std::string filepath);

};

#endif
