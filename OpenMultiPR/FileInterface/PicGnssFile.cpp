#include "PicGnssFile.h"
#include <io.h>

using namespace std;

PicGNSSFile::PicGNSSFile(std::string filepath, int mode, bool ifGNSS, std::string fileKeyWord)
{
	init(filepath, mode, ifGNSS, fileKeyWord );
}

void PicGNSSFile::init(std::string filepath, int mode, bool ifGNSS, std::string fileKeyWord, int interval)
{
	this->mode = mode;
	vector<string> suffix;
	suffix.push_back("png");
	suffix.push_back("jpg");
	suffix.push_back("png.txt");
	suffix.push_back("jpg.txt");
	suffix.push_back("txt");
	//文件句柄
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	for (size_t i = 0; i < suffix.size(); i++)
	{			
		string searchPath = filepath + "\\*" + fileKeyWord + "." + suffix[i];
		if ((hFile = _findfirst(searchPath.c_str(), &fileinfo)) != -1)
		{
			int iii = 0;
			do
			{
				if (iii%interval!=0)
				{
					iii++;
					continue;
				}
				iii++;
				string filename(fileinfo.name);
				size_t pos = filename.find_first_of('c');
				string filePrefix = filename.substr(0U, pos);
				//检查除彩图之外其他图像或文件是否存在
				bool depth, ground;
				if (mode == RGB)
				{
					depth = false;
					ground = false;
					findFilesfromColor(filepath, filePrefix, suffix[i], depth, ground);
					colorFiles.push_back(filepath + "\\" + fileinfo.name);

				}
				else if (mode == RGBD)
				{
					depth = true;
					ground = false;
					findFilesfromColor(filepath, filePrefix, suffix[i], depth, ground);
					if (depth)
					{
						colorFiles.push_back(filepath + "\\" + fileinfo.name);
						depthFiles.push_back(filepath + "\\" + filePrefix + "depth." + suffix[i]);
					}
				}
				else if (mode == RGBDIR)
				{
					depth = true;
					ground = true;
					findFilesfromColor(filepath, filePrefix, suffix[i], depth, ground);
					if (depth && ground)
					{
						colorFiles.push_back(filepath + "\\" + fileinfo.name);
						depthFiles.push_back(filepath + "\\" + filePrefix + "depth." + suffix[i]);
						IRFiles.push_back(filepath + "\\" + filePrefix + "rightIR." + suffix[i]);
					}
				}
				
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
	}
	fileVolume = colorFiles.size();
	// 根据标志位，读入GNSS坐标信息
	readTxtGnssLabel(ifGNSS, filepath);
	filePointer = 0;

}

void PicGNSSFile::findFilesfromColor(string path, string prefix, string suffix, bool& depth, bool&ground)
{
	if (depth)
	{
		fstream fileTry(path + "\\" + prefix + "depth." + suffix);
		if (!fileTry)
		{
			depth = false;
		}
		fileTry.close();
	}
	if (ground)
	{
		fstream fileTry(path + "\\" + prefix + "rightIR." + suffix);
		if (!fileTry)
		{
			ground = false;
		}
		fileTry.close();
	}	
}

bool PicGNSSFile::doMain()
{
	if (filePointer<fileVolume)
	{		
		colorImg = cv::imread(colorFiles[filePointer], cv::IMREAD_COLOR);
		imshow("inColor", colorImg);
		if (!latitude.empty() && !longitude.empty())
		{
			latitudeValue = latitude[filePointer];
			longitudeValue = longitude[filePointer];
		}
		if (mode == PicGNSSFile::RGBDIR || mode == PicGNSSFile::RGBD)
		{
			depthImg = cv::imread(depthFiles[filePointer], cv::IMREAD_GRAYSCALE);
			imshow("inDepth", depthImg);
		}			
		else
		{
			depthImg = cv::Mat();
		}
		if (mode == PicGNSSFile::RGBDIR)
		{
			IRImg = cv::imread(IRFiles[filePointer], cv::IMREAD_GRAYSCALE);
			imshow("inIR", IRImg);
		}			
		else
		{
			IRImg = cv::Mat();
		}
		filePointer++;
		cv::waitKey(1);
		return true;
	}
	else
	{
		return false;
	}
	
}

bool PicGNSSFile::doMainFeatureFile()
{
	if (filePointer < fileVolume)
	{
		std::ifstream featureFile(colorFiles[filePointer]);
		if (featureFile.is_open())
		{
			if(!netVLAD.empty())
				netVLAD.release();
			while (!featureFile.eof())
			{
				float dim;
				featureFile >> dim;
				netVLAD.push_back(dim);
			}
		}
		netVLAD = netVLAD.reshape(1, 1);
		if (!latitude.empty() && !longitude.empty())
		{
			latitudeValue = latitude[filePointer];
			longitudeValue = longitude[filePointer];
		}
		filePointer++;
		cv::waitKey(1);
		return true;
	}
	else
	{
		return false;
	}

}

cv::Size PicGNSSFile::getImgSize()
{
	cv::Mat tmp = cv::imread(colorFiles[filePointer-1]);
	return cv::Size((tmp).cols, tmp.rows);
}

std::vector<std::string> splitWithStl(const std::string &str, const std::string &pattern)
{
	std::vector<std::string> resVec;

	if ("" == str)
	{
		return resVec;
	}
	//方便截取最后一段数据
	std::string strs = str + pattern;

	size_t pos = strs.find(pattern);
	size_t size = strs.size();

	while (pos != std::string::npos)
	{
		std::string x = strs.substr(0, pos);
		resVec.push_back(x);
		strs = strs.substr(pos + 1, size);
		pos = strs.find(pattern);
	}

	return resVec;
}

bool PicGNSSFile::readTxtGnssLabel(bool ifGNSS, std::string filepath)
{
	if (!ifGNSS)
	{
		return false;
	}
	std::ifstream txtGnssLabel(filepath + "\\of.txt", ios::in);
	if (!txtGnssLabel.is_open())
	{
		return false;
	}
	for (size_t i = 0; i < fileVolume; i++)
	{
		string str;
		getline(txtGnssLabel, str);
		std::vector<std::string> splitStr = splitWithStl(str, "\t");
		assert(splitStr.size() >= 3);
		latitude.push_back(atof(splitStr[1].data())); //string->char*->double
		longitude.push_back(atof(splitStr[2].data()));	
	}
	return true;
}
