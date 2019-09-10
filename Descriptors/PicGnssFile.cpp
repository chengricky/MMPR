#include "PicGnssFile.h"
#include <dirent.h>

using namespace std;

PicGNSSFile::PicGNSSFile(std::string filepath, bool ifGNSS)
{
	init(filepath, ifGNSS );
}

void PicGNSSFile::init(std::string filepath, bool ifGNSS,  int interval)
{

	vector<string> suffix;
	suffix.push_back("png.txt");
	suffix.push_back("jpg.txt");
	// suffix.push_back("txt");

	DIR* dir = opendir(filepath.c_str());  //��ָ��Ŀ¼
	dirent* p = nullptr;  //�������ָ��
	while((p = readdir(dir)) != NULL)  //��ʼ�������
	{
		// ����"."��".."�����ļ�
		if(p->d_name[0] != '.')  //d_name��һ��char���飬��ŵ�ǰ���������ļ���  
		{  
			string strName = p->d_name;
			string postfix = strName.substr(strName.length()-7);
			bool flag = false;
			for (auto e:suffix)
			{
				if(e==postfix)
				{
					flag = true;
				}
			}
			featFiles.push_back(filepath+"/"+strName);
			

		}
	}
	closedir(dir);  //�ر�ָ��Ŀ¼  
	fileVolume = featFiles.size();

	// ���ݱ�־λ������GNSS������Ϣ
	readTxtGnssLabel(ifGNSS, filepath);
	filePointer = 0;

}

bool PicGNSSFile::doMainFeatureFile()
{
	if (filePointer < fileVolume)
	{
		std::ifstream featureFile(featFiles[filePointer]);
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

std::vector<std::string> splitWithStl(const std::string &str, const std::string &pattern)
{
	std::vector<std::string> resVec;

	if ("" == str)
	{
		return resVec;
	}
	//�����ȡ���һ������
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
