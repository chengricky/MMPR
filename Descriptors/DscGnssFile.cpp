#include "DscGnssFile.h"
#include <dirent.h>
#include "../Tools/cnpy.h"

using namespace std;

PicGNSSFile::PicGNSSFile(std::string filepath, bool ifGNSS)
{
	init(filepath, ifGNSS );
}

void PicGNSSFile::init(std::string filepath, bool ifGNSS,  int interval)
{
	string suffix_ir = "ir.npy";
	string suffix_rgb = "rgb.npy";

	DIR* dir = opendir((filepath).c_str());  //��ָ��Ŀ¼
	dirent* p = nullptr;  //�������ָ��
	int counter = 0;
	while((p = readdir(dir)) != NULL)  //��ʼ�������
	{
		// ����"."��".."�����ļ�
		if(p->d_name[0] != '.')  //d_name��һ��char���飬��ŵ�ǰ���������ļ���  
		{  
			string strName = p->d_name;
			string postfix = strName.substr(strName.length()-7);
			if(suffix_rgb==postfix&&counter%interval==0)
			{
				featFilesRGB.push_back(filepath+"/"+strName);
				counter++;	
			}
			else if(suffix_ir==postfix.substr(1)&&counter%interval==0)
			{
				featFilesIR.push_back(filepath+"/"+strName);
			}	
			
		}
	}
	closedir(dir);  //�ر�ָ��Ŀ¼  
	fileVolume = featFilesRGB.size();

	// ���ݱ�־λ������GNSS������Ϣ
	readTxtGnssLabel(ifGNSS, filepath);
	filePointer = 0;

}

bool PicGNSSFile::doMainFeatureFile()
{
	if (filePointer < fileVolume)
	{
		// ��ȡȫ��������
		cnpy::NpyArray arr = cnpy::npy_load(featFilesRGB[filePointer]);
		auto vecD = arr.as_vec<float>();
		netVLAD = cv::Mat(vecD).reshape(1, 1);

		// ��ȡ�ֲ���������������
		cnpy::NpyArray dcs = cnpy::npy_load(featFilesIR[filePointer]);
		auto vecDesc = dcs.as_vec<float>();
		mDecs = cv::Mat(vecDesc); // ά��ӦΪC, H, W
		mDecs.reshape(mDesc.size[0], -1).t();

		// for vgg16-conv3
		for (size_t h = 0; h < 60; h++)
		{
			for (size_t w = 0; w < 80; w++)
			{
				cv::KeyPoint kpt(h*4.0+2.0,w*4.0+2.0);
				mKPts.push_back(kpt);
			}
			
		}	


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
