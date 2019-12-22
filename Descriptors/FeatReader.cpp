#include "FeatReader.h"
#include <dirent.h>
#include <string>
#include "cnpy.h"


using namespace std;

// int computeNormalsPC3d(const cv::Mat& PC, cv::Mat& PCNormals, const int NumNeighbors, 
// 						const bool FlipViewpoint, const cv::Vec3f& viewpoint);

FeatReader::FeatReader(std::string filepath, int interval)
{
	string suffix_feat = "npy";
	string suffix_scene = "csv";

	DIR* dir = opendir((filepath).c_str());  //��ָ��Ŀ¼
	dirent* p = nullptr;  //�������ָ��
	int counter = 0;
	while((p = readdir(dir)) != NULL)  //��ʼ�������
	{
		// ����"."��".."�����ļ�
		if(p->d_name[0] != '.')  //d_name��һ��char���飬��ŵ�ǰ���������ļ���  
		{  
			string strName = p->d_name;
			if(strName.length()<7)
				continue;
			string postfix = strName.substr(strName.length()-3);
			if(suffix_feat==postfix && counter%interval==0)
			{
				featFilesRGB.push_back(filepath+"/"+strName);
				counter++;	
			}
			else if(suffix_scene==postfix && counter%interval==0)
			{
				sceneFilesRGB.push_back(filepath+"/"+strName);
				counter++;					
			}
		}
	}
	closedir(dir);  //�ر�ָ��Ŀ¼ 
	
	fileVolume = featFilesRGB.size();
	sort(featFilesRGB.begin(), featFilesRGB.end());

	filePointer = 0;
}

bool FeatReader::getNextFrame()
{
	if (filePointer < fileVolume)
	{		
		std::cout<<"Reading Frame "<<filePointer<<std::endl;

		// ��ȡNetVLADȫ��������
		cnpy::NpyArray arr = cnpy::npy_load(featFilesRGB[filePointer]);
		auto vecD = arr.as_vec<float>();
		netVLAD = cv::Mat(vecD).t(); //size ~ 1*30k for vgg-based netvlad	

		std::ifstream infile(sceneFilesRGB[filePointer]);
    	for (std::string line; std::getline(infile, line); ) 
		{
        	auto startpos = line.find(',');
			auto endpos = line.find(',', startpos+1);
			auto c = line.substr(startpos+1,endpos-startpos);
			cls.push_back(atoi(c.c_str()));
    	}		

		filePointer++;
		return true;
	}
	else
	{
		return false;
	}

}
