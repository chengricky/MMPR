#include "DscGnssFile.h"
#include <dirent.h>
#include "cnpy.h"
#include <opencv2/surface_matching.hpp>
#include <opencv2/surface_matching/ppf_helpers.hpp>

using namespace std;

// int computeNormalsPC3d(const cv::Mat& PC, cv::Mat& PCNormals, const int NumNeighbors, 
// 						const bool FlipViewpoint, const cv::Vec3f& viewpoint);

void PicGNSSFile::collectFileVec(std::string filepath, bool ifGNSS,
								float cx, float cy, float fx, float fy, int interval)
{
	string suffix_ir = "ir.npy";
	string suffix_rgb = "rgb.npy";
	string suffix_depth = "pth.png";

	DIR* dir = opendir((filepath).c_str());  //打开指定目录
	dirent* p = nullptr;  //定义遍历指针
	int counter = 0;
	while((p = readdir(dir)) != NULL)  //开始逐个遍历
	{
		// 过滤"."和".."隐藏文件
		if(p->d_name[0] != '.')  //d_name是一个char数组，存放当前遍历到的文件名  
		{  
			string strName = p->d_name;
			if(strName.length()<7)
				continue;
			string postfix = strName.substr(strName.length()-7);
			if(suffix_rgb==postfix && counter%interval==0)
			{
				featFilesRGB.push_back(filepath+"/"+strName);
				counter++;	
			}
			else if(suffix_ir==postfix.substr(1) && counter%interval==0)
				featFilesIR.push_back(filepath+"/"+strName);
			else if(suffix_depth==postfix && counter%interval==0)
				imgFilesDepth.push_back(filepath+"/"+strName);
		}
	}
	closedir(dir);  //关闭指定目录 
	
	fileVolume = featFilesRGB.size();
	sort(featFilesRGB.begin(), featFilesRGB.end());
	sort(featFilesIR.begin(), featFilesIR.end());
	sort(imgFilesDepth.begin(), imgFilesDepth.end());

	this->cx = cx;
	this->cy = cy;
	this->fx = fx;
	this->fy = fy;

	// 根据标志位，读入GNSS坐标信息
	readTxtGnssLabel(ifGNSS, filepath);
	filePointer = 0;
}

bool PicGNSSFile::getNextFrame()
{
	if (filePointer < fileVolume)
	{		
		std::cout<<"Reading Frame "<<filePointer<<std::endl;
		// 读取NetVLAD全局描述子
		cnpy::NpyArray arr = cnpy::npy_load(featFilesRGB[filePointer]);
		auto vecD = arr.as_vec<float>();
		netVLAD = cv::Mat(vecD).t(); //size~1*30k
		

		// 读取局部描述子生成特征
		cnpy::NpyArray dcs = cnpy::npy_load(featFilesIR[filePointer]);// 维度应为C, H, W		
		mDecs = cv::Mat( dcs.shape[0], dcs.shape[2]*dcs.shape[1], CV_32FC1, dcs.data<float>()).t(); // rows,cols
		assert(dcs.shape[2]*dcs.shape[1]==4800);
		
		// get local key points for vgg16-conv3
		std::vector<cv::Point2f>().swap(mKPts);
		for (int h = 0; h < 60; h++)
			for (int w = 0; w < 80; w++)
				mKPts.push_back(cv::Point2f(w*4.0f+2.0f, h*4.0f+2.0f));			//x,y			

		cv::Mat imgD = cv::imread(imgFilesDepth[filePointer], cv::IMREAD_ANYDEPTH);
		// 计算法向量
		cv::Mat pointCloud, pointNormal;
		std::vector<bool> remain;
		for (size_t i = 0; i < imgD.rows; i++)
		{
			for (size_t j = 0; j < imgD.cols; j++)
			{				
				float z = imgD.at<ushort>(i, j);
				if( z < 10 || z > 65500)
					continue;
				cv::Mat point3D = cv::Mat(1, 3, CV_32FC1, cv::Scalar(0));
				point3D.at<float>(0, 0) = (j-cx)*z/fx;
				point3D.at<float>(0, 1) = (i-cy)*z/fy;
				point3D.at<float>(0, 2) = z;
				pointCloud.push_back(point3D);
				if((i-2)%4==0 && (j-2)%4==0)
				{						
					remain.push_back(true);
				}					
				else
					remain.push_back(false);				
			}
		}

		// cv::Vec2f xr ,yr ,zr;
		// cv::ppf_match_3d::computeBboxStd(pointCloud, xr, yr, zr);
		// std::cout<<xr<<" "<<yr<<" "<<zr<<std::endl;
		cv::ppf_match_3d::computeNormalsPC3d(pointCloud, pointNormal, 6, false, cv::Vec3f(0, 0, 0));
		
		// ptNorm = pointNormal;
		if(!ptNorm.empty())
			ptNorm.release();
		for (size_t i = 0; i < remain.size(); i++)
		{
			if(remain[i])
			{
				if(ptNorm.empty())
					ptNorm = pointNormal.row(i);
				else
					ptNorm.push_back(pointNormal.row(i));
			}				
		}
		
		if (!latitude.empty() && !longitude.empty())
		{
			latitudeValue = latitude[filePointer];
			longitudeValue = longitude[filePointer];
		}

		filePointer++;
		return true;
	}
	else
	{
		return false;
	}

}

std::vector<std::string> PicGNSSFile::splitWithStl(const std::string &str, const std::string &pattern)
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
	std::ifstream txtGnssLabel(filepath + "/of.txt", ios::in);
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

// using namespace cv;

// typedef cv::flann::L2<float> Distance_32F;
// typedef cv::flann::GenericIndex< Distance_32F > FlannIndex;

// void* indexPCFlann(Mat pc)
// {
//   Mat dest_32f;
//   pc.colRange(0,3).copyTo(dest_32f);
//   return new FlannIndex(dest_32f, cvflann::KDTreeSingleIndexParams(8));
// }

// // For speed purposes this function assumes that PC, Indices and Distances are created with continuous structures
// void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, const int numNeighbors)
// {
//   Mat obj_32f;
//   pc.colRange(0, 3).copyTo(obj_32f);
//   ((FlannIndex*)flannIndex)->knnSearch(obj_32f, indices, distances, numNeighbors, cvflann::SearchParams(32));
// }
// void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances)
// {
//   queryPCFlann(flannIndex, pc, indices, distances, 1);
// }

// void destroyFlann(void* flannIndex)
// {
//   delete ((FlannIndex*)flannIndex);
// }

// void meanCovLocalPCInd(const Mat& pc, const int* Indices, const int point_count, Matx33d& CovMat, Vec3d& Mean)
// {
//   int i, j, k;

//   CovMat = Matx33d::all(0);
//   Mean = Vec3d::all(0);
//   for (i = 0; i < point_count; ++i)
//   {
//     const float* cloud = pc.ptr<float>(Indices[i]);
//     for (j = 0; j < 3; ++j)
//     {
//       for (k = 0; k < 3; ++k)
//         CovMat(j, k) += cloud[j] * cloud[k];
//       Mean[j] += cloud[j];
//     }
//   }
//   Mean *= 1.0 / point_count;
//   CovMat *= 1.0 / point_count;

//   for (j = 0; j < 3; ++j)
//     for (k = 0; k < 3; ++k)
//       CovMat(j, k) -= Mean[j] * Mean[k];
// }

// /**
//  *  \brief Flip a normal to the viewing direction
//  *
//  *  \param [in] point Scene point
//  *  \param [in] vp view direction
//  *  \param [in] n normal
//  */
// static inline void flipNormalViewpoint(const Vec3f& point, const Vec3f& vp, Vec3f& n)
// {
//   float cos_theta;

//   // See if we need to flip any plane normals
//   Vec3f diff = vp - point;

//   // Dot product between the (viewpoint - point) and the plane normal
//   cos_theta = diff.dot(n);

//   // Flip the plane normal
//   if (cos_theta < 0)
//   {
//     n *= -1;
//   }
// }

// int computeNormalsPC3d(const Mat& PC, Mat& PCNormals, const int NumNeighbors, const bool FlipViewpoint, const Vec3f& viewpoint)
// {
//   int i;

//   if (PC.cols!=3 && PC.cols!=6) // 3d data is expected
//   {
//     //return -1;
//     CV_Error(cv::Error::BadImageSize, "PC should have 3 or 6 elements in its columns");
//   }

//   PCNormals.create(PC.rows, 6, CV_32F);
//   Mat PCInput = PCNormals.colRange(0, 3);
//   Mat Distances(PC.rows, NumNeighbors, CV_32F);
//   Mat Indices(PC.rows, NumNeighbors, CV_32S);

//   PC.rowRange(0, PC.rows).colRange(0, 3).copyTo(PCNormals.rowRange(0, PC.rows).colRange(0, 3));

//   void* flannIndex = indexPCFlann(PCInput);
//   queryPCFlann(flannIndex, PCInput, Indices, Distances, NumNeighbors);
//   destroyFlann(flannIndex);
//   flannIndex = 0;

// #if defined _OPENMP
// #pragma omp parallel for
// #endif
//   for (i=0; i<PC.rows; i++)
//   {
//     Matx33d C;
//     Vec3d mu;
//     const int* indLocal = Indices.ptr<int>(i);

//     // compute covariance matrix
//     meanCovLocalPCInd(PCNormals, indLocal, NumNeighbors, C, mu);

//     // eigenvectors of covariance matrix
//     Mat eigVect, eigVal;
//     eigen(C, eigVal, eigVect);
//     eigVect.row(2).convertTo(PCNormals.row(i).colRange(3, 6), CV_32F);

//     if (FlipViewpoint)
//     {
//       Vec3f nr(PCNormals.ptr<float>(i) + 3);
//       Vec3f pci(PCNormals.ptr<float>(i));
//       flipNormalViewpoint(pci, viewpoint, nr);
//       Mat(nr).reshape(1, 1).copyTo(PCNormals.row(i).colRange(3, 6));
//     }
//   }

//   return 1;
// }
