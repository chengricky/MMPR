#include "DesriptorFromFile.h"
#include "../Tools/Timer.h"
#include "../Tools/GNSSDistance.h"

DescriptorFromFile::DescriptorFromFile(const GlobalConfig& config, bool isRefImage)
{
	if (isRefImage)
		picPath = config.pathRec;
	else
		picPath = config.pathTest;

	// 对图像文件列表和GNSS、是否关键点数据进行赋值
	picFiles.collectFileVec(picPath, config.withGPS, 
							config.cx, config.cy, config.fx, config.fy, config.interval);


	while (picFiles.getNextFrame())
	{
		/*GPS*/
		if (picFiles.frame.longitudeValue!=0 || picFiles.frame.latitudeValue!=0)
		{
			// ddmm.mmmm --> dd.dddddd (style conversion)
			cv::Mat GPS_row(1, 2, CV_32FC1);
			GPS_row.at<float>(0, 0) = ddmm2dd(picFiles.frame.longitudeValue);
			GPS_row.at<float>(0, 1) = ddmm2dd(picFiles.frame.latitudeValue);
			GPS.push_back(GPS_row);
		}


		// save the descriptor of netVLAD
		// Timer timer;
		// timer.start();

		if(netVLADs.empty())
			netVLADs = picFiles.frame.netVLAD;
		else
			netVLADs.push_back(picFiles.frame.netVLAD);					
		
		descs.push_back(picFiles.frame.mDecs);
		kpts.push_back(picFiles.frame.mKPts);
		
		pt3dNorms.push_back( picFiles.frame.ptNorm );



		// timer.stop();
		// std::cout << "Time consumed: ";
		// timer.print_elapsed_time(TimeExt::MSec);


	}

}


PartialDescriptorsFromFile::PartialDescriptorsFromFile(const GlobalConfig& config)
{
	picPath = config.pathTest;

	// 对图像文件列表和GNSS、是否关键点数据进行赋值
	picFiles = std::make_shared<PicGNSSFile>(picPath, config.withGPS, 
				config.cx, config.cy, config.fx, config.fy, config.interval);

	queryVolume = config.num_queue;	

}

bool PartialDescriptorsFromFile::getFrame()
{
	bool ret = picFiles->getNextFrame();
	if (!ret)
		return ret;
	else
	{
		queryIdx = picFiles->getFilePointer();
		/*GPS*/
		if (picFiles->frame.longitudeValue!=0 || picFiles->frame.latitudeValue!=0)
		{
			// ddmm.mmmm --> dd.dddddd (style conversion)
			cv::Mat GPS_row(1, 2, CV_32FC1);
			GPS_row.at<float>(0, 0) = ddmm2dd(picFiles->frame.longitudeValue);
			GPS_row.at<float>(0, 1) = ddmm2dd(picFiles->frame.latitudeValue);
			GPS.push_back(GPS_row);
			if (GPS.size() > queryVolume)
				GPS.erase(GPS.begin());
		}

		// save the descriptor of netVLAD
		// Timer timer;
		// timer.start();

		netVLADs.push_back(picFiles->frame.netVLAD);	
		if(netVLADs.size() > queryVolume)
			netVLADs.erase(netVLADs.begin());							
		
		descs.push_back(picFiles->frame.mDecs);
		if(descs.size() > queryVolume)
			descs.erase(descs.begin());
		
		kpts.push_back(picFiles->frame.mKPts);
		if(kpts.size() > queryVolume)
			kpts.erase(kpts.begin());
		
		pt3dNorms.push_back( picFiles->frame.ptNorm );
		if(pt3dNorms.size() > queryVolume)
			pt3dNorms.erase(pt3dNorms.begin());
			
		// timer.stop();
		// std::cout << "Time consumed: ";
		// timer.print_elapsed_time(TimeExt::MSec);
		return true;

	}
}
