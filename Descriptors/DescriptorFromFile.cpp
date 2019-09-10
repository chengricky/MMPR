#include "DesriptorFromFile.h"
#include "../Tools/Timer.h"
//#include "../Tools/list_dir.h"
#include "../Tools/GNSSDistance.h"

DescriptorFromFile::DescriptorFromFile(GlobalConfig& config, bool isRefImage)
{
	if (isRefImage)
		picPath = config.pathRec;
	else
		picPath = config.pathTest;

	// ��ͼ���ļ��б��GNSS���Ƿ�ؼ������ݽ��и�ֵ
	picFiles.init(picPath, config.withGPS, config.interval);

	while (picFiles.doMainFeatureFile())
	{
		/*GPS*/
		if (picFiles.longitudeValue!=0 || picFiles.latitudeValue!=0)
		{
			// ddmm.mmmm --> dd.dddddd (style conversion)
			cv::Mat GPS_row(1, 2, CV_32FC1);
			GPS_row.at<float>(0, 0) = ddmm2dd(picFiles.longitudeValue);
			GPS_row.at<float>(0, 1) = ddmm2dd(picFiles.latitudeValue);
			GPS.push_back(GPS_row);
		}


		// save the descriptor of netVLAD
		Timer timer;
		timer.start();

		netVLADs.push_back(picFiles.netVLAD);

		timer.stop();
		std::cout << "Time consumed: ";
		timer.print_elapsed_time(TimeExt::MSec);


	}

}
