#include "DesriptorFromFile.h"
#include "../Tools/Timer.h"
#include "../Tools/GNSSDistance.h"

DescriptorFromFile::DescriptorFromFile(GlobalConfig& config, bool isRefImage)
{
	if (isRefImage)
		picPath = config.pathRec;
	else
		picPath = config.pathTest;

	// ��ͼ���ļ��б��GNSS���Ƿ�ؼ������ݽ��и�ֵ
	pFeatReader= std::make_shared<FeatReader>(picPath, config.interval);

	while (pFeatReader->getNextFrame())
	{
		// save the descriptor of netVLAD
		// Timer timer;
		// timer.start();

		if(netVLADs.empty())
			netVLADs = pFeatReader->netVLAD;
		else
			netVLADs.push_back(pFeatReader->netVLAD);		

		clss.push_back(pFeatReader->cls);			
		
		// timer.stop();
		// std::cout << "Time consumed: ";
		// timer.print_elapsed_time(TimeExt::MSec);
	}
}
