
#include "VisualLocalization.h"
#include "Descriptors/GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "Header.h"
#include "Tools/Timer.h"


GlobalConfig GlobalConfig::config("Config.yaml");

int main()
{
	std::cout<<"START!"<<std::endl;
	
	static GlobalConfig& config = GlobalConfig::instance();
	std::cout<<"==>Configuration File is Read."<<std::endl;
	
	VisualLocalization vl(config);
	
	Timer timer;
	timer.start();

	int k=5;
	vl.getTopKRetrieval(k);
	std::cout<<"==>Top-"<<k<<" Retrieval is Ready."<<std::endl;	
	
	auto ret = vl.getSeqMatch();
	std::cout<<"==>Sequential Matching is Ready."<<std::endl;

	timer.stop();
	std::cout << "Matching Time consumed: ";
	timer.print_elapsed_time(TimeExt::MSec);

	for (auto e : ret)
	{
		std::cout<<e<<std::endl;
	}
	


	cv::waitKey(0);
	return 0;
}
