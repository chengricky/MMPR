#include "Descriptors/DscGnssFile.h"
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

	auto topk = vl.getTopKRetrieval(10);
	std::cout<<"==>Top-"<<10<<" Retrieval is Ready."<<std::endl;
	
	std::vector<int> ret;
	vl.getBestMatch(topk, ret);

	for (auto e : ret)
	{
		std::cout<<e<<std::endl;
	}
	

	//vl.showDistanceMatrix();

	timer.stop();
	std::cout << "Matching Time consumed: ";
	timer.print_elapsed_time(TimeExt::MSec);


	cv::waitKey(0);
	return 0;
}
