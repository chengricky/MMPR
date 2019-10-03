#include "Descriptors/DscGnssFile.h"
#include "VisualLocalization.h"
#include "Descriptors/GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "Header.h"
#include "Tools/Timer.h"
#include "ParameterTuning.h"

GlobalConfig GlobalConfig::config("Config.yaml");

int main()
{
	std::cout<<"START!"<<std::endl;
	
	static GlobalConfig& config = GlobalConfig::instance();
	std::cout<<"==>Configuration File is Read."<<std::endl;
	
	VisualLocalization vl(config);
	
	Timer timer;
	timer.start();

	vl.getTopKRetrieval(10);
	std::cout<<"==>Top-"<<10<<" Retrieval is Ready."<<std::endl;	
	
	vl.getBestGeomValid();
	std::cout<<"==>Best Match is Obatained from Geometric Validation."<<std::endl;

	vl.getSeqMatch();
	std::cout<<"==>Sequential Matching is Ready."<<std::endl;

	timer.stop();
	std::cout << "Matching Time consumed: ";
	timer.print_elapsed_time(TimeExt::MSec);


	cv::waitKey(0);
	return 0;
}
