#include "Descriptors/DscGnssFile.h"
#include "VisualLocalization.h"
#include "Tools/GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "Header.h"
#include "Tools/Timer.h"
#include "ParameterTuning.h"



int main()
{
	std::cout<<"START!"<<std::endl;
	
	GlobalConfig config("Config.yaml");
	std::cout<<"==>Configuration File is Read."<<std::endl;
	
	VisualLocalization vl(config);
	vl.readDatabase();
	vl.readGroundTruth();
	vl.readQuery();

	while(true)
	{
		Timer timer;
		timer.start();

		auto flag = vl.localizeQuery();

		timer.stop();
		std::cout << "Matching Time consumed: ";
		timer.print_elapsed_time(TimeExt::MSec);
		if(!flag)
			break;
	}

	vl.getPerformance();

	cv::waitKey(0);
	return 0;
}
