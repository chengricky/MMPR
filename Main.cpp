#include "FileInterface\picGnssfile.h"
#include "descriptors/CS/CSOperation.h"
#include "VisualLocalization.h"
#include "FileInterface\GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "Header.h"
#include "Tools/Timer.h"



GlobalConfig GlobalConfig::config("Config.yaml");

int main()
{
	static GlobalConfig& config = GlobalConfig::instance();
	VisualLocalization vl(config);

	std::cout << "Matching Time consumed: ";
	Timer timer;
	timer.start();

	//vl.getBestMatch();
	vl.getBestMatch_FeatureFile();
	//vl.showDistanceMatrix();

	timer.stop();
	timer.print_elapsed_time(TimeExt::MSec);


	cv::waitKey(0);
	return 0;
}
