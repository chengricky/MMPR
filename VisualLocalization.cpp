#include "VisualLocalization.h"
#include "Tools/Timer.h"
#include <dirent.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
#include "SequenceSearch.h"
#include <iomanip>

using namespace std;

//#define GAsearch
//#define Sweepingsearch
#define TEST
//#define GPS_TEST

VisualLocalization::VisualLocalization(GlobalConfig& config) :
				retrievalRet(std::vector<std::vector<std::pair<int, float>>>())
{ 	
	if (!config.getValid())
	{
		throw invalid_argument("Configuration is invalid!");
	}
	std::cout << "====>Reading Database Features." << std::endl;
	featurebase = make_shared<DescriptorFromFile>(config, true);
	
	std::cout << "====>Reading Query Features." << std::endl;
	featurequery = make_shared<DescriptorFromFile>(config, false);	

	matRow = featurequery->getVolume();
	matCol = featurebase->getVolume();	

	// construct the FLANN database
	std::cout << "====>Constructing the FLANN database." << std::endl;
	cv::flann::KDTreeIndexParams kdIndex(5); //同时建立多个随机kd树，确定tree的数量
	searchDB = make_shared<cv::flann::Index>(featurebase->netVLADs, kdIndex); //默认为L2 distance	

	// sequence matching
	seqSch = make_shared<SequenceSearch>(cv::Size(matCol, matRow), config);
	sceneTopK = config.sceneTopK;

};


void VisualLocalization::getTopKRetrieval(const int& k)
{
	auto query = featurequery->netVLADs;
	auto ref = featurebase->netVLADs;
	auto qS = featurequery->clss;
	auto rS = featurebase->clss;
	
	if(ref.empty()||query.empty())
		retrievalRet = std::vector<std::vector<std::pair<int, float>>>();
	else
	{
		for(int i = 0; i < query.rows; i++)
		{
			std::vector<int> idx;
			std::vector<float> dist;
			searchDB->knnSearch((vector<float>)(query.row(i)), idx, dist, k, cv::flann::SearchParams());
			std::vector<std::pair<int, float>> topK;
			for(int j=0;j<k;j++)
			{
				if (sceneTopK>0)
				{
					bool hit = false;
					for (size_t m = 0; m < sceneTopK; m++)
					{
						for (size_t n = 0; n < sceneTopK; n++)
						{
							auto qcls = qS[i][m];
							auto rcls = rS[idx[j]][n];
							if(qcls==rcls)
							{
								hit = true;
								break;
							}
						}						
						if(hit)
							break;
					}
					if(hit)
						topK.push_back(std::make_pair(idx[j], dist[j]));
					
				}
				else
					topK.push_back(std::make_pair(idx[j], dist[j]));
			}
			retrievalRet.push_back(topK);	
		}		
	}
}


std::vector<int> VisualLocalization::getSeqMatch()
{	
	seqSch->coneSearch(retrievalRet);
	seqSch->windowedUniquenessThresholding();
	return seqSch->getRet();

}

