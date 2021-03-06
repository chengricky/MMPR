#include "SequenceSearch.h"

void SequenceSearch::trajectorySearch()
{
	int vmin, vmax, vstep;
	//vsteps
}

void SequenceSearch::coneSearch(const bool& biDirection, const bool& isOnlineMode)
{

	if (globalResult.empty())
	{
		scoreMat = cv::Mat(1, numDatabase, CV_32FC1, 0.0f);
		return;
	}
	scoreMat.create(1, numDatabase, CV_32FC1);

	int i = globalResult.size() - 1; //query
	float* pS = scoreMat.ptr<float>(0);
	for (int j = 0; j < numDatabase; j++)//database
	{
		int count = 0, min_y, max_y;				
		assert(isOnlineMode);	// on-line mode

		min_y = std::max(0, (int)i - (numSearch - 1) );
		max_y = i;

		for (int k = min_y; k <= max_y; k++)//query within coneopen
		{
			int min_x=-1, max_x=-1;
			if (k<i)
			{
				min_x = std::max(int((k - i)*vmax + j), 0);
				max_x =  (k - i)*vmin + j;
			}
			else
			{
				max_x = std::min(int((k - i)*vmax + j), numDatabase -1);
				min_x = (k - i)*vmin + j;
			}
			if (globalResult[k][0]>=min_x && globalResult[k][0]<=max_x)
				count++;
			else if(biDirection && globalResult[k][0] >= std::min(2 * j - max_x, numDatabase - 1) 
					&& globalResult[k][0] <= std::min(2 * j - min_x, numDatabase - 1))
				count++;
			if (globalResult[k][1]>=min_x && globalResult[k][1]<=max_x)
				count++;
			else if(biDirection && globalResult[k][1] >= std::min(2 * j - max_x, numDatabase - 1) 
					&& globalResult[k][1] <= std::min(2 * j - min_x, numDatabase - 1))
				count++;					
		}
		if (count)
		{
			pS[j] = float(count) / float(max_y- min_y+1);
		}
		else
		{
			pS[j] = 0;
		}
	}
	
}

