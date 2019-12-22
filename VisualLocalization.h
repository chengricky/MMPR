#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "Descriptors/DesriptorFromFile.h"
#include "SequenceSearch.h"


class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);

	// 使用flann检索knn结果
	void getTopKRetrieval(const int& k);

	// 使用序列匹配获得最终定位结果
	std::vector<int> getSeqMatch();

	// 获得视觉定位各阶段结果
	std::vector<std::vector<std::pair<int, float>>> getRetrievalRet(){return retrievalRet;};
	
private:
	// 训练集数据(保存记录的路径)
	std::shared_ptr<DescriptorFromFile> featurebase;

	// 测试集数据
	std::shared_ptr<DescriptorFromFile> featurequery;
	std::string descriptor;

	// 检索的结果, pair中是由index序号和距离组成
	std::vector<std::vector<std::pair<int, float>>> retrievalRet;
	int sceneTopK;

	/// get a distance matrix, which is as follows:	
	//   ----> database
	//  |
	//  |
	//  V
	//  query images	
	int matRow, matCol;
	
	std::shared_ptr<cv::flann::Index> searchDB; 

	std::shared_ptr<SequenceSearch> seqSch;

};

#endif