#ifndef _VISUALLOCALIZATION_H
#define _VISUALLOCALIZATION_H

#include "Descriptors/DesriptorFromFile.h"
#include "SequenceSearch.h"


class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);

	// ʹ��flann����knn���
	void getTopKRetrieval(const int& k);

	// ʹ������ƥ�������ն�λ���
	std::vector<int> getSeqMatch();

	// ����Ӿ���λ���׶ν��
	std::vector<std::vector<std::pair<int, float>>> getRetrievalRet(){return retrievalRet;};
	
private:
	// ѵ��������(�����¼��·��)
	std::shared_ptr<DescriptorFromFile> featurebase;

	// ���Լ�����
	std::shared_ptr<DescriptorFromFile> featurequery;
	std::string descriptor;

	// �����Ľ��, pair������index��ź;������
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