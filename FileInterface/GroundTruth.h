#pragma once
#include "..\Header.h"

class GroundTruth
{
	std::vector<int> groundTruth;
	std::vector<bool> keyLabel;

public:
	GroundTruth() {};
	GroundTruth(std::string filepathGT, std::string filepathLabel);
	~GroundTruth() {};

	void init(std::string filepathGT, std::string filepathLabel);
	
	std::vector<std::vector<int>> gt;
	void generateGroundTruth(int Tolerence);
};
