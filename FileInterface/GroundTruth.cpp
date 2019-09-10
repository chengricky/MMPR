#include "GroundTruth.h"
#include <fstream>

static std::vector<std::string> splitWithStl(const std::string &str, const std::string &pattern)
{
	std::vector<std::string> resVec;

	if ("" == str)
	{
		return resVec;
	}
	//方便截取最后一段数据
	std::string strs = str + pattern;

	size_t pos = strs.find(pattern);
	size_t size = strs.size();

	while (pos != std::string::npos)
	{
		std::string x = strs.substr(0, pos);
		resVec.push_back(x);
		strs = strs.substr(pos + 1, size);
		pos = strs.find(pattern);
	}

	return resVec;
}

void GroundTruth::init(std::string filepathGT, std::string filepathLabel)
{
	std::ifstream fileGT(filepathGT, std::ios::in);
	if (fileGT.is_open())
	{
		std::string line;
		while (getline(fileGT, line))
		{
			std::vector<std::string> splitStr = splitWithStl(line, "\t");
			assert(splitStr.size() >= 7);
			groundTruth.push_back(atoi(splitStr[6].data())); //string->char*->int
		}
	}
	fileGT.close();
	std::ifstream fileLabel(filepathLabel, std::ios::in);
	if (fileLabel.is_open())
	{
		std::string line;
		while (getline(fileLabel, line))
		{
			std::vector<std::string> splitStr = splitWithStl(line, "\t");
			assert(splitStr.size() >= 6);
			keyLabel.push_back(atoi(splitStr[5].data())); //string->char*->int
		}
	}
	fileLabel.close();
}

GroundTruth::GroundTruth(std::string filepathGT, std::string filepathLabel)
{
	init(filepathGT, filepathLabel);
}

void GroundTruth::generateGroundTruth(int tolerence)
{
	for (auto elem : groundTruth)//1-based
	{
		std::vector<int> gt_element;
		if (elem==-1)
		{
			gt.push_back(gt_element);
			continue;
		}		
		int min = std::max(1, elem - tolerence);
		int max = std::min((int)keyLabel.size(), elem + tolerence);
		for (size_t i = min; i <= max; i++)
		{
			gt_element.push_back(i);
		}
		int min_i = min - 1;
		if (keyLabel[min_i])
		{
			min_i -= 1;
			while (min_i>=0 && keyLabel[min_i])
			{
				gt_element.push_back(min_i);
				min_i -= 1;
			}
		}
		int max_i = max - 1;
		if (keyLabel[max_i])
		{
			max_i += 1;
			while (max_i< (int)keyLabel.size() && keyLabel[max_i])
			{
				gt_element.push_back(max_i);
				max_i += 1;
			}
		}
		gt.push_back(gt_element);
	}
}