#pragma once

// OpenGA library
#include "genetic.hpp"
#include "Header.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "SequenceSearch.h"

struct Params
{
	float lambda[9] = 
	{ 1.24517777,	1.685089537,	
		1.578978596,	1.091489515,	0.987401397,
		0.526293315,	0.623399388,	0.840100646,
		1.422069836	};

	float vmax = 1 / 0.4, vmin = 0.4, numsequence = 10;
	float scoreTh = 0.16;
};

// Genetic Algorithm
struct MyGenes
{
	std::vector<double> x; //paramters to be optimized: lambda

	std::string to_string() const
	{
		std::ostringstream out;
		out << "{";
		for (unsigned long i = 0; i < x.size(); i++)
			out << (i ? "," : "") << std::setprecision(10) << x[i];
		out << "}";
		return out.str();
	}
};
struct MyMiddleCost
{
	// This is where the results of simulation
	// is stored but not yet finalized.
	double cost;
};

class Parameter2F1
{
	std::vector<std::vector<int>> gt;
	std::vector<int> pGlobal[9];
	Params parameters;
	std::vector<int> matchingResults; // -1 means empty, 1-based results
	cv::Size matSize;
	SequenceSearch pSS[9];
public:
	//Parameter2F1(){};
	Parameter2F1(std::vector<std::vector<int>> gt, std::vector<int> GGGlobalBest, std::vector<int> BoWGlobalBest_RGB,  std::vector<int> BoWGlobalBest_IR,
		std::vector<int> GISTGlobalBest_RGB, std::vector<int> GISTGlobalBest_D, std::vector<int> GISTGlobalBest_IR,
		std::vector<int> LDBGlobalBest_RGB, std::vector<int> LDBGlobalBest_D, std::vector<int> LDBGlobalBest_IR, cv::Size matSize) : gt(gt), matSize(matSize)
	{
		pGlobal[0] = BoWGlobalBest_RGB;
		pGlobal[1] = BoWGlobalBest_IR;
		pGlobal[2] = GISTGlobalBest_RGB;
		pGlobal[3] = GISTGlobalBest_D;
		pGlobal[4] = GISTGlobalBest_IR;
		pGlobal[5] = LDBGlobalBest_RGB;
		pGlobal[6] = LDBGlobalBest_D;
		pGlobal[7] = LDBGlobalBest_IR;
		pGlobal[8] = GGGlobalBest;
	};
	~Parameter2F1() {	};
	//update different parameters
	void updateParams(std::vector<double> x, int idx=-1, double val=0){
		if (idx>=0&&idx<9)
		{
			parameters.lambda[idx] = val;
			return;
		}
		for (size_t i = 0; i < 9; i++)
		{
			parameters.lambda[i] = x[i];
		}
	}
	void updateParams(float vmax, float vmin, float numseq)
	{
		parameters.vmax = vmax;
		parameters.vmin = vmin;
		parameters.numsequence = numseq;
	}
	void updateParams(float scoreTh)
	{
		parameters.scoreTh = scoreTh;
	}


	void prepare4MultimodalCoefficients();	
	// get matching results
	void placeRecognition();
	float placeRecognition4MultimodalCoefficients(const MyGenes& p);

	void printMatchingResults() {
		for (size_t i = 0; i < matchingResults.size(); i++)
		{
			std::cout << i << "..." << matchingResults[i] << std::endl;
		}
	}
	void saveMatchingResults() {
		std::ofstream of("result.txt");
		for (size_t i = 0; i < matchingResults.size(); i++)
		{
			of << matchingResults[i] << std::endl;
		}
	}
	std::vector<int> getMatchingResults() { return matchingResults; };

	//calculate F1 score according to groundtruth(gt) and PR results (matchingResults).
	float calculateF1score(float *p = nullptr, float* r = nullptr);
	float calculateErr();

	bool eval_genes(const MyGenes& p, MyMiddleCost &c);
};

typedef EA::Genetic<MyGenes, MyMiddleCost> GA_Type;
typedef EA::GenerationType<MyGenes, MyMiddleCost> Generation_Type;

// functions of GA 
void init_genes(MyGenes& p, const std::function<double(void)> &rand);

MyGenes mutate(const MyGenes& X_base, const std::function<double(void)> &rand, double shrink_scale);
MyGenes crossover(const MyGenes& X1, const MyGenes& X2, const std::function<double(void)> &rand);
double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);
void SO_report_generation(int generation_number, const EA::GenerationType<MyGenes, MyMiddleCost> &last_generation, const MyGenes& best_genes);

// execute optimization for parameters lambda
bool optimizeMultimodalCoefficients(Parameter2F1* pt, std::vector<double>& x);

