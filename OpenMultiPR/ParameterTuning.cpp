#include "ParameterTuning.h"
#include "SequenceSearch.h"
#include <cmath>
#include <algorithm>

// This library is free and distributed under
// Mozilla Public License Version 2.0.

void init_genes(MyGenes& p, const std::function<double(void)> &rand)
{
	for (int i = 0; i < 9; i++) // set the size of init
		p.x.push_back(rand());
}

bool Parameter2F1::eval_genes(	const MyGenes& p,	MyMiddleCost &c)
{
	c.cost = -placeRecognition4MultimodalCoefficients(p);
	//std::cout << c.cost << std::endl;
	return true;
}

MyGenes mutate(	const MyGenes& X_base,	const std::function<double(void)> &rand,	double shrink_scale)
{
	MyGenes X_new;
	double loca_scale = shrink_scale;//default=1
	if (rand() < 0.4)
		loca_scale *= loca_scale;
	else if (rand() < 0.1)
		loca_scale = 1.0;
	bool out_of_range;
	do {
		out_of_range = false;
		X_new = X_base;

		for (unsigned long i = 0; i < X_new.x.size(); i++)
		{
			double mu = 1*loca_scale;
			X_new.x[i] += mu * (rand() - rand());
			if (X_new.x[i] > 1 || X_new.x[i] < 0)
				out_of_range = true;
		}
	} while (out_of_range);
	return X_new;
}

MyGenes crossover(	const MyGenes& X1,	const MyGenes& X2,	const std::function<double(void)> &rand)
{
	MyGenes X_new;
	for (unsigned long i = 0; i < X1.x.size(); i++)
	{
		double r = rand();
		X_new.x.push_back(r*X1.x[i] + (1.0 - r)*X2.x[i]);
	}
	return X_new;
}

double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
	// finalize the cost
	return X.middle_costs.cost;
}

void SO_report_generation(	int generation_number,	const EA::GenerationType<MyGenes, MyMiddleCost> &last_generation,
	const MyGenes& best_genes)
{
	std::cout
		<< "Generation [" << generation_number << "], "
		<< "Best=" << last_generation.best_total_cost << ", "
		<< "Average=" << last_generation.average_cost << ", "
		<< "Best genes=(" << best_genes.to_string() << ")" << ", "
		<< "Exe_time=" << last_generation.exe_time
		<< std::endl;

	std::cout
		<<"generation_number: " << generation_number << "\t"
		<<"average_cost: " <<last_generation.average_cost << "\t"
		<<"best_total_cost: " <<last_generation.best_total_cost << "\n"
		<< best_genes.x[0] << "\t"
		<< best_genes.x[1] << "\t"
		<< best_genes.x[2] << "\n"
		<< best_genes.x[3] << "\t"
		<< best_genes.x[4] << "\t"
		<< best_genes.x[5] << "\n"
		<< best_genes.x[6] << "\t"
		<< best_genes.x[7] << "\t"
		<< best_genes.x[8] << "\t"
		<< "\n";
}

bool optimizeMultimodalCoefficients(Parameter2F1* pt, std::vector<double>& x)
{
	EA::Chronometer timer;
	timer.tic();

	std::vector<std::vector<double>> xVec;
	std::vector<float> f1Vec;
	for (int i = 0; i < 15; i++)
	{
		GA_Type ga_obj;
		ga_obj.problem_mode = EA::GA_MODE::SOGA;
		ga_obj.multi_threading = true;
		ga_obj.dynamic_threading = false;
		ga_obj.idle_delay_us = 0; // switch between threads quickly
		ga_obj.verbose = false;
		ga_obj.population = 500;
		ga_obj.generation_max = 80;
		ga_obj.calculate_SO_total_fitness = calculate_SO_total_fitness;
		ga_obj.init_genes = init_genes;
		ga_obj.eval_genes = std::bind(&Parameter2F1::eval_genes, pt, std::placeholders::_1, std::placeholders::_2);
		ga_obj.mutate = mutate;
		ga_obj.crossover = crossover;
		ga_obj.SO_report_generation = SO_report_generation;
		ga_obj.best_stall_max = 100;
		ga_obj.average_stall_max = 100;
		ga_obj.tol_stall_best = -1;
		ga_obj.tol_stall_average = -1;
		ga_obj.elite_count = 50;
		ga_obj.crossover_fraction = 0.7;
		ga_obj.mutation_rate = 0.3;

		std::cout << "STOP Reason: " << ga_obj.stop_reason_to_string(ga_obj.solve()) << std::endl;
		xVec.push_back(ga_obj.last_generation.chromosomes[ga_obj.last_generation.best_chromosome_index].genes.x);
		f1Vec.push_back(-ga_obj.last_generation.chromosomes[ga_obj.last_generation.best_chromosome_index].middle_costs.cost);
	}
	auto iter_max = std::max_element(f1Vec.begin(), f1Vec.end());
	x = *(xVec.begin() + int(iter_max - f1Vec.begin()));
	std::cout << "Optimized Coeffs\t";
	for (auto e : x)
	{
		std::cout << e<<" ";
	}
	std::cout << "\nf1:  " << *(f1Vec.begin() + int(iter_max - f1Vec.begin()));
	std::cout << "\nThe problem is optimized in " << timer.toc() << " seconds." << std::endl;
	for (size_t i = 0; i < f1Vec.size(); i++)
	{
		std::cout << "\nf1:  " << f1Vec[i]<<"\n";
		std::cout << "Optimized Coeffs\t";
		for (auto e : xVec[i])
		{
			std::cout << e << " ";
		}
	}
	return true;
}



static bool getScoreMap(cv::Mat DistanceMat)
{
	cv::Mat CS_norm;
	cv::normalize(DistanceMat, CS_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::resize(CS_norm, CS_norm, cv::Size(CS_norm.cols , CS_norm.rows ));
	//cv::destroyWindow("synScoreMat");
	cv::imshow("synScoreMat", CS_norm);
	return true;
}

void Parameter2F1::placeRecognition()
{
	if (!matchingResults.empty())
	{
		matchingResults.clear();
		std::vector<int>().swap(matchingResults);
	}
	//SequenceSearch* pSS = new SequenceSearch[9];
	cv::Mat synScoreMat(matSize, CV_32FC1);
	synScoreMat.setTo(0);
	float sum = 0;
	for (size_t i = 0; i < 9; i++)
	{
		pSS[i].init((pGlobal[i]), matSize, parameters.numsequence, parameters.vmin, parameters.vmax);
		pSS[i].coneSearch();
		if (!pSS[i].scoreMat.empty())
		{
			synScoreMat += parameters.lambda[i] * (pSS[i].scoreMat);
			sum += parameters.lambda[i];
		}
	}
	synScoreMat = synScoreMat / sum;
	getScoreMap( synScoreMat);
	cv::waitKey(1);
	for (size_t i = 0; i < synScoreMat.rows; i++)//query
	{
		double maxVal;
		int* maxPos = new int[2];
		cv::minMaxIdx(synScoreMat.row(i), nullptr, &maxVal, nullptr, maxPos);

		//if (maxVal > parameters.scoreTh)
		//{
		//	matchingResults.push_back(maxPos[1] + 1);
		//}
		//else
		//{
		//	matchingResults.push_back(-1);
		//}
		//delete maxPos;
		
		//  π”√Windowed Uniqueness Thresholding
		int window = 10.0;
		double ratioTh = 1.1;
		int a = std::max(0.0f, maxPos[1] - std::round(window / 2.0f));
		int b = std::min(synScoreMat.cols-1, maxPos[1] + (int)std::round(window / 2));
		cv::Mat synScoreMask = cv::Mat::ones(1, synScoreMat.cols, synScoreMat.type());
		for (size_t i = 0; i < synScoreMat.cols; i++)
		{
			if (i >= a && i <= b)
			{
				synScoreMask.at<float>(0, i) = 0;
			}
		}
		synScoreMask = synScoreMat.row(i).mul(synScoreMask);
		double maxVal_;
		int* maxPos_ = new int[2];
		cv::minMaxIdx(synScoreMask, nullptr, &maxVal_, nullptr, maxPos_);
		double ratio = maxVal / maxVal_;

		if (ratio > ratioTh)
		{
			matchingResults.push_back(maxPos[1] + 1);
		}
		else
		{
			matchingResults.push_back(-1);
		}
		delete maxPos;

	}
	//delete pSS;
	//matchingResults.clear();
	//std::vector<int>().swap(matchingResults);
	//std::fstream fin("T3-T3_GG_flow.txt", std::ios::in);
	//while (!fin.eof()) {
	//	int value;
	//	fin >> value >> value;
	//	matchingResults.push_back(value);
	//}
}

void Parameter2F1::prepare4MultimodalCoefficients()
{
	for (size_t i = 0; i < 9; i++)
	{
		pSS[i].init((pGlobal[i]), matSize, parameters.numsequence, parameters.vmin, parameters.vmax);
		pSS[i].coneSearch();
	}
}

float Parameter2F1::placeRecognition4MultimodalCoefficients(const MyGenes& p)
{
	std::vector<int> matchingResults;
	cv::Mat synScoreMat(matSize, CV_32FC1);
	synScoreMat.setTo(0);
	float sum = 0;
	for (size_t i = 0; i < 9; i++)
	{
		if (!pSS[i].scoreMat.empty())
		{
			synScoreMat += p.x[i] * (pSS[i].scoreMat);
			sum += p.x[i];
		}
	}
	synScoreMat = synScoreMat / sum;
	getScoreMap(synScoreMat);
	cv::waitKey(1);

	for (size_t i = 0; i < synScoreMat.rows; i++)//query
	{
		double maxVal;
		int* maxPos = new int[2];
		cv::minMaxIdx(synScoreMat.row(i), nullptr, &maxVal, nullptr, maxPos);
		if (maxVal > parameters.scoreTh)
		{
			matchingResults.push_back(maxPos[1] + 1);
		}
		else
		{
			matchingResults.push_back(-1);
		}
		delete maxPos;
	}
	//F1 score
	int fp = 0, tp = 0, tn = 0, fn = 0;
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (gt[i].empty()) // ground truth is empty 
		{
			if (matchingResults[i] == -1)
			{
				tn++;
			}
			else
			{
				fp++;
			}
		}
		else  // ground truth is not empty
		{
			if (matchingResults[i] == -1)
			{
				fn++;
			}
			else
			{
				std::vector<int>::iterator ret = std::find(gt[i].begin(), gt[i].end(), matchingResults[i]);
				if (ret == gt[i].end()) // no findings
				{
					fp++;
				}
				else
				{
					tp++;
				}
			}

		}
	}
	float pr = (float)tp / (tp + fp);
	float re = (float)tp / (tp + fn);
	float f1 = 2 * (pr*re) / (pr + re);
	return f1;
}

float Parameter2F1::calculateF1score(float* precision, float* recall)
{
	cv::Mat plotPR(matSize.height, matSize.width, CV_8UC3, cv::Scalar(255, 255, 255));
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (!gt[i].empty())
		{
			for (auto var : gt[i])
			{
				plotPR.at<cv::Vec3b>(i, var-1) = cv::Vec3b(0,255,0);
			}			
		}
	}
	for (size_t i = 0; i < matchingResults.size(); i++)
	{
		if (matchingResults[i] != -1)
			plotPR.at<cv::Vec3b>(i, matchingResults[i]-1) = cv::Vec3b(0, 0, 255);
	}
	cv::resize(plotPR, plotPR, matSize *2);
	cv::imshow("plotPR", plotPR);
	cv::waitKey(1);
	int fp = 0, tp = 0, tn = 0, fn = 0;
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (gt[i].empty()) // ground truth is empty 
		{
			if (matchingResults[i] == -1)
			{
				tn++;
			}
			else
			{
				fp++;
			}
		}
		else  // ground truth is not empty
		{
			if (matchingResults[i]==-1)
			{
				fn++;
			}
			else
			{
				std::vector<int>::iterator ret = std::find(gt[i].begin(), gt[i].end(), matchingResults[i]);
				if (ret == gt[i].end()) // no findings
				{
					fp++;
				}
				else
				{
					tp++;
				}
			}

		}
	}
	float p = (float)tp / (tp + fp);
	float r = (float)tp / (tp + fn);
	float f1 = 2 * (p*r) / (p + r);
	if (precision!=nullptr)
	{
		*precision = p;
	}
	if (recall!=nullptr)
	{
		*recall = r;
	}
	return f1;
}


float Parameter2F1::calculateErr()
{
	cv::Mat plotPR(matSize.height, matSize.width, CV_8UC3, cv::Scalar(255, 255, 255));
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (!gt[i].empty())
		{
			for (auto var : gt[i])
			{
				plotPR.at<cv::Vec3b>(i, var - 1) = cv::Vec3b(0, 255, 0);
			}
		}
	}
	for (size_t i = 0; i < matchingResults.size(); i++)
	{
		if (matchingResults[i] != -1)
			plotPR.at<cv::Vec3b>(i, matchingResults[i] - 1) = cv::Vec3b(0, 0, 255);
	}
	cv::resize(plotPR, plotPR, matSize  );
	cv::imshow("plotPR", plotPR);
	cv::waitKey(1);
	int err_sum = 0, num_sum = 0;
	for (size_t i = 0; i < gt.size(); i++)
	{
		if (!gt[i].empty()&& matchingResults[i] != -1) // ground truth is not empty 
		{
			std::vector<int>::iterator ret = std::find(gt[i].begin(), gt[i].end(), matchingResults[i]);
			if (ret == gt[i].end()) // no findings
			{
				//err_sum += std::min(std::abs(matchingResults[i]- *(gt[i].begin())), std::abs(matchingResults[i] - *(gt[i].end()-1)));
				err_sum += std::abs(matchingResults[i] - (*(gt[i].begin()) + *(gt[i].end() - 1))/2.0);
			}
			num_sum += 1;
		}
	}
	return float(err_sum)/ num_sum;
}