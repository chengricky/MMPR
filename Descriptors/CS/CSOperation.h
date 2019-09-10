#pragma once

#include "CompressedSensingExample.h" 
#include "../../FileInterface/PicGnssFile.h"

//std::cout << "Start of KL1p compressed-sensing example." << std::endl;
//std::cout << "Try to determine a sparse vector x " << std::endl;
//std::cout << "from an underdetermined set of linear measurements y=A*x, " << std::endl;
//std::cout << "where A is a random gaussian i.i.d sensing matrix." << std::endl;
class CSCreater
{
public:
	CSCreater(cv::Size imgSize_in, double rho_in = 0.1, bool randomDic = true, bool uniform = false, bool bWrite_in = false);
	~CSCreater();
	void printParams();
	bool initA(int numDic);
	bool initA(std::string pathsDic, bool uniform = false);	
	arma::Col<klab::DoubleReal> generateCS(cv::Mat singleChannel);
private:
	cv::Size imgSize;
	klab::UInt32 n;							// Size of the original signal x0.
	//klab::DoubleReal alpha = 0.5;			// Ratio of the cs-measurements.
	klab::DoubleReal rho;					// Ratio of the sparsity of the signal x0.	
	klab::UInt32 k;							// Sparsity of the signal x0 (number of non-zero elements).
	bool bWrite;
	bool bRandomDic;
	bool bUniform;
	klab::UInt32 m;
	klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal> > A;//matrix A of size (m,n).
};