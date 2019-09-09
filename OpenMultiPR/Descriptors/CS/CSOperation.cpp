#include "CSOperation.h"


CSCreater::CSCreater(cv::Size imgSize_in, double rho_in, bool randomDic, bool uniform, bool bWrite_in) :imgSize(imgSize_in), rho(rho_in)
{
	n = imgSize.area();
	k = klab::UInt32(rho*n);
	bRandomDic = randomDic;
	bWrite = bWrite_in;
	bUniform = uniform;
	if (randomDic)
	{
		initA(256);
	}
	else
	{
		std::string pathsDic = "D:\\[CWPCL]_above_the_papers\\RS_DATA\\20171130RGBDIRGNSS\\20171124_10-32-33";
		initA(pathsDic, uniform);
	}
}

CSCreater::~CSCreater()
{

}

bool CSCreater::initA(int numDic)
{
		m = numDic;
		A = new kl1p::TNormalRandomMatrixOperator<klab::DoubleReal>(n, m, 0.5, 1);//0 1
		//A = new kl1p::TScalingOperator<klab::DoubleReal>(A, 1.0 / klab::Sqrt(klab::DoubleReal(m)));	// Pseudo-normalization of the matrix (required for AMP and EMBP solvers).
		return true;
}

bool CSCreater::initA(std::string pathsDic, bool uniform)
{
		//pathsDic.push_back("D:\\[CWPCL]_above_the_papers\\RS_DATA\\20171130RGBDIRGNSS\\20171124_10-32-33");
		// 对象建立后，对图像文件列表和GNSS、是否关键点数据进行赋值
		PicGNSSFile picsDic(pathsDic, PicGNSSFile::RGBDIR, true, "*");

		m = picsDic.getFileVolume();//klab::UInt32(alpha*n);	// Number of cs-measurements.字典个数

		arma::Mat<double> Amat(n, m);//row,col
		while (picsDic.doMain())
		{
			cv::Mat gray;
			cvtColor(picsDic.colorImg, gray, cv::COLOR_BGR2GRAY);
			//gray = picsDic.IRImg;
			resize(gray, gray, imgSize, 0, 0, cv::INTER_AREA);
			cv::Mat grayCol = gray.reshape(0, n);
			cv::imshow("gray", gray);
			cv::waitKey(1);
			
			double* data = Amat.colptr(picsDic.getFilePointer()-1);
			for (size_t i = 0; i < n; i++)
			{
				if (uniform)
				{
					data[i] = grayCol.at<uchar>(i, 0)/256.0;
				}
				else
				{
					data[i] = grayCol.at<uchar>(i, 0);
				}
				
			}

		}
		A = new kl1p::TMatrixOperator<klab::DoubleReal>(Amat);
	
	return true;
}

void CSCreater::printParams()
{
	// Display signal informations.
	std::cout << "==============================" << std::endl;
	std::cout << "N=" << n << " (signal size)" << std::endl;//CS维度，200是否太小？
	std::cout << "M=" << m /*<< "=" << std::setprecision(5) << (alpha*100.0)*/ << "% (number of measurements)" << std::endl;//原始图像维度
	std::cout << "K=" << k /*<< "=" << std::setprecision(5) << (rho*100.0)*/ << "% (signal sparsity)" << std::endl;//sparsity*m
	std::cout << "==============================" << std::endl;
}

arma::Col<klab::DoubleReal> CSCreater::generateCS(cv::Mat singleChannel)
{
	cv::Mat gray;
	cv::resize(singleChannel, gray, imgSize, 0, 0, cv::INTER_AREA);
	cv::Mat  grayCol = gray.reshape(0, imgSize.area());
	//cv::imshow("gray", gray);
	//cv::waitKey(1);
	
	//// Perform cs-measurements of size m.
	arma::Col<klab::DoubleReal> y(n);
	//A->apply(x0, y);
	double* data = y.colptr(0);
	for (size_t i = 0; i < n; i++)
	{
		if (bUniform)
		{
			data[i] = grayCol.at<uchar>(i, 0) / 256.0;
		}
		else
		{
			data[i] = grayCol.at<uchar>(i, 0);
		}
		
	}

	//

	klab::DoubleReal tolerance = 1e-5;	// Tolerance of the solution.
	arma::Col<klab::DoubleReal> x;		// Will contain the solution of the reconstruction.

	klab::KTimer timer;

	// Compute CoSaMP.
	std::cout << "------------------------------" << std::endl;
	std::cout << "[CoSaMP] Start." << std::endl;

	timer.start();
	kl1p::TCoSaMPSolver<klab::DoubleReal> cosamp(tolerance);
	cosamp.solve(y, A, k, x);
	//xRec.push_back(x);
	//x.print("x=");

	timer.stop();
	//		std::cout << "[CoSaMP] Done - SNR=" << std::setprecision(5) << klab::SNR(x, x0) << " - "
	std::cout << "Time=" << klab::UInt32(timer.durationInMilliseconds()) << "ms" << " - "
		<< "Iterations=" << cosamp.iterations() << std::endl;
	//if (bWrite)
	//	kl1p::WriteToCSVFile(x, "CoSaMP-Signal.csv");	// Write solution to a file.
	std::cout << "------------------------------" << std::endl;

	//std::cout << std::endl;
	//std::cout << "End of example." << std::endl;
	return x;
}
