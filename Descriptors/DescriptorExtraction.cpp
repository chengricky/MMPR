#include "DescriptorExtraction.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"
#include "../Tools/Timer.h"

GoogLeNetExtractor::GoogLeNetExtractor(int imgIdx) : ImgDescriptorExtractor(imgIdx)
{
	//net = cv::dnn::readNetFromCaffe("bvlc_googlenet\\deploy.prototxt", "bvlc_googlenet\\bvlc_googlenet.caffemodel");
	net = cv::dnn::readNetFromCaffe("places_googlenet\\deploy_googlenet_places365.prototxt", "places_googlenet\\googlenet_places365.caffemodel");
	
	if (net.empty())
	{
		std::cout << "Read file ERROR!" << std::endl;
	}
	std::ifstream fIdx("compressIdx.txt", std::ios::in);
	std::string str;	
	while (std::getline(fIdx, str))
	{
		//std::cout << str<<std::endl;
		idxes.push_back(atoi(str.data()));
	}
	fIdx.close();
}

static cv::Mat featureReshape(std::vector<cv::Mat>& layer1_Mat)
{
	cv::Mat layer1_vec;
	for (auto mat : layer1_Mat)
	{
		if (layer1_vec.empty())
		{
			//std::cout << mat.size[0] << std::endl;
			//std::cout << mat.size[1] << std::endl;
			//std::cout << mat.size[2] << std::endl;
			//std::cout << mat.size[3] << std::endl;
			layer1_vec = mat.reshape(1, 1);
		}
		else
		{
			//std::cout << mat.size[0] << std::endl;
			//std::cout << mat.size[1] << std::endl;
			//std::cout << mat.size[2] << std::endl;
			//std::cout << mat.size[3] << std::endl;
			cv::Mat m = mat.reshape(1, 1);
			//std::cout<<m.type();//类型是float
			cv::hconcat(layer1_vec, m, layer1_vec);
		}
	}
	return layer1_vec;
}

bool GoogLeNetExtractor::extract(std::vector<cv::Mat> img)
{
	//std::cout << "\nGoogleNet Time consumed: ";
	//Timer timer;
	//timer.start();

	result = cv::Mat();
	cv::Mat imgResize;
	cv::resize(img[imgIdx], imgResize, cv::Size(224, 224));
	net.setInput(cv::dnn::blobFromImage(imgResize, 1, cv::Size(224, 224), cv::Scalar(104.051007218, 112.514489108, 116.676038934)), "data");

	std::vector<std::string> layers;
	layers.push_back("inception_3a/3x3_reduce");
	layers.push_back("inception_3a/3x3");
	std::vector<cv::Mat> features;
	net.forward(features, layers);

	auto feature_GoogLeNet = featureReshape(features);
	for (auto id : idxes)
	{
		result.push_back(feature_GoogLeNet.at<float>(0, id));
	}
	result = result.reshape(1, 1);
	
	//timer.stop();
	//timer.print_elapsed_time(TimeExt::MSec);
	return true;
}

ORBExtractor::ORBExtractor(int imgIdx): OCVExtractor(imgIdx)
{
	detector = cv::ORB::create(1000, 1.2f, 8, 19, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); //same with the paper "ORBSLAM2"
	//detector = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); 	// BoW使用默认配置获得ORB特征
}

#ifdef USE_CONTRIB
SURFExtractor::SURFExtractor(int imgIdx) : OCVExtractor(imgIdx)
{
	detector = cv::xfeatures2d::SURF::create();
}
#endif

bool OCVExtractor::extract(std::vector<cv::Mat> img)
{
	detector->detectAndCompute(img[imgIdx], cv::Mat(), keypoints, result);//描述子数目与关键点数相同，现在是每个描述子是32维,256维度的二进制数值-orb
	return true;
}


GISTExtractor::GISTExtractor(int imgIdx, bool useColor, bool isNormalize, cv::Size imgSize) : ImgDescriptorExtractor(imgIdx),
GIST_PARAMS({ useColor, imgSize.width, imgSize.height, 4, 3, { 8, 8, 8 } }) {}

bool GISTExtractor::extract(std::vector<cv::Mat> todoImages)
{
	//std::cout << "\nGIST Time consumed: ";
	//Timer timer;
	//timer.start();

	cls::GIST gist_ext(GIST_PARAMS);
	std::vector<float> result_vec;
	gist_ext.extract(todoImages[imgIdx], result_vec, isNormalize);//输入彩色图，内部会自动根据DEFAULT_PARAMS需要，转换灰度	
	cv::Mat(result_vec).reshape(1, 1).copyTo(result);

	//timer.stop();
	//timer.print_elapsed_time(TimeExt::MSec);
	return true;
}

CSExtractor::CSExtractor(int imgIdx, cv::Size imgSize) : ImgDescriptorExtractor(imgIdx), imgSize(imgSize) {}

bool CSExtractor::extract(std::vector<cv::Mat> todoImages)
{
	cv::Mat img;
	arma::Col<klab::DoubleReal> result_col;
	if (todoImages[imgIdx].channels() != 1)
		cv::cvtColor(todoImages[imgIdx], img, cv::COLOR_BGR2GRAY);
	else
		img = todoImages[imgIdx];
	try
	{
		cv::Size imgSizeCS = cv::Size(imgSize.width / 2, imgSize.height / 2);
		CSCreater csCreater(imgSizeCS, 0.1, true, false);
		result_col = csCreater.generateCS(img);
	}
	catch (klab::KException& e)
	{
		std::cout << "ERROR! KLab exception : " << klab::FormatExceptionToString(e) << std::endl;
		return false;
	}
	catch (std::exception& e)
	{
		std::cout << "ERROR! Standard exception : " << klab::FormatExceptionToString(e) << std::endl;
		return false;
	}
	catch (...)
	{
		std::cout << "ERROR! Unknown exception !" << std::endl;
		return false;
	}
	for (arma::Col<klab::DoubleReal>::iterator i = result_col.begin(); i != result_col.end(); i++)
	{
		result.push_back(*i);
	}
	return true;
}

LDBExtractor::LDBExtractor(int imgIdx, bool useColor) : ImgDescriptorExtractor(imgIdx), useColor(useColor) {}

bool LDBExtractor::extract(std::vector<cv::Mat> todoImages)
{
	// Select the central keypoint
	std::vector<cv::KeyPoint> kpts;
	cv::KeyPoint kpt;
	kpt.pt.x = todoImages[imgIdx].cols / 2 + 1;
	kpt.pt.y = todoImages[imgIdx].rows / 2 + 1;
	kpt.size = 1.0;
	kpt.angle = 0.0;
	kpts.push_back(kpt);
	LDB ldb;
	// attention: color image are not supported in the ldb extraction 内部会转换为灰度图
	cv::Mat matrix;
	if (todoImages[imgIdx].channels() != 1 && useColor)
		matrix = illumination_conversion(todoImages[imgIdx]);
	else
		matrix = todoImages[imgIdx];
	ldb.compute(matrix, kpts, result);

	return true;
}

/**
* @brief Converts an image to illumination invariant image.
* @param image Computed image
* @return Illumination invariant image
*/
cv::Mat LDBExtractor::illumination_conversion(cv::Mat image) {

	float alpha = 0.47;

	std::vector<cv::Mat> channels(3);
	split(image, channels);

	cv::Mat imageB, imageG, imageR;
	cv::Mat imageI = cv::Mat(cv::Size(image.cols, image.rows), CV_32FC1);
	cv::Mat imageI8U = cv::Mat(cv::Size(image.cols, image.rows), CV_8UC1);

	channels[0].convertTo(imageB, CV_32FC1, 1.0 / 255.0, 0);
	channels[1].convertTo(imageG, CV_32FC1, 1.0 / 255.0, 0);
	channels[2].convertTo(imageR, CV_32FC1, 1.0 / 255.0, 0);

	float valueG, valueB, valueR;

	for (int i = 0; i < imageI.rows; i++) {

		for (int j = 0; j < imageI.cols; j++) {

			if (imageG.at<float>(i, j) != 0)
				valueG = log(imageG.at<float>(i, j));
			else
				valueG = 0;

			if (imageB.at<float>(i, j) != 0)
				valueB = alpha*log(imageB.at<float>(i, j));
			else
				valueB = 0;

			if (imageR.at<float>(i, j) != 0)
				valueR = (1 - alpha)*log(imageR.at<float>(i, j));
			else
				valueR = 0;

			imageI.at<float>(i, j) = 0.5 + valueG - valueB - valueR;

			if (imageI.at<float>(i, j)<0) imageI.at<float>(i, j) = 1;

		}

	}

	imageI.convertTo(imageI8U, CV_8UC1, 255.0, 0);

	return imageI8U;

}

ImgDescriptorExtractor* Extraction::getResult(int idx)
{
	if (idx>=extractions.size())
	{
		throw std::range_error("exceed the number of extractor.");
	}
	return extractions[idx];
}

Extraction::~Extraction()
{
	//for (size_t i = 0; i < extractions.size(); i++)
	//{
	//	delete extractions[i];
	//}
	//extractions.clear();
}

void Extraction::run(std::vector<cv::Mat> const&  todoImages) {

	auto it = extractions.begin(); 

	while (it != extractions.end()) 
		(*it++)->extract(todoImages);
}
