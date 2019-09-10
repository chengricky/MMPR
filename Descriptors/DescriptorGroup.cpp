#include "DescriptorGroup.h"
#include "..\tools\Timer.h"
#include "..\tools\list_dir.h"
#include "direct.h"
#include "../Tools/GNSSDistance.h"

Descriptors::Descriptors(GlobalConfig& config, bool isRefImage)
{
	isColor = config.isColor;
	if (isRefImage)
		picPath = config.pathRec;
	else
		picPath = config.pathTest;

	// 对图像文件列表和GNSS、是否关键点数据进行赋值
	if (config.useColor&&!config.useDepth&&!config.useIR)
	{
		picFiles.init(picPath, PicGNSSFile::RGB, config.withGPS, "color" );
	}
	else if (config.useColor && config.useDepth && !config.useIR)
	{
		picFiles.init(picPath, PicGNSSFile::RGBD, config.withGPS, "color" );
	}
	else if (config.useColor && config.useDepth && config.useIR)
	{
		picFiles.init(picPath, PicGNSSFile::RGBDIR, config.withGPS, "color" );
	}

	assert(config.useColor);


	// set descriptor extractors
	if (config.useBoW == true)
	{
		extraction.add(new  ORBExtractor(0));
		if (config.useIR)
		{
			extraction.add(new ORBExtractor(2));
		}
	}
	if (config.useGIST == true)
	{
		extraction.add(new GISTExtractor(0, config.isColor, true, config.dImgSize));
		if (config.useDepth)
		{
			extraction.add(new GISTExtractor(1, false, true, config.dImgSize));
		}
		if (config.useIR)
		{
			extraction.add(new GISTExtractor(2, false, true, config.dImgSize));
		}
	}
	if (config.useLDB == true)
	{
		extraction.add(new LDBExtractor(0, config.isColor));
		if (config.useDepth)
		{
			extraction.add(new LDBExtractor(1, config.isColor));
		}
		if (config.useIR)
		{
			extraction.add(new LDBExtractor(2, config.isColor));
		}
	}
	if (config.useCS == true)
	{
		extraction.add(new CSExtractor(0, config.dImgSize));
	}
	extraction.add(new GoogLeNetExtractor(0));

	while (picFiles.doMain())
	{
		/*GPS*/
		// ddmm.mmmm --> dd.dddddd (style conversion)
		cv::Mat GPS_row(1, 2, CV_32FC1);
		GPS_row.at<float>(0, 0) = ddmm2dd(picFiles.longitudeValue);
		GPS_row.at<float>(0, 1) = ddmm2dd(picFiles.latitudeValue);
		GPS.push_back(GPS_row);

		// Mat order color-depth-IR
		std::vector<cv::Mat> todoImages = getAllImage(picFiles, config.dImgSize);

		// save the descriptor of GIST and LDB
		std::cout << "Time consumed: ";
		Timer timer;
		timer.start();

		extraction.run(todoImages);
		std::vector<cv::Mat> xORBChannel;
		for (size_t i = 0; i < extraction.getSize(); i++)
		{
			ImgDescriptorExtractor* pDescriptor = extraction.getResult(i);
			if (typeid(*pDescriptor) == typeid(GISTExtractor))
			{
				GIST[pDescriptor->getImgIdx()].push_back(pDescriptor->getResult());
			}
			else if (typeid(*pDescriptor) == typeid(LDBExtractor))
			{
				LDB[pDescriptor->getImgIdx()].push_back(pDescriptor->getResult());
			}
			else if (typeid(*pDescriptor) == typeid(ORBExtractor))
			{
				xORBChannel.push_back(pDescriptor->getResult());
			}
			else if (typeid(*pDescriptor) == typeid(CSExtractor))
			{
				CS.push_back(pDescriptor->getResult());
			}
			else if (typeid(*pDescriptor) == typeid(GoogLeNetExtractor))
			{
				GG.push_back(pDescriptor->getResult());
			}
		}

		timer.stop();
		timer.print_elapsed_time(TimeExt::MSec);

		/*CS*/
		//CS.push_back(xCS);
		/*ORB*/
		if (xORBChannel.size()>=1)
		{
			ORB_RGB.push_back(xORBChannel[0]);
			if (xORBChannel.size() >= 2)
			{
				ORB_IR.push_back(xORBChannel[1]);	
			}
		}
	}

	extraction.release();
}


std::vector<cv::Mat> Descriptors::getAllImage(const PicGNSSFile& picsRec, const cv::Size& imgSize)
{
	// Mat order color-depth-IR
	std::vector<cv::Mat> todoImages(3);


	cv::resize(picsRec.colorImg, todoImages[0], imgSize);
	if (!isColor)
	{
		cv::Mat tmp;
		cvtColor(picsRec.colorImg, tmp, cv::COLOR_BGR2GRAY);
		cv::resize(tmp, todoImages[0], imgSize);
	}
	if (!picsRec.depthImg.empty())
	{
		cv::resize(picsRec.depthImg, todoImages[1], imgSize);
	}
	if (!picsRec.IRImg.empty())
	{
		cv::resize(picsRec.IRImg, todoImages[2], imgSize);
	}


	return todoImages;
}