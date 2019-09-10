#pragma once

//#define USE_CONTRIB

#include <opencv2/opencv.hpp>

#ifdef _DEBUG  
#pragma comment(lib, "opencv_world401d.lib")
//#pragma comment(lib, "opencv_core401d.lib")
//#pragma comment(lib, "opencv_highgui401d.lib")
//#pragma comment(lib, "opencv_imgproc401d.lib")
//#pragma comment(lib, "opencv_imgcodecs401d.lib")
//#pragma comment(lib, "opencv_features2d401d.lib")
//#pragma comment(lib, "opencv_dnn401d.lib")
//#pragma comment(lib, "opencv_videoio401d.lib")
#else
#pragma comment(lib, "opencv_world401.lib")
//#pragma comment(lib, "opencv_core401.lib")
//#pragma comment(lib, "opencv_highgui401.lib")
//#pragma comment(lib, "opencv_imgproc401.lib")
//#pragma comment(lib, "opencv_imgcodecs401.lib")
//#pragma comment(lib, "opencv_features2d401.lib")
//#pragma comment(lib, "opencv_dnn401.lib")
//#pragma comment(lib, "opencv_videoio401.lib")
#endif


#if ((defined USE_CONTRIB) && (defined _DEBUG))  
#include <opencv2/xfeatures2d.hpp>
#pragma comment(lib, "opencv_xfeatures2d401d.lib")
#endif  

#if ((defined USE_CONTRIB) && (!defined _DEBUG))  
#include <opencv2/xfeatures2d.hpp>
#pragma comment(lib, "opencv_xfeatures2d401.lib")
#endif 
