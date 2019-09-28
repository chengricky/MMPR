#ifndef FRAME_H
#define FRAME_H

#include "Header.h"

class Frame
{
public:
    Frame();

    ~Frame();

    Frame(const Frame &frame);

    Frame(const cv::Mat &mDspts, const std::vector<cv::KeyPoint>& mKpts, const cv::Mat &K);

    void CopyFrom(const Frame &frame);


public:
    //keypoints and descriptors
    cv::Mat mDspts;
    std::vector<cv::KeyPoint> mKpts;

    //intrinsic and extrinsic
    cv::Mat mK;
};


#endif //FRAME_H
