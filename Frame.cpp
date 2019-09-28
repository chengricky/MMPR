#include "Frame.h"

Frame::Frame(){}

Frame::~Frame(){}

Frame::Frame(const Frame& frame)
{
    mKpts = frame.mKpts;
    mDspts = frame.mDspts;
    mK = frame.mK;
}

Frame::Frame(const cv::Mat &mDspts, const std::vector<cv::KeyPoint>& mKpts, const cv::Mat &K)
{
    mDspts.copyTo(this->mDspts);
    mKpts = mKpts;
    K.copyTo(mK);
}

void Frame::CopyFrom(const Frame &frame)
{
    mKpts = frame.mKpts;
    frame.mDspts.copyTo(mDspts);
    frame.mK.copyTo(mK);
}
