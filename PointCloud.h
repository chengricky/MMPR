#include "Header.h"

class myICP
{
public:
    enum
    {
        ICP_SAMPLING_TYPE_UNIFORM = 0,
        ICP_SAMPLING_TYPE_GELFAND = 1
    };

    myICP(const int iterations = 250, const float tolerence = 0.05f, const float rejectionScale = 2.5f, 
    const int numLevels = 6, const int sampleType = myICP::ICP_SAMPLING_TYPE_UNIFORM, const int numMaxCorr = 1)
    {    
        m_tolerance = tolerence;
        m_numNeighborsCorr = numMaxCorr;
        m_rejectionScale = rejectionScale;
        m_maxIterations = iterations;
        m_numLevels = numLevels;
        m_sampleType = sampleType;
    }
    /**
     *  \brief Perform registration
     *
     *  @param [in] srcPC The input point cloud for the model. Expected to have the normals (Nx6). Currently,
     *  CV_32F is the only supported data type.
     *  @param [in] dstPC The input point cloud for the scene. It is assumed that the model is registered on the scene. Scene remains static. Expected to have the normals (Nx6). Currently, CV_32F is the only supported data type.
     *  @param [out] residual The output registration error.
     *  @param [out] pose Transformation between srcPC and dstPC.
     *  \return On successful termination, the function returns 0.
     *
     *  \details It is assumed that the model is registered on the scene. Scene remains static, while the model transforms. The output poses transform the models onto the scene. Because of the point to plane minimization, the scene is expected to have the normals available. Expected to have the normals (Nx6).
     */
    int registerModelToScene(const cv::Mat& srcPC, const cv::Mat& dstPC, const std::vector<cv::DMatch>& correspondence, 
            double& residual, cv::Matx44d& pose);    

private:
    float m_tolerance;
    int m_maxIterations;
    float m_rejectionScale;
    int m_numNeighborsCorr;
    int m_numLevels;
    int m_sampleType;
};