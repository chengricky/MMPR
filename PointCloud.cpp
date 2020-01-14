/// Adapted from the OpenCV implementation
#include "PointCloud.h"

using namespace cv;


static void subtractColumns(Mat srcPC, Vec3d& mean)
{
  int height = srcPC.rows;

  for (int i=0; i<height; i++)
  {
    float *row = srcPC.ptr<float>(i);
    {
      row[0]-=(float)mean[0];
      row[1]-=(float)mean[1];
      row[2]-=(float)mean[2];
    }
  }
}

// as in PCA
static void computeMeanCols(Mat srcPC, Vec3d& mean)
{
  int height = srcPC.rows;

  double mean1=0, mean2 = 0, mean3 = 0;

  for (int i=0; i<height; i++)
  {
    const float *row = srcPC.ptr<float>(i);
    {
      mean1 += (double)row[0];
      mean2 += (double)row[1];
      mean3 += (double)row[2];
    }
  }

  mean1/=(double)height;
  mean2/=(double)height;
  mean3/=(double)height;

  mean[0] = mean1;
  mean[1] = mean2;
  mean[2] = mean3;
}

// compute the average distance to the origin
static double computeDistToOrigin(Mat srcPC)
{
  int height = srcPC.rows;
  double dist = 0;

  for (int i=0; i<height; i++)
  {
    const float *row = srcPC.ptr<float>(i);
    dist += sqrt(row[0]*row[0]+row[1]*row[1]+row[2]*row[2]);
  }

  return dist;
}


#include "t_hash_int.hpp"
using namespace cv::ppf_match_3d;

/* Fast way to look up the duplicates
duplicates is pre-allocated
make sure that the max element in array will not exceed maxElement
*/
static hashtable_int* getHashtable(int* data, size_t length, int numMaxElement)
{
  hashtable_int* hashtable = hashtableCreate(static_cast<size_t>(numMaxElement*2), 0);
  for (size_t i = 0; i < length; i++)
  {
    const KeyType key = (KeyType)data[i];
    hashtableInsertHashed(hashtable, key+1, reinterpret_cast<void*>(i+1));
  }

  return hashtable;
}

// Kok Lim Low's linearization
static void minimizePointToPlaneMetric(Mat Src, Mat Dst, Vec3d& rpy, Vec3d& t)
{
  //Mat sub = Dst - Src;
  Mat A = Mat(Src.rows, 6, CV_64F);
  Mat b = Mat(Src.rows, 1, CV_64F);
  Mat rpy_t;

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<Src.rows; i++)
  {
    const Vec3d srcPt(Src.ptr<double>(i));
    const Vec3d dstPt(Dst.ptr<double>(i));
    const Vec3d normals(Dst.ptr<double>(i) + 3);
    const Vec3d sub = dstPt - srcPt;
    const Vec3d axis = srcPt.cross(normals);

    *b.ptr<double>(i) = sub.dot(normals);
    hconcat(axis.reshape<1, 3>(), normals.reshape<1, 3>(), A.row(i));
  }

  cv::solve(A, b, rpy_t, DECOMP_SVD);
  rpy_t.rowRange(0, 3).copyTo(rpy);
  rpy_t.rowRange(3, 6).copyTo(t);
}

/**
 *  \brief Compute a rotation in order to rotate around X direction
 */
static inline void getUnitXRotation(double angle, Matx33d& Rx)
{
  const double sx = sin(angle);
  const double cx = cos(angle);

  Mat(Rx.eye()).copyTo(Rx);
  Rx(1, 1) = cx;
  Rx(1, 2) = -sx;
  Rx(2, 1) = sx;
  Rx(2, 2) = cx;
}

/**
*  \brief Compute a rotation in order to rotate around Y direction
*/
static inline void getUnitYRotation(double angle, Matx33d& Ry)
{
  const double sy = sin(angle);
  const double cy = cos(angle);

  Mat(Ry.eye()).copyTo(Ry);
  Ry(0, 0) = cy;
  Ry(0, 2) = sy;
  Ry(2, 0) = -sy;
  Ry(2, 2) = cy;
}

/**
*  \brief Compute a rotation in order to rotate around Z direction
*/
static inline void getUnitZRotation(double angle, Matx33d& Rz)
{
  const double sz = sin(angle);
  const double cz = cos(angle);

  Mat(Rz.eye()).copyTo(Rz);
  Rz(0, 0) = cz;
  Rz(0, 1) = -sz;
  Rz(1, 0) = sz;
  Rz(1, 1) = cz;
}


/**
*  \brief Convert euler representation to rotation matrix
*
*  \param [in] euler RPY angles
*  \param [out] R 3x3 Rotation matrix
*/
static inline void eulerToDCM(const Vec3d& euler, Matx33d& R)
{
  Matx33d Rx, Ry, Rz;

  getUnitXRotation(euler[0], Rx);
  getUnitYRotation(euler[1], Ry);
  getUnitZRotation(euler[2], Rz);

  Mat(Rx * (Ry * Rz)).copyTo(R);
}

static inline void rtToPose(const Matx33d& R, const Vec3d& t, Matx44d& Pose)
{
  Matx34d P;
  hconcat(R, t, P);
  vconcat(P, Matx14d(0, 0, 0, 1), Pose);
}

static void getTransformMat(Vec3d& euler, Vec3d& t, Matx44d& Pose)
{
  Matx33d R;
  eulerToDCM(euler, R);
  rtToPose(R, t, Pose);
}

static inline void poseToR(const Matx44d& Pose, Matx33d& R)
{
  Mat(Pose).rowRange(0, 3).colRange(0, 3).copyTo(R);
}

static inline void poseToRT(const Matx44d& Pose, Matx33d& R, Vec3d& t)
{
  poseToR(Pose, R);
  Mat(Pose).rowRange(0, 3).colRange(3, 4).copyTo(t);
}

const float EPS = 1.192092896e-07F;        /* smallest such that 1.0+FLT_EPSILON != 1.0 */

Mat transformPCPose(Mat pc, const Matx44d& Pose)
{
  Mat pct = Mat(pc.rows, pc.cols, CV_32F);

  Matx33d R;
  Vec3d t;
  poseToRT(Pose, R, t);

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<pc.rows; i++)
  {
    const float *pcData = pc.ptr<float>(i);
    const Vec3f n1(&pcData[3]);

    Vec4d p = Pose * Vec4d(pcData[0], pcData[1], pcData[2], 1);
    Vec3d p2(p.val);

    // p2[3] should normally be 1
    if (fabs(p[3]) > EPS)
    {
      Mat((1.0 / p[3]) * p2).reshape(1, 1).convertTo(pct.row(i).colRange(0, 3), CV_32F);
    }

    // If the point cloud has normals,
    // then rotate them as well
    if (pc.cols == 6)
    {
      Vec3d n(n1), n2;

      n2 = R * n;
      double nNorm = cv::norm(n2);

      if (nNorm > EPS)
      {
        Mat((1.0 / nNorm) * n2).reshape(1, 1).convertTo(pct.row(i).colRange(3, 6), CV_32F);
      }
    }
  }

  return pct;
}

typedef cv::flann::L2<float> Distance_32F;
typedef cv::flann::GenericIndex< Distance_32F > FlannIndex;
void* indexPCFlann(Mat pc)
{
  Mat dest_32f;
  pc.colRange(0,3).copyTo(dest_32f);
  return new FlannIndex(dest_32f, cvflann::KDTreeSingleIndexParams(8));
}
void destroyFlann(void* flannIndex)
{
  delete ((FlannIndex*)flannIndex);
}

// From numerical receipes: Finds the median of an array
static float medianF(float arr[], int n)
{
  int low, high ;
  int median;
  int middle, ll, hh;

  low = 0 ;
  high = n-1 ;
  median = (low + high) >>1;
  for (;;)
  {
    if (high <= low) /* One element only */
      return arr[median] ;

    if (high == low + 1)
    {
      /* Two elements only */
      if (arr[low] > arr[high])
        std::swap(arr[low], arr[high]) ;
      return arr[median] ;
    }
    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) >>1;
    if (arr[middle] > arr[high])
      std::swap(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])
      std::swap(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])
      std::swap(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    std::swap(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;)
    {
      do
        ll++;
      while (arr[low] > arr[ll]) ;
      do
        hh--;
      while (arr[hh]  > arr[low]) ;

      if (hh < ll)
        break;

      std::swap(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    std::swap(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
      low = ll;
    if (hh >= median)
      high = hh - 1;
  }
}

static float getRejectionThreshold(float* r, int m, float outlierScale)
{
  float* t=(float*)calloc(m, sizeof(float));
  int i=0;
  float s=0, medR, threshold;

  memcpy(t, r, m*sizeof(float));
  medR=medianF(t, m);

  for (i=0; i<m; i++)
    t[i] = (float)fabs((double)r[i]-(double)medR);

  s = 1.48257968f * medianF(t, m);

  threshold = (outlierScale*s+medR);

  free(t);
  return threshold;
}

void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, const int numNeighbors)
{
  Mat obj_32f;
  pc.colRange(0, 3).copyTo(obj_32f);
  ((FlannIndex*)flannIndex)->knnSearch(obj_32f, indices, distances, numNeighbors, cvflann::SearchParams(32));
}

// For speed purposes this function assumes that PC, Indices and Distances are created with continuous structures
void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances)
{
  queryPCFlann(flannIndex, pc, indices, distances, 1);
}

// source point clouds are assumed to contain their normals
int myICP::registerModelToScene(const Mat& srcPC, const Mat& dstPC, const std::vector<cv::DMatch>& correspondence, double& residual, Matx44d& pose)
{
    int n = srcPC.rows;
    Mat srcTemp = srcPC.clone();
    Mat dstTemp = dstPC.clone();
    Vec3d meanSrc, meanDst;
    computeMeanCols(srcTemp, meanSrc);
    computeMeanCols(dstTemp, meanDst);
    Vec3d meanAvg = 0.5 * (meanSrc + meanDst);
    subtractColumns(srcTemp, meanAvg);
    subtractColumns(dstTemp, meanAvg);
    
    const int MaxIterationsPyr = m_maxIterations;
    const double TolP = m_tolerance;

  double distSrc = computeDistToOrigin(srcTemp);
  double distDst = computeDistToOrigin(dstTemp);

  double scale = (double)n / ((distSrc + distDst)*0.5);

  srcTemp(cv::Range(0, srcTemp.rows), cv::Range(0,3)) *= scale;
  dstTemp(cv::Range(0, dstTemp.rows), cv::Range(0,3)) *= scale;    

    Mat srcPC0 = srcTemp;
    Mat dstPC0 = dstTemp;
    Mat srcPCT = srcPC0;
    Mat dstPCS = dstPC0;

    int numElSrc = dstPCS.rows;
    int sizesResult[2] = {(int)numElSrc, 1};
    int* newI = new int[numElSrc];
    int* newJ = new int[numElSrc];  

    int* indices = new int[numElSrc]; 
    float* distances = new float[numElSrc];
    Mat Indices(2, sizesResult, CV_32S, indices, 0);
    Mat Distances(2, sizesResult, CV_32F, distances, 0);    

    int* indicesModel = new int[numElSrc];
    int* indicesScene = new int[numElSrc];
    
    for(int i=0;i<numElSrc;i++)
    {
        newI[i] = i;//correspondence[i].queryIdx;
        newJ[i] = i;//correspondence[i].trainIdx;
        assert(newI[i]<srcPC0.rows && newJ[i]<dstPC0.rows);
        distances[i] = cv::norm(srcPC0.row(newI[i]).colRange(0, 3), dstPC0.row(newJ[i]).colRange(0, 3));
    }
    
    double fval_old=9999999999;
    double fval_perc=0;
    double fval_min=9999999999;
    Mat Src_Moved = srcPCT.clone();
    int i=0;
    Matx44d PoseX = Matx44d::eye();
    double tempResidual = 0;

    void* flann = indexPCFlann(dstPCS);

    while ( (!(fval_perc<(1+TolP) && fval_perc>(1-TolP))) && i<MaxIterationsPyr)
    {
      uint di=0, selInd = 0;

      if(i>0)
      {
        queryPCFlann(flann, Src_Moved, Indices, Distances);
        for (di=0; di<numElSrc; di++)
        {
          newI[di] = di;
          newJ[di] = indices[di];
        }       
      


        if (true)//useRobustReject
        {
        int numInliers = 0;
        float threshold = getRejectionThreshold(distances, Distances.rows, m_rejectionScale);
        Mat acceptInd = Distances<threshold;

        uchar *accPtr = (uchar*)acceptInd.data;
        for (int l=0; l<acceptInd.rows; l++)
        {
            if (accPtr[l])
            {
            newI[numInliers] = l;
            newJ[numInliers] = indices[l];
            numInliers++;
            }
        }
        numElSrc=numInliers;
        }
      }

      // Step 2: Picky ICP
      // Among the resulting corresponding pairs, if more than one scene point p_i
      // is assigned to the same model point m_j, then select p_i that corresponds
      // to the minimum distance

      hashtable_int* duplicateTable = getHashtable(newJ, numElSrc, dstPCS.rows);

      for (di=0; di<duplicateTable->size; di++)
      {
        hashnode_i *node = duplicateTable->nodes[di];

        if (node)
        {
          // select the first node
          size_t idx = reinterpret_cast<size_t>(node->data)-1, dn=0;
          int dup = (int)node->key-1;
          size_t minIdxD = idx;
          float minDist = distances[idx];

          while ( node )
          {
            idx = reinterpret_cast<size_t>(node->data)-1;

            if (distances[idx] < minDist)
            {
              minDist = distances[idx];
              minIdxD = idx;
            }

            node = node->next;
            dn++;
          }

          indicesModel[ selInd ] = newI[ minIdxD ];
          indicesScene[ selInd ] = dup ;
          selInd++;
        }
      }

      hashtableDestroy(duplicateTable);

      if (selInd >= 6)
      {

        Mat Src_Match = Mat(selInd, srcPCT.cols, CV_64F);
        Mat Dst_Match = Mat(selInd, srcPCT.cols, CV_64F);

        for (di=0; di<selInd; di++)
        {
          const int indModel = indicesModel[di];
          const int indScene = indicesScene[di];
          const float *srcPt = srcPCT.ptr<float>(indModel);
          const float *dstPt = dstPCS.ptr<float>(indScene);
          double *srcMatchPt = Src_Match.ptr<double>(di);
          double *dstMatchPt = Dst_Match.ptr<double>(di);
          int ci=0;

          for (ci=0; ci<srcPCT.cols; ci++)
          {
            srcMatchPt[ci] = (double)srcPt[ci];
            dstMatchPt[ci] = (double)dstPt[ci];
          }
        }

        Vec3d rpy, t;
        minimizePointToPlaneMetric(Src_Match, Dst_Match, rpy, t);
        if (cvIsNaN(cv::trace(rpy)) || cvIsNaN(cv::norm(t)))
          break;
        getTransformMat(rpy, t, PoseX);
        Src_Moved = transformPCPose(srcPCT, PoseX);

        double fval = cv::norm(Src_Match, Dst_Match)/(double)(Src_Moved.rows);

        // Calculate change in error between iterations
        fval_perc=fval/fval_old;

        // Store error value
        fval_old=fval;

        if (fval < fval_min)
          fval_min = fval;
      }
      else
        break;

      i++;



    }

    pose = PoseX * pose;
    residual = tempResidual;

    delete[] newI;
    delete[] newJ;
    delete[] indicesModel;
    delete[] indicesScene;
    delete[] distances;
    delete[] indices;

    tempResidual = fval_min;

    destroyFlann(flann);

    
  

  Matx33d Rpose;
  Vec3d Cpose;
  poseToRT(pose, Rpose, Cpose);
  Cpose = Cpose / scale + meanAvg - Rpose * meanAvg;
  rtToPose(Rpose, Cpose, pose);

  residual = tempResidual;

  return 0;
}