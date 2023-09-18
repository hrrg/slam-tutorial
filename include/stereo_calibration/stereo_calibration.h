#ifndef STEREO_CALIBRATION_H
#define STEREO_CALIBRATION_H

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

Eigen::MatrixXd compute_fundamental_matrix(const Eigen::MatrixXd &points1,
                                           const Eigen::MatrixXd &points2);

Eigen::MatrixXd computeFundamentalmatrixNormalized(const Eigen::MatrixXd &points1,
                                                   const Eigen::MatrixXd &points2);

Eigen::Vector3d compute_epipole(const Eigen::MatrixXd &F);

std::pair<Eigen::Matrix3d, Eigen::Matrix3d>
compute_matching_homographies(const Eigen::Vector3d &e2, const Eigen::MatrixXd &F,
                              const cv::Mat &im2, const Eigen::MatrixXd &points1,
                              const Eigen::MatrixXd &points2);

void obtainCorrespondingPoints(cv::Mat &image_left, cv::Mat &image_right,
                               std::vector<cv::Point2f> &matched_left,
                               std::vector<cv::Point2f> &matched_right, int num_points = 20,
                               bool show = false);

void equalizeStereoHist(cv::Mat &image1, cv::Mat &image2, int method = 0, bool show = false);

void undistortStereoImages(cv::Mat &image_left, cv::Mat &image_right, cv::Mat &undistort_left,
                           cv::Mat &undistort_right, const cv::Mat &K_left, const cv::Mat &K_right,
                           const cv::Mat &D_left, const cv::Mat &D_right, bool show = false);

void undistortKeyPoints(const std::vector<cv::Point2f> &matched_left,
                        const std::vector<cv::Point2f> &matched_right,
                        std::vector<cv::Point2f> &undistort_left,
                        std::vector<cv::Point2f> &undistort_right, const cv::Mat &K_left,
                        const cv::Mat &K_right, const cv::Mat &D_left, const cv::Mat &D_right);

Eigen::MatrixXd convertToEigenMatrix(const std::vector<cv::Point3f> &points);

Eigen::VectorXd leastSquares(const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

Eigen::MatrixXd divideByZ(const Eigen::MatrixXd &points);

cv::Mat eigenToMat(const Eigen::MatrixXd &eigenMatrix);

#endif