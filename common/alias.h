// Alias for various Eigen matrix types.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include "sophus/se3.hpp"

#include "rodrigues.h"

namespace xivo {

using number_t = double;

using Mat3 = Eigen::Matrix<number_t, 3, 3>;
using Vec3 = Eigen::Matrix<number_t, 3, 1>;
using Mat4 = Eigen::Matrix<number_t, 4, 4>;
using Vec4 = Eigen::Matrix<number_t, 4, 1>;
using Mat2 = Eigen::Matrix<number_t, 2, 2>;
using Vec2 = Eigen::Matrix<number_t, 2, 1>;
using Vec6 = Eigen::Matrix<number_t, 6, 1>;
using Vec8 = Eigen::Matrix<number_t, 8, 1>;
using Vec9 = Eigen::Matrix<number_t, 9, 1>;
using Mat6 = Eigen::Matrix<number_t, 6, 6>;
using Mat23 = Eigen::Matrix<number_t, 2, 3>;
using Mat32 = Eigen::Matrix<number_t, 3, 2>;
using Mat34 = Eigen::Matrix<number_t, 3, 4>;
using Mat36 = Eigen::Matrix<number_t, 3, 6>;
using Mat93 = Eigen::Matrix<number_t, 9, 3>;
using Mat39 = Eigen::Matrix<number_t, 3, 9>;
using MatX = Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic>;
using VecX = Eigen::Matrix<number_t, Eigen::Dynamic, 1>;
using MatX2 = Eigen::Matrix<number_t, Eigen::Dynamic, 2>;
using MatX3 = Eigen::Matrix<number_t, Eigen::Dynamic, 3>;
using MatX6 = Eigen::Matrix<number_t, Eigen::Dynamic, 6>;
using MatX7 = Eigen::Matrix<number_t, Eigen::Dynamic, 7>;

using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat4f = Eigen::Matrix<float, 4, 4>;
using Mat34f = Eigen::Matrix<float, 3, 4>;
using Mat23f = Eigen::Matrix<float, 2, 3>;
using Mat24f = Eigen::Matrix<float, 2, 4>;
using MatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3>;

using Mat3d = Eigen::Matrix<double, 3, 3>;
using Mat4d = Eigen::Matrix<double, 4, 4>;
using Mat34d = Eigen::Matrix<double, 3, 4>;
using Mat23d = Eigen::Matrix<double, 2, 3>;
using Mat24d = Eigen::Matrix<double, 2, 4>;
using MatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using MatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

using Vec2f = Eigen::Matrix<float, 2, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using VecXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using Vec2d = Eigen::Matrix<double, 2, 1>;
using Vec3d = Eigen::Matrix<double, 3, 1>;
using Vec4d = Eigen::Matrix<double, 4, 1>;
using VecXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Vec2i = Eigen::Matrix<int, 2, 1>;
using Vec3i = Eigen::Matrix<int, 3, 1>;
using Vec4i = Eigen::Matrix<int, 4, 1>;
using VecXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using MatXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using MatX3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;

using SE3 = Sophus::SE3<number_t>;
using SO3 = Sophus::SO3<number_t>;

using SE3f = Sophus::SE3<float>;
using SO3f = Sophus::SO3<float>;

using SE3d = Sophus::SE3<double>;
using SO3d = Sophus::SO3<double>;

using AngleAxis = Eigen::AngleAxisd;
using Quat = Eigen::Quaternion<double, Eigen::AutoAlign>;

static const number_t eps = 1e-4f;

} // namespace xivo
