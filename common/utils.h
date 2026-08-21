// Utilities including timer, file IO, manipulation JSON files, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "alias.h"
#include "traits.h"

// stl
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
// 3rdparty
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "json/json.h"

#include "timer.h"

namespace xivo {

template <typename T> using own = T;

/// \brief colorful texts for terminal
struct TermColor {
  static const std::string red;
  static const std::string green;
  static const std::string blue;
  static const std::string cyan;
  static const std::string yellow;
  static const std::string magenta;
  static const std::string gray;
  static const std::string white;
  static const std::string bold;
  static const std::string end;
  static const std::string endl;

private:
  TermColor() = delete;
};


/// \brief generate a random matrix of dimension N x M
template <int N, int M = N>
Eigen::Matrix<number_t, N, M>
RandomMatrix(number_t meanVal = 0.0, number_t stdVal = 1.0,
             std::shared_ptr<std::knuth_b> p_engine = nullptr) {

  using number_t = number_t;

  std::normal_distribution<number_t> dist(meanVal, stdVal);
  Eigen::Matrix<number_t, N, M> v;
  if (p_engine == nullptr) {
    std::default_random_engine engine;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j) {
        v(i, j) = dist(engine);
      }
  } else {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j) {
        v(i, j) = dist(*p_engine);
      }
  }
  return v;
};

/// \brief generate a random vecotr of dimension N
template <int N>
Eigen::Matrix<number_t, N, 1>
RandomVector(number_t meanVal = 0.0, number_t stdVal = 1.0,
             std::shared_ptr<std::knuth_b> p_engine = nullptr) {

  using number_t = number_t;

  std::normal_distribution<number_t> dist(meanVal, stdVal);
  Eigen::Matrix<number_t, N, 1> v;
  if (p_engine == nullptr) {
    std::default_random_engine engine;
    for (int i = 0; i < N; ++i) {
      v(i) = dist(engine);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      v(i) = dist(*p_engine);
    }
  }
  return v;
};

/// \brief detect if any compoennt of a matrix is nan
/// NOTE: use the runtime rows()/cols() rather than Rows/ColsAtCompileTime:
/// the latter are Eigen::Dynamic (-1) for dynamically sized types, which made
/// the loops below body-less, so this silently returned false for every
/// MatX/VecX it was handed.
template <typename Derived> bool anynan(const Eigen::MatrixBase<Derived> &m) {
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      if (std::isnan(m(i, j))) {
        return true;
      }
  return false;
};

/// \brief print enum value as int
template <typename Enumeration>
typename std::underlying_type<Enumeration>::type
as_integer(Enumeration const value) {
  using type = typename std::underlying_type<Enumeration>::type;
  return static_cast<type>(value);
}

/// \brief: Convert from std vector of Eigen vectors to Eigen matrix.
template <typename T, int DIM = 3>
Eigen::Matrix<T, Eigen::Dynamic, DIM> StdVectorOfEigenVectorToEigenMatrix(
    const std::vector<Eigen::Matrix<T, DIM, 1>> &v) {
  Eigen::Matrix<T, Eigen::Dynamic, DIM> out;
  out.resize(v.size(), DIM);
  for (int i = 0; i < v.size(); ++i) {
    out.row(i) = v[i];
  }
  return out;
}

/// \brief: Convert from an Eigen matrix type to a std vector of Eigen vectors.
template <typename T, int DIM = 3>
std::vector<Eigen::Matrix<T, DIM, 1>> EigenMatrixToStdVectorOfEigenVector(
    const Eigen::Matrix<T, Eigen::Dynamic, DIM> &m) {
  std::vector<Eigen::Matrix<T, DIM, 1>> out(m.rows());
  for (int i = 0; i < m.rows(); ++i)
    out[i] = m.row(i);
  return out;
};

/// \brief: Convert from a std vector of Eigen vectors to an Eigen matrix.
template <typename T, int DIM = 3>
Eigen::Matrix<T, DIM, 1>
StdVectorOfEigenVectorMean(const std::vector<Eigen::Matrix<T, DIM, 1>> &v) {
  return StdVectorOfEigenVectorToEigenMatrix(v).colwise().mean();
};

/// \brief: Compute the rotation matrix to align two vectors.
template <typename T>
Eigen::Matrix<T, 3, 3> RotationBetweenVectors(Eigen::Matrix<T, 3, 1> u,
                                              Eigen::Matrix<T, 3, 1> v) {
  return Eigen::Quaternion<T>::FromTwoVectors(u, v).toRotationMatrix();
};

inline constexpr int cube(int x) { return x * x * x; }
inline constexpr int square(int x) { return x * x; }

template <int L = 8>
std::vector<std::array<uint8_t, 3>> GenerateRandomColorMap() {
  uint8_t q = 255 / L;
  std::vector<std::array<uint8_t, 3>> cm(cube(8));
  auto generator = std::knuth_b(0);
  for (int i = 0; i < cube(L); ++i) {
    cm[i] = {q * (i % 8u), q * ((i / 8) % 8u), q * ((i / 64) % 8u)};
    std::shuffle(cm[i].begin(), cm[i].end(), generator);
  }
  std::shuffle(cm.begin() + 1, cm.end(), generator);
  return cm;
};

/// \brief: List files with the given extension.
/// \param path: Directory of files.
/// \param extension: File extension.
/// \param filenames: List of returned file names.
/// \return: True if reads the file list successfully.
bool Glob(const std::string &path, const std::string &extension,
          std::vector<std::string> &filenames);

bool Glob(const std::string &path, const std::string &extension,
          const std::string &prefix, std::vector<std::string> &filenames);

template <typename T>
T BilinearSample(const cv::Mat &img, const Eigen::Matrix<float, 2, 1> &xy) {
  int col{xy(0)};
  int row{xy(1)};
  T v1 = (row + 1 - xy(1)) * (col + 1 - xy(0)) * img.at<T>(row, col);
  T v2 = (xy(1) - row) * (col + 1 - xy(0)) * img.at<T>(row + 1, col);
  T v3 = (xy(1) - row) * (xy(0) - col) * img.at<T>(row + 1, col + 1);
  T v4 = (row + 1 - xy(1)) * (xy(0) - col) * img.at<T>(row, col + 1);
  return v1 + v2 + v3 + v4;
}

////////////////////////////////////////////////////////////////////////////////
// I/O UTILITIES
////////////////////////////////////////////////////////////////////////////////

enum class JsonMatLayout { OneDim, RowMajor, ColMajor };
/// \brief load NxM double matrix from json file.
/// \param v: json record
/// \param key: the key
/// \param layout: Whether the matrix is arranged as an one-dim array or two-dim
/// matrix.
template <typename T = number_t, int N = 3, int M = N>
Eigen::Matrix<T, N, M>
GetMatrixFromJson(const Json::Value &v, const std::string &key,
                  JsonMatLayout layout = JsonMatLayout::OneDim) {

  Eigen::Matrix<T, N, M> ret;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      if (layout == JsonMatLayout::OneDim) {
        ret(i, j) = v[key][i * M + j].asDouble();
      } else if (layout == JsonMatLayout::RowMajor) {
        ret(i, j) = v[key][i][j].asDouble();
      } else {
        ret(i, j) = v[key][j][i].asDouble();
      }
  return ret;
}

/// \brief load N-dim double vector from json file
template <typename T = number_t, int N>
Eigen::Matrix<T, N, 1> GetVectorFromJson(const Json::Value &v,
                                         const std::string &key) {

  return GetMatrixFromJson<T, N, 1>(v, key);
};

template <typename Derived>
void WriteMatrixToJson(Json::Value &d, const std::string &key,
                       const Eigen::MatrixBase<Derived> &m) {
  d[key] = Json::Value(Json::arrayValue);
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      d[key].append(Json::Value(m(i, j)));
}

/// \brief: Merge json b to json a.
void MergeJson(Json::Value &a, const Json::Value &b);
Json::Value LoadJson(const std::string &filename);
void SaveJson(const Json::Value &j, const std::string &filename);

template <typename Derived>
std::vector<typename Derived::Scalar>
Flatten(const Eigen::MatrixBase<Derived> &m) {
  // Naive implementation of flatten, can use Eigen::Map to simplify code
  int rows = m.template rows();
  int cols = m.template cols();
  std::vector<typename Derived::Scalar> out(rows * cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      out[i * cols + j] = m(i, j);
    }
  return out;
}

template <typename T> void SaveMat(std::string filename, const cv::Mat &mat) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if (!out.is_open()) {
    LOG(FATAL) << "Failed to open output file";
  }
  int rows = mat.rows;
  int cols = mat.cols;
  out.write((char *)&rows, sizeof rows);
  out.write((char *)&cols, sizeof cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      T z = mat.at<T>(i, j);
      out.write((char *)&z, sizeof z);
    }
  out.close();
}

template <typename Derived>
void WriteMatrixToFile(const std::string &filename,
                       const Eigen::MatrixBase<Derived> &m) {
  LOG(INFO) << "Writing matrix to " << filename;
  std::ofstream ostream(filename, std::ios::out);
  if (!ostream.is_open()) {
    LOG(FATAL) << "Failed to open output file";
  }
  ostream << m;
  ostream.close();
}

namespace impl {

// implementation details
template <typename T>
std::enable_if_t<std::is_same<plain<T>, std::string>::value, const char*>
StrArgPass(T &arg) {
  // std::cout << __PRETTY_FUNCTION__ << std::endl;
  return arg.c_str();
}

template <typename T>
std::enable_if_t<!std::is_same<plain<T>, std::string>::value, T>
StrArgPass(T&& arg) {
  // std::cout << __PRETTY_FUNCTION__ << std::endl;
  return arg;
}

} // namespace impl


constexpr int kStrBufSize = 512;
template <typename... Args>
std::string StrFormat(const char *format, Args... args) {
  char buf[kStrBufSize];
  if (snprintf(&buf[0], kStrBufSize, format, impl::StrArgPass(args)...) < 0) {
    LOG(FATAL) << "failed to format string!!!";
  }
  return buf;
}

std::vector<std::string> StrSplit(const std::string &str, char delimiter);

/*
template<typename T>
void col_major_Mat33_copy(Eigen::Matrix<T, 3, 3> &out, number_t *input) {
  out(0,0) = input[0];
  out(1,0) = input[1];
  out(2,0) = input[2];
  out(0,1) = input[3];
  out(1,1) = input[4];
  out(2,1) = input[5];
  out(0,2) = input[6];
  out(1,2) = input[7];
  out(2,2) = input[8];
}
*/

} // namespace xivo
