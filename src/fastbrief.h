//
// Created by feixh on 9/6/15.
//
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// this is a fast version of brief used in DBoW2's Vocabulary
// since the original version used boost::dynamic_bitset as the underlying
// data structure of descriptors, manipulation of them is very slow
// In this version, gcc intrinsics are used and 10x speedup is achieved

#ifndef DBOW2_EXTENSION_FASTBRIEF_H
#define DBOW2_EXTENSION_FASTBRIEF_H

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>
#include <string>

#include "DBoW2/FClass.h"

namespace xivo {

/// Functions to manipulate BRIEF descriptors
#define BRIEF_BYTES 32
class FastBrief: protected DBoW2::FClass {
public:

//   typedef uint64_t TDescriptor __attribute__ ((vector_size (BRIEF_BYTES)));
  typedef uint64_t * TDescriptor;
  typedef const TDescriptor * pDescriptor;

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors,
                        TDescriptor &mean);

  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b);

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);

  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<TDescriptor> &descriptors,
                       cv::Mat &mat);

};

}

#endif //DBOW2_EXTENSION_FASTBRIEF_H
