//
// Created by feixh on 9/6/15.
// Adapted to XIVO by stsuei Jan 2022.
//
// stl
#include <vector>
#include <string>
#include <sstream>

// corvis
#include "fastbrief.h"

using namespace std;

namespace xivo {
// --------------------------------------------------------------------------

void FastBrief::meanValue(const std::vector<FastBrief::pDescriptor> &descriptors,
                          FastBrief::TDescriptor &mean)
{
  // initialize
  // (this used to be `mean = new uint64_t[4]` followed by
  //  `memset(&mean, 0, 32)`, which wrote 32 bytes over the caller's 8-byte
  //  pointer variable -- nulling the pointer it had just been handed -- and
  //  leaked the array; `mean` owning its own storage removes both faults)
  mean.fill(0);

  if(descriptors.empty()) return;

  const int N2 = descriptors.size() / 2;
  const int L = (BRIEF_BYTES << 3);

  vector<int> counters(L, 0);

  vector<FastBrief::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it) {
    // v[ p >> 6 ] & ((unsigned long) 1 << ( i & ( (i << 6)-1) )
    const FastBrief::TDescriptor &desc = **it;
    for(int i = 0; i < L; ++i) {
      if ( desc[ i >> 6 ] & ((uint64_t)1 << ( i & ( (i<<6)-1 ) ) ) ) {
        ++counters[i];
      }
    }
  }

  for(int i = 0; i < L; ++i) {
    if(counters[i] > N2) {
      mean[ i >> 6 ] |= ((uint64_t)1 << ( i & ( (i<<6)-1 ) ) );
    }
  }

}

// --------------------------------------------------------------------------

double FastBrief::distance(const FastBrief::TDescriptor &a, const FastBrief::TDescriptor &b)
{

#ifndef USE_ANDROID

#if BRIEF_BYTES == 64
  return    __builtin_popcountl(a[0] ^ b[0])
            + __builtin_popcountl(a[1] ^ b[1])
            + __builtin_popcountl(a[2] ^ b[2])
            + __builtin_popcountl(a[3] ^ b[3])
            + __builtin_popcountl(a[4] ^ b[4])
            + __builtin_popcountl(a[5] ^ b[5])
            + __builtin_popcountl(a[6] ^ b[6])
            + __builtin_popcountl(a[7] ^ b[7]);
#else
  return    __builtin_popcountl(a[0] ^ b[0])
            + __builtin_popcountl(a[1] ^ b[1])
            + __builtin_popcountl(a[2] ^ b[2])
            + __builtin_popcountl(a[3] ^ b[3]);
#endif

#else

#if BRIEF_BYTES == 64
  return    __builtin_popcountll(a[0] ^ b[0])
            + __builtin_popcountll(a[1] ^ b[1])
            + __builtin_popcountll(a[2] ^ b[2])
            + __builtin_popcountll(a[3] ^ b[3])
            + __builtin_popcountll(a[4] ^ b[4])
            + __builtin_popcountll(a[5] ^ b[5])
            + __builtin_popcountll(a[6] ^ b[6])
            + __builtin_popcountll(a[7] ^ b[7]);
#else
  return    __builtin_popcountll(a[0] ^ b[0])
            + __builtin_popcountll(a[1] ^ b[1])
            + __builtin_popcountll(a[2] ^ b[2])
            + __builtin_popcountll(a[3] ^ b[3]);
#endif

#endif
}

// --------------------------------------------------------------------------

std::string FastBrief::toString(const FastBrief::TDescriptor &a)
{
  stringstream ss;
  for(int i = 0; i < (BRIEF_BYTES >> 3); ++i) {
    ss << a[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------

void FastBrief::fromString(FastBrief::TDescriptor &a, const std::string &s)
{
  stringstream ss(s);
  for(int i = 0; i < (BRIEF_BYTES >> 3); ++i) {
    ss >> a[i];
  }
}

// --------------------------------------------------------------------------

void FastBrief::toMat32F(const std::vector<TDescriptor> &descriptors, cv::Mat &mat)
{
//        if(descriptors.empty())
//        {
//            mat.release();
//            return;
//        }
//
//        const int N = descriptors.size();
//        const int L = 64;
//
//        mat.create(N, L, CV_8UC1 );
//
//        for(int i = 0; i < N; ++i)
//        {
//            const TDescriptor& desc = descriptors[i];
//            float *p = mat.ptr<float>(i);
//            for(int j = 0; j < L; ++j, ++p)
//            {
//                *p = (desc[j] ? 1 : 0);
//            }
//        }
}

// --------------------------------------------------------------------------

}
