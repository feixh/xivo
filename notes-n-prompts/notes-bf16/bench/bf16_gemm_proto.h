// SUPERSEDED. The kernel that ships is xivo-bf16/common/bf16_gemm.h; this is the
// M0 prototype, kept only so gemm_shapes.cpp still reproduces the kernel-level
// table in m0-baseline.md. Its namespace is `xivo::bf16`, which the tree version
// could not keep (`xivo::bf16` is the scalar type in common/bf16.h), and its
// entry points take a `beta` instead of an `accumulate` flag. Do not extend it;
// extend the tree header and time it with kernel_api.cpp.
// bf16 GEMM with fp32 accumulation, for the dense covariance algebra of an EKF.
//
//   C(MxN) = alpha_op( A(MxK) * B(KxN) ) + beta * C
//
// A and B arrive as fp32 (or fp64) and are rounded to bfloat16 while being
// packed; every product is accumulated in fp32 by AVX512-BF16's `vdpbf16ps`,
// which multiplies two bf16 pairs and adds into an fp32 lane. So the only error
// is the input rounding (relative 2^-9 per element, round-to-nearest-even), not
// the accumulation -- the same arrangement a tensor core uses.
//
// Layouts. Everything is column-major, matching Eigen's default, and `ld*` are
// column strides.
//
// Packing. `vdpbf16ps` consumes k and k+1 together, so both operands are packed
// with pairs of k-neighbours interleaved into one 32-bit word ("VNNI layout"):
//   A panel: for each block of MR rows, for each k-pair, MR words
//            (word i = A[i, 2t] , A[i, 2t+1])
//   B panel: for each block of NR columns, for each k-pair, NR words
//            (word j = B[2t, j] , B[2t+1, j])
// The A word is broadcast to all 16 lanes; the B words are a straight vector
// load. Panels are zero-padded to whole tiles so the microkernel never has to
// mask -- zeros contribute nothing to a dot product.
//
// Author: Xiaohan Fei <hzhsfxh@gmail.com>
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX512BF16__)
#include <immintrin.h>
#define XIVO_HAS_BF16_GEMM 1
#else
#define XIVO_HAS_BF16_GEMM 0
#endif

namespace xivo {
namespace bf16 {

constexpr int MR = 6;  // rows per microkernel tile
constexpr int NR = 32; // columns per microkernel tile (2 zmm wide)

/// \brief round-to-nearest-even fp32 -> bf16, returned in the high half.
inline uint16_t rne(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  // NaN/Inf keep their exponent; the mantissa rounding below is harmless there.
  const uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

/// \brief the two bf16 of a k-pair, packed into one 32-bit word (k in the low
/// half, k+1 in the high half -- `vdpbf16ps` treats the two symmetrically).
inline uint32_t pair(float lo, float hi) {
  return static_cast<uint32_t>(rne(lo)) |
         (static_cast<uint32_t>(rne(hi)) << 16);
}

/// \brief scratch buffers, so a caller in a hot loop allocates once.
struct Workspace {
  std::vector<uint32_t> pa, pb;
  void reserve_a(size_t n) { if (pa.size() < n) pa.resize(n); }
  void reserve_b(size_t n) { if (pb.size() < n) pb.resize(n); }
};

inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

/// \brief pack A (MxK, column-major, stride lda) into MR-row panels.
template <typename T>
void pack_a(int M, int K, const T *A, int lda, uint32_t *pa) {
  const int mt = round_up(M, MR) / MR, kt = round_up(K, 2) / 2;
  for (int p = 0; p < mt; ++p) {
    uint32_t *dst = pa + static_cast<size_t>(p) * kt * MR;
    const int m0 = p * MR, mend = (m0 + MR < M) ? m0 + MR : M;
    for (int t = 0; t < kt; ++t) {
      const int k0 = 2 * t;
      const T *c0 = A + static_cast<size_t>(k0) * lda;
      const T *c1 = (k0 + 1 < K) ? A + static_cast<size_t>(k0 + 1) * lda : nullptr;
      for (int i = m0; i < mend; ++i) {
        dst[(size_t)t * MR + (i - m0)] =
            pair(static_cast<float>(c0[i]), c1 ? static_cast<float>(c1[i]) : 0.f);
      }
      for (int i = mend; i < m0 + MR; ++i) {
        dst[(size_t)t * MR + (i - m0)] = 0;
      }
    }
  }
}

/// \brief pack A given row-major-with-stride input, i.e. pack A^T from a
/// column-major buffer of A^T (KxM). Used for the very common `A^T * B` shape.
template <typename T>
void pack_a_trans(int M, int K, const T *At, int ldat, uint32_t *pa) {
  const int mt = round_up(M, MR) / MR, kt = round_up(K, 2) / 2;
  for (int p = 0; p < mt; ++p) {
    uint32_t *dst = pa + static_cast<size_t>(p) * kt * MR;
    const int m0 = p * MR, mend = (m0 + MR < M) ? m0 + MR : M;
    for (int t = 0; t < kt; ++t) {
      const int k0 = 2 * t;
      for (int i = m0; i < mend; ++i) {
        const T *col = At + static_cast<size_t>(i) * ldat;
        dst[(size_t)t * MR + (i - m0)] =
            pair(static_cast<float>(col[k0]),
                 (k0 + 1 < K) ? static_cast<float>(col[k0 + 1]) : 0.f);
      }
      for (int i = mend; i < m0 + MR; ++i) {
        dst[(size_t)t * MR + (i - m0)] = 0;
      }
    }
  }
}

/// \brief pack B (KxN, column-major, stride ldb) into NR-column panels.
template <typename T>
void pack_b(int K, int N, const T *B, int ldb, uint32_t *pb) {
  const int nt = round_up(N, NR) / NR, kt = round_up(K, 2) / 2;
  for (int q = 0; q < nt; ++q) {
    uint32_t *dst = pb + static_cast<size_t>(q) * kt * NR;
    const int n0 = q * NR, nend = (n0 + NR < N) ? n0 + NR : N;
    for (int t = 0; t < kt; ++t) {
      const int k0 = 2 * t;
      for (int j = n0; j < nend; ++j) {
        const T *col = B + static_cast<size_t>(j) * ldb;
        dst[(size_t)t * NR + (j - n0)] =
            pair(static_cast<float>(col[k0]),
                 (k0 + 1 < K) ? static_cast<float>(col[k0 + 1]) : 0.f);
      }
      for (int j = nend; j < n0 + NR; ++j) {
        dst[(size_t)t * NR + (j - n0)] = 0;
      }
    }
  }
}

#if XIVO_HAS_BF16_GEMM

/// \brief 6x32 microkernel: 12 fp32 accumulators, 2 B loads + 6 broadcasts per
/// k-pair, writing an MRxNR fp32 tile into `acc` (column-major, stride MR).
inline void kernel_6x32(int kt, const uint32_t *pa, const uint32_t *pb,
                        float *out /* NR*MR, column-major NR-major */) {
  __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
  __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
  __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
  __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
  __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
  __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
  for (int t = 0; t < kt; ++t) {
    const __m512bh b0 = (__m512bh)_mm512_loadu_si512(pb + (size_t)t * NR);
    const __m512bh b1 = (__m512bh)_mm512_loadu_si512(pb + (size_t)t * NR + 16);
    const uint32_t *a = pa + (size_t)t * MR;
    __m512bh av = (__m512bh)_mm512_set1_epi32((int)a[0]);
    c00 = _mm512_dpbf16_ps(c00, av, b0); c01 = _mm512_dpbf16_ps(c01, av, b1);
    av = (__m512bh)_mm512_set1_epi32((int)a[1]);
    c10 = _mm512_dpbf16_ps(c10, av, b0); c11 = _mm512_dpbf16_ps(c11, av, b1);
    av = (__m512bh)_mm512_set1_epi32((int)a[2]);
    c20 = _mm512_dpbf16_ps(c20, av, b0); c21 = _mm512_dpbf16_ps(c21, av, b1);
    av = (__m512bh)_mm512_set1_epi32((int)a[3]);
    c30 = _mm512_dpbf16_ps(c30, av, b0); c31 = _mm512_dpbf16_ps(c31, av, b1);
    av = (__m512bh)_mm512_set1_epi32((int)a[4]);
    c40 = _mm512_dpbf16_ps(c40, av, b0); c41 = _mm512_dpbf16_ps(c41, av, b1);
    av = (__m512bh)_mm512_set1_epi32((int)a[5]);
    c50 = _mm512_dpbf16_ps(c50, av, b0); c51 = _mm512_dpbf16_ps(c51, av, b1);
  }
  _mm512_storeu_ps(out + 0 * NR + 0, c00);  _mm512_storeu_ps(out + 0 * NR + 16, c01);
  _mm512_storeu_ps(out + 1 * NR + 0, c10);  _mm512_storeu_ps(out + 1 * NR + 16, c11);
  _mm512_storeu_ps(out + 2 * NR + 0, c20);  _mm512_storeu_ps(out + 2 * NR + 16, c21);
  _mm512_storeu_ps(out + 3 * NR + 0, c30);  _mm512_storeu_ps(out + 3 * NR + 16, c31);
  _mm512_storeu_ps(out + 4 * NR + 0, c40);  _mm512_storeu_ps(out + 4 * NR + 16, c41);
  _mm512_storeu_ps(out + 5 * NR + 0, c50);  _mm512_storeu_ps(out + 5 * NR + 16, c51);
}

/// \brief C = A*B (+C if beta==1), fp32 output, bf16 arithmetic.
/// A: MxK column-major stride lda (or, if a_trans, KxM stride lda holding A^T)
/// B: KxN column-major stride ldb.  C: MxN column-major stride ldc.
template <typename TA, typename TB, typename TC>
void gemm(int M, int N, int K, const TA *A, int lda, bool a_trans, const TB *B,
          int ldb, TC *C, int ldc, float beta, Workspace &ws) {
  const int mt = round_up(M, MR) / MR, nt = round_up(N, NR) / NR;
  const int kt = round_up(K, 2) / 2;
  ws.reserve_a((size_t)mt * kt * MR);
  ws.reserve_b((size_t)nt * kt * NR);
  if (a_trans) {
    pack_a_trans(M, K, A, lda, ws.pa.data());
  } else {
    pack_a(M, K, A, lda, ws.pa.data());
  }
  pack_b(K, N, B, ldb, ws.pb.data());

  alignas(64) float tile[MR * NR];
  for (int q = 0; q < nt; ++q) {
    const uint32_t *pbq = ws.pb.data() + (size_t)q * kt * NR;
    const int n0 = q * NR, nend = (n0 + NR < N) ? n0 + NR : N;
    for (int p = 0; p < mt; ++p) {
      const uint32_t *pap = ws.pa.data() + (size_t)p * kt * MR;
      const int m0 = p * MR, mend = (m0 + MR < M) ? m0 + MR : M;
      kernel_6x32(kt, pap, pbq, tile);
      for (int j = n0; j < nend; ++j) {
        TC *c = C + (size_t)j * ldc;
        const float *src = tile + (j - n0);
        if (beta == 0.f) {
          for (int i = m0; i < mend; ++i) c[i] = (TC)src[(i - m0) * NR];
        } else {
          for (int i = m0; i < mend; ++i) c[i] += (TC)src[(i - m0) * NR];
        }
      }
    }
  }
}

/// \brief C = A*B with A and B already packed (see pack_a / pack_b), so a
/// matrix reused across many calls is rounded and packed once.
template <typename TC>
void gemm_packed(int M, int N, int K, const uint32_t *pa, const uint32_t *pb,
                 TC *C, int ldc, float beta) {
  const int mt = round_up(M, MR) / MR, nt = round_up(N, NR) / NR;
  const int kt = round_up(K, 2) / 2;
  alignas(64) float tile[MR * NR];
  for (int q = 0; q < nt; ++q) {
    const int n0 = q * NR, nend = (n0 + NR < N) ? n0 + NR : N;
    for (int p = 0; p < mt; ++p) {
      const int m0 = p * MR, mend = (m0 + MR < M) ? m0 + MR : M;
      kernel_6x32(kt, pa + (size_t)p * kt * MR, pb + (size_t)q * kt * NR, tile);
      for (int j = n0; j < nend; ++j) {
        TC *c = C + (size_t)j * ldc;
        const float *src = tile + (j - n0);
        if (beta == 0.f) {
          for (int i = m0; i < mend; ++i) c[i] = (TC)src[(i - m0) * NR];
        } else {
          for (int i = m0; i < mend; ++i) c[i] += (TC)src[(i - m0) * NR];
        }
      }
    }
  }
}

#endif // XIVO_HAS_BF16_GEMM

} // namespace bf16
} // namespace xivo
