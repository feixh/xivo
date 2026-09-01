// Raw instruction-throughput probe: how many MACs/cycle does this box do in
// fp64 (vfmadd*pd), fp32 (vfmadd*ps) and bf16 (vdpbf16ps, AVX512-BF16)?
//
// Everything stays in registers with 16 independent accumulator chains, so the
// loop measures issue throughput, not latency and not memory.
//
// build: g++ -O3 -march=native -std=c++17 peak.cpp -o peak
#include <cstdio>
#include <cstdint>
#include <immintrin.h>
#include <chrono>

static const long ITER = 200000000L; // per-chain iterations / 16

template <class F> double timeit(const char *name, long macs_per_iter, F f) {
  auto t0 = std::chrono::steady_clock::now();
  f();
  auto t1 = std::chrono::steady_clock::now();
  double s = std::chrono::duration<double>(t1 - t0).count();
  double gflops = 2.0 * macs_per_iter / s / 1e9;
  printf("%-12s %8.3f s  %9.1f GFLOP/s\n", name, s, gflops);
  return gflops;
}

int main() {
  const long N = ITER;
  double g64, g32, gbf;
  {
    __m512d a = _mm512_set1_pd(1.0000001), b = _mm512_set1_pd(0.9999999);
    __m512d c0 = _mm512_setzero_pd(), c1 = c0, c2 = c0, c3 = c0;
    __m512d c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    g64 = timeit("fp64 fma", N * 8 * 8, [&] {
      for (long i = 0; i < N; ++i) {
        c0 = _mm512_fmadd_pd(a, b, c0); c1 = _mm512_fmadd_pd(a, b, c1);
        c2 = _mm512_fmadd_pd(a, b, c2); c3 = _mm512_fmadd_pd(a, b, c3);
        c4 = _mm512_fmadd_pd(a, b, c4); c5 = _mm512_fmadd_pd(a, b, c5);
        c6 = _mm512_fmadd_pd(a, b, c6); c7 = _mm512_fmadd_pd(a, b, c7);
      }
    });
    volatile double sink = _mm512_reduce_add_pd(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
    (void)sink;
  }
  {
    __m512 a = _mm512_set1_ps(1.0000001f), b = _mm512_set1_ps(0.9999999f);
    __m512 c0 = _mm512_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    __m512 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    g32 = timeit("fp32 fma", N * 8 * 16, [&] {
      for (long i = 0; i < N; ++i) {
        c0 = _mm512_fmadd_ps(a, b, c0); c1 = _mm512_fmadd_ps(a, b, c1);
        c2 = _mm512_fmadd_ps(a, b, c2); c3 = _mm512_fmadd_ps(a, b, c3);
        c4 = _mm512_fmadd_ps(a, b, c4); c5 = _mm512_fmadd_ps(a, b, c5);
        c6 = _mm512_fmadd_ps(a, b, c6); c7 = _mm512_fmadd_ps(a, b, c7);
      }
    });
    volatile float sink = _mm512_reduce_add_ps(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
    (void)sink;
  }
  {
    // one vdpbf16ps = 32 bf16 MACs into 16 fp32 accumulators
    __m512bh a = (__m512bh)_mm512_set1_epi32(0x3f803f80);
    __m512bh b = (__m512bh)_mm512_set1_epi32(0x3f803f80);
    __m512 c0 = _mm512_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    __m512 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    gbf = timeit("bf16 dpbf16", N * 8 * 32, [&] {
      for (long i = 0; i < N; ++i) {
        c0 = _mm512_dpbf16_ps(c0, a, b); c1 = _mm512_dpbf16_ps(c1, a, b);
        c2 = _mm512_dpbf16_ps(c2, a, b); c3 = _mm512_dpbf16_ps(c3, a, b);
        c4 = _mm512_dpbf16_ps(c4, a, b); c5 = _mm512_dpbf16_ps(c5, a, b);
        c6 = _mm512_dpbf16_ps(c6, a, b); c7 = _mm512_dpbf16_ps(c7, a, b);
      }
    });
    volatile float sink = _mm512_reduce_add_ps(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
    (void)sink;
  }
  printf("\nratios: fp32/fp64 = %.2fx   bf16/fp64 = %.2fx   bf16/fp32 = %.2fx\n",
         g32 / g64, gbf / g64, gbf / g32);
  return 0;
}
