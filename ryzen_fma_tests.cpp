#include <stdint.h>
#include <stdio.h>
#include <algorithm>

extern "C" uint64_t t128_fma(float a, float b, int cnt, float* dummy);
extern "C" uint64_t t256_fma(float a, float b, int cnt, float* dummy);
extern "C" uint64_t t128_fma1_add1(float a, float b, int cnt, float* dummy);
extern "C" uint64_t t128_fma1_add1_m9a5(float a, float b, int cnt, float* dummy);
extern "C" uint64_t t128_fma2_add1(float a, float b, int cnt, float* dummy);
extern "C" uint64_t t256_fma1_add1(float a, float b, int cnt, float* dummy);
extern "C" uint64_t t256_fma2_add1(float a, float b, int cnt, float* dummy);



static void run_test(const char* name, uint64_t (*uut)(float a, float b, int nIter, float* dummy))
{
  const int N_ITER     = 11;
  const int LOOP_COUNT = 100000;
  uint64_t resarr[N_ITER];
  for (int it = 0; it < N_ITER; ++it)
    resarr[it] = uut(1.0f, 1.0f, LOOP_COUNT, 0);
  std::nth_element(resarr, resarr + N_ITER/2, resarr + N_ITER);
  double res = resarr[N_ITER/2]* 1e-3; // median;
  printf("%-50s: %8.3f clocks\n", name, res);
}

int main()
{
  run_test("FMA128 * 1", t128_fma);
  run_test("FMA256 * 1", t256_fma);
  run_test("FMA128 * 1 + FADD128 * 1 (10+4 accum)", t128_fma1_add1);
  run_test("FMA128 * 1 + FADD128 * 1 (9+5 accum)",  t128_fma1_add1_m9a5);
  run_test("FMA256 * 1 + FADD256 * 1", t256_fma1_add1);
  run_test("FMA128 * 2 + FADD128 * 1", t128_fma2_add1);
  run_test("FMA256 * 2 + FADD256 * 1", t256_fma2_add1);
  return 0;
}