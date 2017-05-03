#include <stdint.h>
#include <windows.h>
#include <immintrin.h>

uint64_t t256_fma(float a, float b, int nIter, float* dummy)
{
  __m256 aa = _mm256_broadcast_ss(&a);
  __m256 bb = _mm256_broadcast_ss(&b);
  __m256 acc0  = _mm256_set_ps(0.0f, 1.0f, 2.0f, 3.0, 4.0f, 5.0f, 6.0f, 6.0);
  __m256 acc1  = _mm256_set_ps(0.1f, 0.2f, 0.3f, 0.4, 0.5f, 0.6f, 0.7f, 0.8);
  __m256 acc2  = _mm256_add_ps(acc1, acc0);
  __m256 acc3  = _mm256_add_ps(acc2, acc0);
  __m256 acc4  = _mm256_add_ps(acc3, acc0);
  __m256 acc5  = _mm256_add_ps(acc4, acc0);
  __m256 acc6  = _mm256_add_ps(acc5, acc0);
  __m256 acc7  = _mm256_add_ps(acc6, acc0);
  __m256 acc8  = _mm256_add_ps(acc7, acc0);
  __m256 acc9  = _mm256_add_ps(acc8, acc0);
  __m256 acc10 = _mm256_add_ps(acc9, acc0); // according to Agner, 10 will suffice
  __m256 acc11 = _mm256_add_ps(acc10, acc0);
  int k = nIter;
  uint64_t t0 = __rdtsc();
  do {
    acc0 = _mm256_fmadd_ps(aa, bb, acc0 );
    acc1 = _mm256_fmadd_ps(aa, bb, acc1 );
    acc2 = _mm256_fmadd_ps(aa, bb, acc2 );
    acc3 = _mm256_fmadd_ps(aa, bb, acc3 );
    acc4 = _mm256_fmadd_ps(aa, bb, acc4 );
    acc5 = _mm256_fmadd_ps(aa, bb, acc5 );
    acc6 = _mm256_fmadd_ps(aa, bb, acc6 );
    acc7 = _mm256_fmadd_ps(aa, bb, acc7 );
    acc8 = _mm256_fmadd_ps(aa, bb, acc8 );
    acc9 = _mm256_fmadd_ps(aa, bb, acc9 );
    acc10= _mm256_fmadd_ps(aa, bb, acc10);
    acc11= _mm256_fmadd_ps(aa, bb, acc11);
  } while (--k);
  acc0 = _mm256_xor_ps(acc0, acc6 );
  acc1 = _mm256_xor_ps(acc1, acc7 );
  acc2 = _mm256_xor_ps(acc2, acc8 );
  acc3 = _mm256_xor_ps(acc3, acc9 );
  acc4 = _mm256_xor_ps(acc4, acc10);
  acc5 = _mm256_xor_ps(acc5, acc11);

  acc0 = _mm256_xor_ps(acc0, acc3 );
  acc1 = _mm256_xor_ps(acc1, acc4 );
  acc2 = _mm256_xor_ps(acc2, acc5 );

  acc0 = _mm256_xor_ps(acc0, acc1 );
  acc0 = _mm256_xor_ps(acc0, acc2 );

  if (dummy)
    _mm256_storeu_ps(dummy, acc0);

  uint64_t t1 = __rdtsc();

  return (t1-t0)*1000/nIter/12;
}

