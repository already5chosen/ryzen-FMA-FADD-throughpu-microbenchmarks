#include <stdint.h>
#include <windows.h>
#include <immintrin.h>

uint64_t t128_fma(float a, float b, int nIter, float* dummy)
{
  __m128 aa = _mm_broadcast_ss(&a);
  __m128 bb = _mm_broadcast_ss(&b);
  __m128 acc0  = _mm_set_ps(0.0f, 1.0f, 2.0f, 3.0);
  __m128 acc1  = _mm_set_ps(0.1f, 0.2f, 0.3f, 0.4);
  __m128 acc2  = _mm_add_ps(acc1, acc0);
  __m128 acc3  = _mm_add_ps(acc2, acc0);
  __m128 acc4  = _mm_add_ps(acc3, acc0);
  __m128 acc5  = _mm_add_ps(acc4, acc0);
  __m128 acc6  = _mm_add_ps(acc5, acc0);
  __m128 acc7  = _mm_add_ps(acc6, acc0);
  __m128 acc8  = _mm_add_ps(acc7, acc0);
  __m128 acc9  = _mm_add_ps(acc8, acc0);
  __m128 acc10 = _mm_add_ps(acc9, acc0); // according to Agner, 10 will suffice
  __m128 acc11 = _mm_add_ps(acc10, acc0);
  int k = nIter;
  uint64_t t0 = __rdtsc();
  do {
    acc0 = _mm_fmadd_ps(aa, bb, acc0 );
    acc1 = _mm_fmadd_ps(aa, bb, acc1 );
    acc2 = _mm_fmadd_ps(aa, bb, acc2 );
    acc3 = _mm_fmadd_ps(aa, bb, acc3 );
    acc4 = _mm_fmadd_ps(aa, bb, acc4 );
    acc5 = _mm_fmadd_ps(aa, bb, acc5 );
    acc6 = _mm_fmadd_ps(aa, bb, acc6 );
    acc7 = _mm_fmadd_ps(aa, bb, acc7 );
    acc8 = _mm_fmadd_ps(aa, bb, acc8 );
    acc9 = _mm_fmadd_ps(aa, bb, acc9 );
    acc10= _mm_fmadd_ps(aa, bb, acc10);
    acc11= _mm_fmadd_ps(aa, bb, acc11);
  } while (--k);
  acc0 = _mm_xor_ps(acc0, acc6 );
  acc1 = _mm_xor_ps(acc1, acc7 );
  acc2 = _mm_xor_ps(acc2, acc8 );
  acc3 = _mm_xor_ps(acc3, acc9 );
  acc4 = _mm_xor_ps(acc4, acc10);
  acc5 = _mm_xor_ps(acc5, acc11);

  acc0 = _mm_xor_ps(acc0, acc3 );
  acc1 = _mm_xor_ps(acc1, acc4 );
  acc2 = _mm_xor_ps(acc2, acc5 );

  acc0 = _mm_xor_ps(acc0, acc1 );
  acc0 = _mm_xor_ps(acc0, acc2 );

  if (dummy)
    _mm_storeu_ps(dummy, acc0);

  uint64_t t1 = __rdtsc();

  return (t1-t0)*1000/nIter/12;
}

