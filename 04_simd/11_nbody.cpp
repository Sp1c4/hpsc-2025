#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __mmask16 mask = ~(__mmask16)(1 << i);
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 xjvec = _mm512_load_ps(x);
    __m512 rxvec = _mm512_sub_ps(xivec,xjvec);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 yjvec = _mm512_load_ps(y);
    __m512 ryvec = _mm512_sub_ps(yivec,yjvec);
    __m512 r2vec = _mm512_add_ps(_mm512_mul_ps(rxvec,rxvec),_mm512_mul_ps(ryvec,ryvec));
    __m512 sqrt = _mm512_maskz_rsqrt14_ps(mask,r2vec);
    __m512 mvec = _mm512_load_ps(m);
    __m512 xm = _mm512_maskz_mul_ps(mask,rxvec,mvec);
    __m512 ym = _mm512_maskz_mul_ps(mask,ryvec,mvec);
    __m512 xsub = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(xm,sqrt),sqrt),sqrt);
    __m512 ysub = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(ym,sqrt),sqrt),sqrt);
    fx[i] -= _mm512_reduce_add_ps(xsub);
    fy[i] -= _mm512_reduce_add_ps(ysub);

    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
