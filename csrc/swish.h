#include <torch/types.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <c10/util/Half.h>
#define GLOBAL_INLINE __forceinline__ __host__ __device__
#else
#include <cmath>
#define GLOBAL_INLINE __inline__
#endif

// TODO: Try and convert these to lambda functions
template <typename scalar_t>
GLOBAL_INLINE
void swish_fwd_func(scalar_t &out, const scalar_t &inp) {
  if (inp < 0) {
    const scalar_t e = exp(inp);
    out = inp * e / (scalar_t(1.0) + e);
  } else {
    out = inp / (scalar_t(1.0) + exp(-inp));
  }
};

template <typename scalar_t>
GLOBAL_INLINE
void swish_bwd_func(scalar_t &grad_inp, const scalar_t &inp, const scalar_t &grad_out) {
  scalar_t grad;
  if (inp < 0) {
    const scalar_t e = exp(inp);
    grad = (scalar_t(1.0) + inp / (scalar_t(1.0) + e)) * e / (scalar_t(1.0) + e);
  } else {
    const scalar_t e = exp(-inp);
    grad = (scalar_t(1.0) + inp * e / (scalar_t(1.0) + e)) / (scalar_t(1.0) + e);
  }
  grad_inp = grad_out * grad;
};

// Specialisations for Half to calculate as float
// Increases precision and also lacking certain instrinsics for Half
template <>
GLOBAL_INLINE
void swish_fwd_func(c10::Half &out, const c10::Half &inp) {
  float res;
  swish_fwd_func<float>(res, (float)inp);
  out = res;
};

template <>
GLOBAL_INLINE
void swish_bwd_func(c10::Half &grad_inp, const c10::Half &inp, const c10::Half &grad_out) {
  float res;
  swish_bwd_func<float>(res, (float)inp, (float)grad_out);
  grad_inp = res;
};