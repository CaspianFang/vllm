#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "bgmv/bgmv_config.h"
#include "sgmv/sgmv.h"
#include "sgmv_flashinfer/sgmv_config.h"

namespace {

//====== utils ======

inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x)                                                        \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b)                                                         \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b)                                                         \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

//====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(cond, ...) \
  [&]() -> bool {                   \
    switch (cond) {                 \
      __VA_ARGS__                   \
      default:                      \
        return false;               \
    }                               \
  }()

#define _DISPATCH_DTYPE_CASE(enum_type, c_type_, ...) \
  case enum_type: {                                   \
    using c_type = c_type_;                           \
    return __VA_ARGS__();                             \
  }

#define _DISPATCH_DTYPE_CASES(...)                                 \
  _DISPATCH_DTYPE_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_DTYPE_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_DTYPE_CASES(__VA_ARGS__))

//====== bgmv ======

template <typename in_T, typename out_T, typename W_T>
inline bool launch_bgmv_kernel(out_T *Y, const in_T *X, const W_T *W,
                               const int64_t *lora_indices,
                               uint16_t in_features, uint16_t out_features,
                               int64_t y_offset, int64_t full_y_size,
                               int64_t batch_size, int64_t num_layers,
                               int64_t layer_idx, float scale) {
  switch (pack_u16(in_features, out_features)) {
#define CASE_ONESIDE(_in_T, _out_T, _W_T, feat_in, feat_out)                   \
  case pack_u16(feat_in, feat_out):                                            \
    bgmv_kernel<feat_in, feat_out>(Y, X, W, lora_indices, y_offset,            \
                                   full_y_size, batch_size, num_layers,        \
                                   layer_idx, scale);                          \
    break;
#define CASE(_in_T, _out_T, _W_T, narrow, wide)                                \
  CASE_ONESIDE(in_T, out_T, W_T, narrow, wide)                                 \
  CASE_ONESIDE(in_T, out_T, W_T, wide, narrow)

    FOR_BGMV_WIDE_NARROW(CASE, _, _, _)
#undef CASE
#undef CASE_ONESIDE
  default:
    return false;
  }

  return true;
}

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, float scale) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t h_in = x.size(1);
  int64_t h_out = y.size(1);
  int64_t num_layers = w.size(1);
  CHECK_EQ(w.size(3), h_in);
  CHECK_EQ(w.size(2), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  bool ok = false;
  if (h_in < 65536 && h_out < 65536) {
    // TODO: See if we can get rid of this massive nested switch
    switch (x.scalar_type()) {
    case at::ScalarType::Half:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::BFloat16:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::Float:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    default:
      break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type(), " out_dtype=", y.scalar_type());
}

void dispatch_bgmv_low_level(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                             torch::Tensor indicies, int64_t layer_idx,
                             float scale, int64_t h_in, int64_t h_out,
                             int64_t y_offset) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t num_layers = w.size(1);
  int64_t full_y_size = y.size(1);
  CHECK_EQ(w.size(3), h_in);
  CHECK_EQ(w.size(2), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  bool ok = false;
  if (h_in < 65536 && h_out < 65536) {
    // TODO: See if we can get rid of this massive nested switch
    switch (x.scalar_type()) {
    case at::ScalarType::Half:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::BFloat16:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::Float:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    default:
      break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type(), " out_dtype=", y.scalar_type());
}

//====== sgmv ======

void dispatch_sgmv_cutlass(torch::Tensor y, torch::Tensor x,
                           torch::Tensor w_ptr, torch::Tensor s,
                           torch::Tensor tmp, int layer_idx) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(tmp);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, tmp);

  int num_problems = s.size(0) - 1;
  int d_in = x.size(1);
  int d_out = y.size(1);
  CHECK_EQ(tmp.size(0), static_cast<int64_t>(sgmv_tmp_size(num_problems)));
  bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&] {
    return sgmv<c_type>((c_type*)y.data_ptr(), (c_type*)x.data_ptr(),
                        (c_type**)w_ptr.data_ptr(), s.data_ptr<int32_t>(),
                        tmp.data_ptr<uint8_t>(), num_problems, d_in, d_out,
                        layer_idx);
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type());
}

void dispatch_sgmv_shrink(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                          torch::Tensor s, torch::Tensor tmp, int layer_idx) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(tmp);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, tmp);

  uint32_t num_problems = s.size(0) - 1;
  uint32_t d_in = x.size(1);
  uint32_t d_out = y.size(1);
  CHECK_EQ(tmp.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(tmp.size(0), 8 * 1024 * 1024);

#define CASE(_T, D_OUT)                                    \
  case D_OUT:                                              \
    return sgmv_shrink<c_type, D_OUT>(                     \
        (c_type*)y.data_ptr(), (c_type*)x.data_ptr(),      \
        (c_type**)w_ptr.data_ptr(), s.data_ptr<int32_t>(), \
        tmp.data_ptr<uint8_t>(), num_problems, d_in, layer_idx);

  bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&] {
    switch (d_out) {
      FOR_SGMV_NARROW(CASE, c_type);
      default:
        return false;
    }
  });

#undef CASE
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type(),
              " d_out=", d_out);
}

} // namespace

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dispatch_bgmv", &dispatch_bgmv, "dispatch_bgmv");
  m.def("dispatch_bgmv_low_level", &dispatch_bgmv_low_level,
        "dispatch_bgmv_low_level");

  m.def("sgmv_cutlass", &dispatch_sgmv_cutlass, "");
  m.def("sgmv_cutlass_tmp_size", &sgmv_tmp_size, "");
  m.def("sgmv_shrink", &dispatch_sgmv_shrink, "");
}
